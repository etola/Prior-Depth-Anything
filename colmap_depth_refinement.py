#!/usr/bin/env python3
"""
COLMAP-Based Depth Refinement Script
===================================

This script processes COLMAP reconstructions to create depth priors for refining 
VGGT depth estimates using Prior-Depth-Anything.

Usage:
    python colmap_depth_refinement.py -s <scene_folder> -o <output_folder>

Where:
    - scene_folder/images contains input images
    - scene_folder/sparse contains COLMAP reconstruction
    - Output point clouds are saved to scene_folder/output_folder/
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import open3d as o3d

# Prior Depth Anything imports
from prior_depth_anything.plugin import PriorDARefiner
from prior_depth_anything.utils import depth2disparity, disparity2depth

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map

# COLMAP utilities
from colmap_utils import ColmapReconstruction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Refine VGGT depth estimates using COLMAP 3D points as priors"
    )
    parser.add_argument(
        "-s", "--scene_folder", 
        type=str, 
        required=True,
        help="Path to scene folder containing images/ and sparse/ subdirectories"
    )
    parser.add_argument(
        "-o", "--output_folder", 
        type=str, 
        required=True,
        help="Output folder name (will be created under scene_folder)"
    )
    parser.add_argument(
        "--min_track_length", 
        type=int, 
        default=3,
        help="Minimum track length for 3D points to be used as priors (default: 3)"
    )
    parser.add_argument(
        "--target_size", 
        type=int, 
        default=518,
        help="Target image size for processing (default: 518)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--K", 
        type=int, 
        default=5,
        help="K parameter for KNN alignment in depth completion (default: 5)"
    )
    
    return parser.parse_args()


class ColmapDepthRefinement:
    """Main class for COLMAP-based depth refinement pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.target_size = args.target_size
        self.K = getattr(args, 'K', 5)  # KNN parameter for depth completion
        
        # Initialize scene paths
        self.scene_folder = Path(args.scene_folder)
        self.images_folder = self.scene_folder / "images"
        self.sparse_folder = self.scene_folder / "sparse"
        self.output_folder = self.scene_folder / args.output_folder
        
        # Validate input paths
        self._validate_paths()
        
        # Create output directory
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        # Initialize models
        self._init_models()
        
        # Load COLMAP reconstruction
        self._load_colmap_reconstruction()
        
    def _validate_paths(self):
        """Validate that required input paths exist."""
        if not self.scene_folder.exists():
            raise FileNotFoundError(f"Scene folder not found: {self.scene_folder}")
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        if not self.sparse_folder.exists():
            raise FileNotFoundError(f"Sparse folder not found: {self.sparse_folder}")
            
    def _init_models(self):
        """Initialize VGGT and Prior Depth Anything models."""
        print("Initializing VGGT model...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.dtype = dtype
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        
        print("Initializing Prior Depth Anything...")
        from prior_depth_anything import PriorDepthAnything
        
        # Initialize PriorDepthAnything which handles model downloads automatically
        self.prior_depth_anything = PriorDepthAnything(
            device=str(self.device),
            coarse_only=False  # We want both coarse and fine processing
        )
        
    def _load_colmap_reconstruction(self):
        """Load COLMAP reconstruction."""
        print(f"Loading COLMAP reconstruction from {self.sparse_folder}")
        self.reconstruction = ColmapReconstruction(str(self.sparse_folder))
        summary = self.reconstruction.get_summary()
        print(f"Loaded reconstruction: {summary}")
        
        # Build base image-to-point mappings for efficiency
        print("Building image-to-3D point mappings...")
        self.reconstruction._ensure_image_point_maps()
        print("3D point mappings ready!")
        
    def resize_and_scale_calibration(self, camera, original_size, target_size):
        """
        Resize image and scale camera calibration using VGGT's exact logic.
        
        Args:
            camera: COLMAP camera object
            original_size: (width, height) of original image
            target_size: target size (518) used by VGGT
            
        Returns:
            new_size: (width, height) of resized image (matches VGGT)
            scaled_K: (3, 3) scaled intrinsic matrix accounting for cropping
            crop_offset_y: vertical offset due to center cropping (0 if no crop)
        """
        orig_w, orig_h = original_size
        
        # Use VGGT's exact resizing logic from load_and_preprocess_images
        # Step 1: Resize with width = target_size (518px)
        new_w = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        resized_h = round(orig_h * (new_w / orig_w) / 14) * 14
        
        # Step 2: Center crop height if it's larger than target_size (VGGT behavior)
        crop_offset_y = 0
        if resized_h > target_size:
            crop_offset_y = (resized_h - target_size) // 2
            final_h = target_size
        else:
            final_h = resized_h
        
        # Scale intrinsic matrix accounting for resize and crop
        K = camera.calibration_matrix()
        scale_x = new_w / orig_w
        scale_y = resized_h / orig_h  # Use the full resized height for scaling
        
        scaled_K = K.copy()
        scaled_K[0, 0] *= scale_x  # fx
        scaled_K[1, 1] *= scale_y  # fy  
        scaled_K[0, 2] *= scale_x  # cx
        scaled_K[1, 2] = scaled_K[1, 2] * scale_y - crop_offset_y  # cy adjusted for crop
        
        return (new_w, final_h), scaled_K, crop_offset_y
        
    def project_3d_points_to_depth_prior(self, points_3d, camera_pose, scaled_K, image_size):
        """
        Project 3D points to create depth prior map.
        
        Args:
            points_3d: (N, 3) world coordinates
            camera_pose: 4x4 camera pose matrix (cam_from_world)
            scaled_K: (3, 3) scaled intrinsic matrix  
            image_size: (width, height) of target image
            
        Returns:
            depth_prior: (H, W) depth prior map
            sparse_mask: (H, W) mask indicating valid depth points
        """
        if len(points_3d) == 0:
            return np.zeros(image_size[::-1]), np.zeros(image_size[::-1], dtype=bool)
            
        # Convert to homogeneous coordinates
        points_3d_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        
        # Transform to camera coordinates
        points_cam = (camera_pose @ points_3d_homo.T).T[:, :3]
        
        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0
        points_cam = points_cam[valid_depth]
        
        if len(points_cam) == 0:
            return np.zeros(image_size[::-1]), np.zeros(image_size[::-1], dtype=bool)
        
        # Project to image plane
        points_proj = (scaled_K @ points_cam.T).T
        points_proj = points_proj / points_proj[:, [2]]  # Normalize by z
        
        # Convert to pixel coordinates
        u = np.round(points_proj[:, 0]).astype(int)
        v = np.round(points_proj[:, 1]).astype(int)
        depths = points_cam[:, 2]
        
        # Filter points within image bounds
        width, height = image_size
        valid_pixels = (0 <= u) & (u < width) & (0 <= v) & (v < height)
        
        u = u[valid_pixels]
        v = v[valid_pixels] 
        depths = depths[valid_pixels]
        
        # Create depth prior map
        depth_prior = np.zeros((height, width))
        sparse_mask = np.zeros((height, width), dtype=bool)
        
        # Handle overlapping projections by keeping closest depth
        for i in range(len(u)):
            if sparse_mask[v[i], u[i]]:
                # Keep closer depth
                if depths[i] < depth_prior[v[i], u[i]]:
                    depth_prior[v[i], u[i]] = depths[i]
            else:
                depth_prior[v[i], u[i]] = depths[i]
                sparse_mask[v[i], u[i]] = True
                
        return depth_prior, sparse_mask
        
    def estimate_depth_with_vggt(self, image_tensor):
        """
        Estimate depth using VGGT model.
        
        Args:
            image_tensor: (1, 3, H, W) preprocessed image tensor
            
        Returns:
            depth_map: (H, W) estimated depth map  
            confidence: (H, W) confidence map
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.vggt(image_tensor)
                
        depth_map = predictions['depth'].squeeze().cpu()
        confidence = predictions['depth_conf'].squeeze().cpu()
        
        return depth_map, confidence
        
    def refine_depth_with_prior(self, image, vggt_depth, vggt_confidence, depth_prior):
        """
        Refine VGGT depth using Prior Depth Anything with 3D point priors.
        
        Args:
            image: (H, W, 3) uint8 image
            vggt_depth: (H, W) VGGT depth estimate
            vggt_confidence: (H, W) VGGT confidence 
            depth_prior: (H, W) depth prior from 3D points
            
        Returns:
            refined_depth: (H, W) refined depth map
        """
        # Convert depth prior to create sparse depth for refinement
        prior_mask = depth_prior > 0
        
        if not np.any(prior_mask):
            print("Warning: No valid depth priors found, using VGGT depth only")
            return vggt_depth.numpy() if isinstance(vggt_depth, torch.Tensor) else vggt_depth
        
        # Prepare tensors for Prior Depth Anything
        # Convert to torch tensors with consistent float32 dtype and add batch dimension
        # Prepare tensors for Prior Depth Anything
        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        vggt_depth_np = vggt_depth if isinstance(vggt_depth, np.ndarray) else vggt_depth.numpy()
        vggt_depth_tensor = torch.from_numpy(vggt_depth_np.astype(np.float32)).unsqueeze(0).to(self.device)
        depth_prior_tensor = torch.from_numpy(depth_prior.astype(np.float32)).unsqueeze(0).to(self.device)
        prior_mask_tensor = torch.from_numpy(prior_mask.astype(bool)).unsqueeze(0).to(self.device)
        
        # Use Prior Depth Anything's completion.forward with correct parameters:
        # - sparse_depths: COLMAP depth priors  
        # - sparse_masks: mask for COLMAP priors
        # - geometric_depths: VGGT depth estimate
        # - ret: 'knn' for KNN alignment
        result = self.prior_depth_anything.completion.forward(
            images=image_tensor,
            sparse_depths=depth_prior_tensor,
            sparse_masks=prior_mask_tensor,
            geometric_depths=vggt_depth_tensor,
            ret='knn'
        )
        
        # Extract refined depth and convert back to numpy
        refined_depth = result.squeeze(0).cpu().numpy().astype(np.float32)
        return refined_depth
        
    def generate_point_cloud(self, image, depth_map, scaled_K, camera_pose, image_id):
        """
        Generate point cloud from refined depth map.
        
        Args:
            image: (H, W, 3) image array
            depth_map: (H, W) refined depth map
            scaled_K: (3, 3) scaled intrinsic matrix
            camera_pose: (4, 4) camera pose matrix
            image_id: COLMAP image ID for naming
            
        Returns:
            points_3d: (N, 3) 3D points in world coordinates
            colors: (N, 3) RGB colors for points
        """
        height, width = depth_map.shape
        
        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Valid depth mask
        valid_mask = depth_map > 0
        
        if not np.any(valid_mask):
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # Get valid coordinates and depths
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        depth_valid = depth_map[valid_mask]
        
        # Unproject to camera coordinates
        K_inv = np.linalg.inv(scaled_K)
        
        # Create homogeneous pixel coordinates
        pixel_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)
        
        # Unproject to camera coordinates
        cam_coords = K_inv @ pixel_coords
        cam_coords = cam_coords * depth_valid[np.newaxis, :]
        
        # Add homogeneous coordinate
        cam_coords_homo = np.concatenate([cam_coords, np.ones((1, len(depth_valid)))], axis=0)
        
        # Transform to world coordinates
        # COLMAP's cam_from_world is typically 3x4 [R|t], need to make it 4x4 for inversion
        if camera_pose.shape == (3, 4):
            # Add bottom row [0, 0, 0, 1] to make it 4x4
            bottom_row = np.array([[0, 0, 0, 1]])
            camera_pose_4x4 = np.vstack([camera_pose, bottom_row])
        else:
            camera_pose_4x4 = camera_pose
            
        world_from_cam = np.linalg.inv(camera_pose_4x4)
        world_coords = (world_from_cam @ cam_coords_homo).T[:, :3]
        
        # Get corresponding colors
        colors = image[valid_mask] / 255.0  # Normalize to [0, 1]
        
        return world_coords, colors
        
    def save_point_cloud(self, points_3d, colors, output_path):
        """Save point cloud using Open3D."""
        if len(points_3d) == 0:
            print(f"Warning: No points to save for {output_path}")
            return
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        
        # Set points (Open3D expects float64)
        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        
        # Set colors (Open3D expects float64 in range [0, 1])
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
        # Save point cloud
        success = o3d.io.write_point_cloud(str(output_path), pcd)
        
        if success:
            print(f"Saved point cloud with {len(points_3d)} points to {output_path}")
        else:
            raise RuntimeError(f"Failed to save point cloud to {output_path}")
        
    def process_frame(self, image_id):
        """
        Process a single frame through the complete pipeline.
        
        Args:
            image_id: COLMAP image ID to process
        """
        print(f"\nProcessing frame {image_id}...")
        
        # Get image information
        image_name = self.reconstruction.get_image_name(image_id)
        image_path = self.images_folder / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return
            
        # Load image
        image_pil = Image.open(image_path).convert('RGB')
        original_size = image_pil.size  # (width, height)
        
        # Get camera information
        camera = self.reconstruction.get_image_camera(image_id)
        camera_pose = self.reconstruction.get_image_cam_from_world(image_id).matrix()
        
        # Resize image and scale calibration
        new_size, scaled_K, crop_offset_y = self.resize_and_scale_calibration(
            camera, original_size, self.target_size
        )
        
        # Resize image using VGGT's approach (resize then crop)
        # First resize to full dimensions
        orig_w, orig_h = original_size
        resized_w = self.target_size
        resized_h = round(orig_h * (resized_w / orig_w) / 14) * 14
        
        image_resized = image_pil.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        
        # Then center crop if needed (matching VGGT exactly)
        if resized_h > self.target_size:
            start_y = (resized_h - self.target_size) // 2
            image_resized = image_resized.crop((0, start_y, resized_w, start_y + self.target_size))
        
        image_array = np.array(image_resized)
        
        # Get filtered 3D points
        points_3d, points_2d, point_ids = self.reconstruction.get_visible_3d_points(image_id, self.args.min_track_length)
        print(f"Found {len(points_3d)} 3D points with track length >= {self.args.min_track_length}")
        
        # Project 3D points to create depth prior
        depth_prior, sparse_mask = self.project_3d_points_to_depth_prior(
            points_3d, camera_pose, scaled_K, new_size
        )
        
        # Prepare image for VGGT using the same image we resized
        # This ensures the VGGT output matches our depth prior dimensions
        from torchvision import transforms as TF
        to_tensor = TF.ToTensor()
        image_tensor = to_tensor(image_resized).to(torch.float32).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        # Estimate depth with VGGT  
        # Run VGGT to get initial depth estimate
        vggt_depth, vggt_confidence = self.estimate_depth_with_vggt(image_tensor)
        
        # Refine depth using priors
        refined_depth = self.refine_depth_with_prior(
            image_array, vggt_depth, vggt_confidence, depth_prior
        )
        
        # Generate point cloud using the same scaled_K that accounts for cropping
        points_3d_refined, colors = self.generate_point_cloud(
            image_array, refined_depth, scaled_K, camera_pose, image_id
        )
        
        # Save point cloud
        output_filename = f"frame_{image_id:06d}_{Path(image_name).stem}.ply"
        output_path = self.output_folder / output_filename
        self.save_point_cloud(points_3d_refined, colors, output_path)
        
    def run(self):
        """Run the complete pipeline on all frames in the reconstruction."""
        print(f"Processing {self.reconstruction.get_num_images()} images...")
        
        image_ids = self.reconstruction.get_all_image_ids()
        
        for image_id in tqdm(image_ids, desc="Processing frames"):
            try:
                self.process_frame(image_id)
            except Exception as e:
                print(f"Error processing frame {image_id}: {e}")
                continue
                
        print(f"\nProcessing complete! Point clouds saved to: {self.output_folder}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run the refinement pipeline
    pipeline = ColmapDepthRefinement(args)
    pipeline.run()


if __name__ == "__main__":
    main()
