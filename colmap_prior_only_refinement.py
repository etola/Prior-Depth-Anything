#!/usr/bin/env python3
"""
COLMAP Prior-Only Depth Completion Script
=========================================

This script processes COLMAP reconstructions to create depth completions using 
ONLY COLMAP 3D points as priors (no VGGT geometric depths) with Prior-Depth-Anything.

Usage:
    python colmap_prior_only_refinement.py -s <scene_folder> -o <output_folder> [--prior_extra <extra_reconstruction_path>]

Where:
    - scene_folder/images contains input images
    - scene_folder/sparse contains COLMAP reconstruction
    - Output point clouds are saved to scene_folder/output_folder/
    - --prior_extra (optional): Path to extra COLMAP reconstruction to augment 3D points (must have same images and camera calibrations)
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
from prior_depth_anything import PriorDepthAnything

# COLMAP utilities
from colmap_utils import ColmapReconstruction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete sparse COLMAP depth using Prior Depth Anything (no VGGT)"
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
        "--prior_extra", 
        type=str, 
        default=None,
        help="Optional path to extra COLMAP reconstruction to augment 3D points (must have same images and camera calibrations as main reconstruction)"
    )
    
    return parser.parse_args()


class ColmapPriorOnlyRefinement:
    """Main class for COLMAP prior-only depth completion pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.target_size = args.target_size
        
        # Initialize scene paths
        self.scene_folder = Path(args.scene_folder)
        self.images_folder = self.scene_folder / "images"
        self.sparse_folder = self.scene_folder / "sparse"
        self.output_folder = self.scene_folder / args.output_folder
        
        # Validate input paths
        self._validate_paths()
        
        # Create output directory
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        # Initialize Prior Depth Anything model
        self._init_model()
        
        # Load COLMAP reconstruction
        self._load_colmap_reconstruction()
        
        # Load extra COLMAP reconstruction if provided
        self.extra_reconstruction = None
        if args.prior_extra:
            self._load_extra_colmap_reconstruction()
        
    def _validate_paths(self):
        """Validate that required input paths exist."""
        if not self.scene_folder.exists():
            raise FileNotFoundError(f"Scene folder not found: {self.scene_folder}")
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        if not self.sparse_folder.exists():
            raise FileNotFoundError(f"Sparse folder not found: {self.sparse_folder}")
            
    def _init_model(self):
        """Initialize Prior Depth Anything model."""
        print("Initializing Prior Depth Anything...")
        
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
        
    def _load_extra_colmap_reconstruction(self):
        """Load extra COLMAP reconstruction for augmenting 3D points."""
        extra_path = Path(self.args.prior_extra)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra reconstruction path not found: {extra_path}")
            
        print(f"Loading extra COLMAP reconstruction from {extra_path}")
        self.extra_reconstruction = ColmapReconstruction(str(extra_path))
        summary = self.extra_reconstruction.get_summary()
        print(f"Loaded extra reconstruction: {summary}")
        
        # Build image-to-point mappings for efficiency
        print("Building extra reconstruction image-to-3D point mappings...")
        self.extra_reconstruction._ensure_image_point_maps()
        print("Extra reconstruction 3D point mappings ready!")
        
        
    def resize_and_scale_calibration(self, camera, original_size, target_size):
        """
        Resize image and scale camera calibration.
        
        Args:
            camera: COLMAP camera object
            original_size: (width, height) of original image
            target_size: target size (518) 
            
        Returns:
            new_size: (width, height) of resized image
            scaled_K: (3, 3) scaled intrinsic matrix accounting for cropping
            crop_offset_y: vertical offset due to center cropping (0 if no crop)
        """
        orig_w, orig_h = original_size
        
        # Simple resize maintaining aspect ratio, make width = target_size
        new_w = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        resized_h = round(orig_h * (new_w / orig_w) / 14) * 14
        
        # Center crop height if it's larger than target_size
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
        
    def complete_depth_with_prior_only(self, image, depth_prior, sparse_mask):
        """
        Complete depth using Prior Depth Anything with ONLY COLMAP priors (no geometric depths).
        
        Args:
            image: (H, W, 3) uint8 image
            depth_prior: (H, W) depth prior from 3D points
            sparse_mask: (H, W) mask indicating valid prior depths
            
        Returns:
            completed_depth: (H, W) completed depth map
        """
        print(f"DEBUG: Input shapes - image: {image.shape}, depth_prior: {depth_prior.shape}, sparse_mask: {sparse_mask.shape}")
        
        if not np.any(sparse_mask):
            print("Warning: No valid depth priors found, cannot complete")
            return np.zeros_like(depth_prior)
        
        print(f"DEBUG: Found {np.sum(sparse_mask)} valid sparse points")
        
        # Prepare tensors for Prior Depth Anything
        image_tensor = torch.from_numpy(image.astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(self.device)  # Use uint8 for image
        depth_prior_tensor = torch.from_numpy(depth_prior.astype(np.float32)).unsqueeze(0).to(self.device)
        sparse_mask_tensor = torch.from_numpy(sparse_mask.astype(bool)).unsqueeze(0).unsqueeze(1).to(self.device)  # Add channel dimension
        
        print(f"DEBUG: Tensor shapes - image_tensor: {image_tensor.shape}, depth_prior_tensor: {depth_prior_tensor.shape}, sparse_mask_tensor: {sparse_mask_tensor.shape}")
        
        try:
            print("DEBUG: Calling Prior Depth Anything...")
            # Use Prior Depth Anything for depth completion WITHOUT geometric depths
            # This will use the coarse depth estimator for initial prediction, then refine with priors
            result = self.prior_depth_anything(
                images=image_tensor,
                sparse_depths=depth_prior_tensor,
                sparse_masks=sparse_mask_tensor,
                # No geometric_depths parameter - Prior Depth Anything will use its own coarse estimator
            )
            print(f"DEBUG: Prior Depth Anything result shape: {result.shape}")
            
        except Exception as e:
            print(f"ERROR in Prior Depth Anything call: {e}")
            print(f"ERROR: Input tensor shapes were:")
            print(f"  - images: {image_tensor.shape}")
            print(f"  - sparse_depths: {depth_prior_tensor.shape}")
            print(f"  - sparse_masks: {sparse_mask_tensor.shape}")
            raise e
        
        # Extract completed depth and convert back to numpy
        completed_depth = result.squeeze().detach().cpu().numpy().astype(np.float32)  # Squeeze all single dimensions
        print(f"DEBUG: Final completed depth shape: {completed_depth.shape}")
        return completed_depth
        
    def generate_point_cloud(self, image, depth_map, scaled_K, camera_pose, image_id):
        """
        Generate point cloud from completed depth map.
        
        Args:
            image: (H, W, 3) image array
            depth_map: (H, W) completed depth map
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
    
    def colorize_and_save_prior_points(self, points_3d, image_array, scaled_K, camera_pose, image_name):
        """
        Colorize COLMAP 3D points with image colors and save as point cloud.
        """
        if len(points_3d) == 0:
            print(f"Warning: No prior points to save for {image_name}")
            return
            
        height, width = image_array.shape[:2]
        
        # Convert to homogeneous coordinates
        points_3d_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        
        # Transform to camera coordinates
        points_cam = (camera_pose @ points_3d_homo.T).T[:, :3]
        
        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0
        points_cam_valid = points_cam[valid_depth]
        points_3d_valid = points_3d[valid_depth]
        
        if len(points_cam_valid) == 0:
            print(f"Warning: No valid prior points in front of camera for {image_name}")
            return
        
        # Project to image plane
        points_proj = (scaled_K @ points_cam_valid.T).T
        points_proj = points_proj / points_proj[:, [2]]  # Normalize by z
        
        # Convert to pixel coordinates
        u = np.round(points_proj[:, 0]).astype(int)
        v = np.round(points_proj[:, 1]).astype(int)
        
        # Filter points within image bounds
        valid_pixels = (0 <= u) & (u < width) & (0 <= v) & (v < height)
        
        u_valid = u[valid_pixels]
        v_valid = v[valid_pixels]
        points_3d_final = points_3d_valid[valid_pixels]
        
        if len(points_3d_final) == 0:
            print(f"Warning: No prior points project within image bounds for {image_name}")
            return
        
        # Sample colors from image
        colors = image_array[v_valid, u_valid] / 255.0  # Normalize to [0, 1]
        
        # Save prior point cloud
        prior_filename = f"prior-{Path(image_name).stem}.ply"
        prior_output_path = self.output_folder / prior_filename
        self.save_point_cloud(points_3d_final, colors, prior_output_path)
        
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
        
        # Resize image
        orig_w, orig_h = original_size
        resized_w = self.target_size
        resized_h = round(orig_h * (resized_w / orig_w) / 14) * 14
        
        print(f"DEBUG: Original size: {original_size}, Resized dimensions: {resized_w}x{resized_h}")
        
        image_resized = image_pil.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        
        # Center crop if needed
        if resized_h > self.target_size:
            start_y = (resized_h - self.target_size) // 2
            image_resized = image_resized.crop((0, start_y, resized_w, start_y + self.target_size))
            print(f"DEBUG: Applied center crop from y={start_y} to y={start_y + self.target_size}")
        
        image_array = np.array(image_resized)
        print(f"DEBUG: Final image array shape: {image_array.shape}")
        print(f"DEBUG: New size from calibration: {new_size}")
        
        # Get filtered 3D points from main reconstruction
        points_3d, points_2d, point_ids = self.reconstruction.get_visible_3d_points(image_id, self.args.min_track_length)
        print(f"Found {len(points_3d)} 3D points from main reconstruction with track length >= {self.args.min_track_length}")
        
        # Augment with points from extra reconstruction if available
        # Since images and camera calibrations are exactly the same, we can use the same image_id
        if self.extra_reconstruction is not None:
            if self.extra_reconstruction.has_image(image_id):
                extra_points_3d, extra_points_2d, extra_point_ids = self.extra_reconstruction.get_visible_3d_points(
                    image_id, self.args.min_track_length
                )
                print(f"Found {len(extra_points_3d)} additional 3D points from extra reconstruction")
                
                # Combine points from both reconstructions
                if len(extra_points_3d) > 0:
                    points_3d = np.vstack([points_3d, extra_points_3d])
                    points_2d = np.vstack([points_2d, extra_points_2d])
                    point_ids = np.concatenate([point_ids, extra_point_ids])
                    print(f"Total augmented points: {len(points_3d)}")
            else:
                print(f"Warning: Image ID {image_id} not found in extra reconstruction")
        
        # Save colorized prior point cloud
        self.colorize_and_save_prior_points(points_3d, image_array, scaled_K, camera_pose, image_name)
        
        # Project 3D points to create depth prior
        depth_prior, sparse_mask = self.project_3d_points_to_depth_prior(
            points_3d, camera_pose, scaled_K, new_size
        )
        
        print(f"DEBUG: After projection - depth_prior shape: {depth_prior.shape}, sparse_mask shape: {sparse_mask.shape}")
        print(f"Depth prior has {np.sum(sparse_mask)} valid pixels")
        print(f"DEBUG: Image array shape before completion: {image_array.shape}")
        
        # Complete depth using only COLMAP priors
        completed_depth = self.complete_depth_with_prior_only(
            image_array, depth_prior, sparse_mask
        )
        
        # Generate point cloud from completed depth
        points_3d_completed, colors = self.generate_point_cloud(
            image_array, completed_depth, scaled_K, camera_pose, image_id
        )
        
        # Save completed point cloud
        output_filename = f"completed_{image_id:06d}_{Path(image_name).stem}.ply"
        output_path = self.output_folder / output_filename
        self.save_point_cloud(points_3d_completed, colors, output_path)
        
    def run(self):
        """Run the complete pipeline on all frames in the reconstruction."""
        print(f"Processing {self.reconstruction.get_num_images()} images...")
        
        image_ids = self.reconstruction.get_all_image_ids()
        
        counter = 0        
        for image_id in tqdm(image_ids, desc="Processing frames"):
            self.process_frame(image_id)
                
            # counter += 1
            # if counter > 0:
            #     break
                
        print(f"\nProcessing complete! Point clouds saved to: {self.output_folder}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run the refinement pipeline
    pipeline = ColmapPriorOnlyRefinement(args)
    pipeline.run()


if __name__ == "__main__":
    main()
