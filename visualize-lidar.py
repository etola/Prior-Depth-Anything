#!/usr/bin/env python3
"""
LiDAR Point Cloud Visualization Script
=====================================

This script loads a COLMAP reconstruction and converts LiDAR depth maps into a merged point cloud.

Usage:
    python visualize-lidar.py -s <scene_folder>

Where:
    - scene_folder/images contains calibrated images (e.g., image_00000.jpg)
    - scene_folder/sparse contains COLMAP reconstruction
    - scene_folder/lidar contains depth maps (e.g., depthmap_00001.tiff) and confidence maps (e.g., confidence_00001.tiff)
    
Note: Depth/confidence indices are 1 more than image indices (image_00000.jpg -> depthmap_00001.tiff)
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import open3d as o3d
import re
import torch
import torch.nn.functional as F

# Prior Depth Anything imports
from prior_depth_anything import PriorDepthAnything

# COLMAP utilities
from colmap_utils import ColmapReconstruction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LiDAR depth maps to point clouds using COLMAP calibration with optional depth completion. Saves individual and merged point clouds to output folder."
    )
    parser.add_argument(
        "-s", "--scene_folder", 
        type=str, 
        required=True,
        help="Path to scene folder containing images/, sparse/, and lidar/ subdirectories"
    )
    parser.add_argument(
        "-c", "--confidence_threshold", 
        type=float, 
        default=0.5,
        help="Minimum confidence threshold for LiDAR points (default: 0.5)"
    )
    parser.add_argument(
        "-o", "--output_folder", 
        type=str, 
        default="output",
        help="Output folder for point clouds (default: output)"
    )
    parser.add_argument(
        "-d", "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "-t", "--target_size", 
        type=int, 
        default=518,
        help="Target image size for processing (default: 518)"
    )
    parser.add_argument(
        "-e", "--enable_depth_completion", 
        action="store_true",
        help="Enable depth completion using Prior Depth Anything (default: False)"
    )
    
    return parser.parse_args()


class LidarPointCloudGenerator:
    """Main class for converting LiDAR depth maps to point clouds using COLMAP calibration."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.target_size = args.target_size
        
        # Initialize scene paths
        self.scene_folder = Path(args.scene_folder)
        self.images_folder = self.scene_folder / "images"
        self.sparse_folder = self.scene_folder / "sparse"
        self.lidar_folder = self.scene_folder / "lidar"
        self.output_folder = self.scene_folder / args.output_folder
        
        # Validate input paths
        self._validate_paths()
        
        # Initialize Prior Depth Anything model only if depth completion is enabled
        if args.enable_depth_completion:
            self._init_model()
        else:
            self.prior_depth_anything = None
        
        # Load COLMAP reconstruction
        self._load_colmap_reconstruction()
        
        # Find matching image-lidar pairs
        self._find_matching_pairs()
        
        # Create output directory
        self.output_folder.mkdir(exist_ok=True, parents=True)
        print(f"Output folder: {self.output_folder}")
        
        # Compute global scale factor
        self.global_scale = self._compute_global_scale_factor()
        
    def _validate_paths(self):
        """Validate that required input paths exist."""
        if not self.scene_folder.exists():
            raise FileNotFoundError(f"Scene folder not found: {self.scene_folder}")
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        if not self.sparse_folder.exists():
            raise FileNotFoundError(f"Sparse folder not found: {self.sparse_folder}")
        if not self.lidar_folder.exists():
            raise FileNotFoundError(f"LiDAR folder not found: {self.lidar_folder}")
            
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
        
    def _extract_number_from_filename(self, filename):
        """Extract number from filename like 'image_00000.jpg' -> 0"""
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
        
    def _find_matching_pairs(self):
        """Find matching image-lidar pairs based on naming convention."""
        self.matching_pairs = []
        
        # Get all calibrated images from COLMAP
        calibrated_image_names = set()
        for image_id in self.reconstruction.get_all_image_ids():
            image_name = self.reconstruction.get_image_name(image_id)
            calibrated_image_names.add(image_name)
        
        print(f"Found {len(calibrated_image_names)} calibrated images in COLMAP reconstruction")
        
        # Find matching LiDAR data
        for image_id in self.reconstruction.get_all_image_ids():
            image_name = self.reconstruction.get_image_name(image_id)
            image_path = self.images_folder / image_name
            
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}")
                continue
            
            # Extract image number and calculate lidar index (image index + 1)
            image_number = self._extract_number_from_filename(image_name)
            if image_number is None:
                print(f"Warning: Could not extract number from image name: {image_name}")
                continue
                
            lidar_index = image_number + 1
            
            # Construct lidar filenames
            depth_filename = f"depthmap_{lidar_index:05d}.tiff"
            confidence_filename = f"confidence_{lidar_index:05d}.tiff"
            
            depth_path = self.lidar_folder / depth_filename
            confidence_path = self.lidar_folder / confidence_filename
            
            if depth_path.exists() and confidence_path.exists():
                self.matching_pairs.append({
                    'image_id': image_id,
                    'image_name': image_name,
                    'image_path': image_path,
                    'depth_path': depth_path,
                    'confidence_path': confidence_path,
                    'image_number': image_number,
                    'lidar_index': lidar_index
                })
            else:
                missing_files = []
                if not depth_path.exists():
                    missing_files.append(str(depth_path))
                if not confidence_path.exists():
                    missing_files.append(str(confidence_path))
                print(f"Warning: Missing LiDAR files for {image_name}: {missing_files}")
        
        print(f"Found {len(self.matching_pairs)} matching image-LiDAR pairs")
        
        if len(self.matching_pairs) == 0:
            raise ValueError("No matching image-LiDAR pairs found. Check your file naming convention.")
    
    def _compute_global_scale_factor(self):
        """
        Compute global scale factor by analyzing scale distribution across all cameras.
        Uses middle 20% of scales to remove outliers' influence.
        
        Returns:
            global_scale: Single scale factor to apply to entire scene
        """
        print("\nComputing global scale factor from all cameras...")
        all_scales = []
        
        for pair in tqdm(self.matching_pairs, desc="Analyzing scales"):
            try:
                # Load LiDAR data
                depth_map, confidence_map = self.load_lidar_data(
                    pair['depth_path'], pair['confidence_path']
                )
                
                # Get point-wise scales for this camera
                scales = self._compute_pointwise_scales(
                    depth_map, confidence_map, pair['image_id']
                )
                
                if len(scales) > 0:
                    all_scales.extend(scales)
                    print(f"  {pair['image_name']}: {len(scales)} valid scales")
                else:
                    print(f"  {pair['image_name']}: No valid scales found")
                    
            except Exception as e:
                print(f"  Error processing {pair['image_name']}: {e}")
                continue
        
        if len(all_scales) == 0:
            print("Warning: No valid scales found across all cameras, using default scale of 1.0")
            return 1.0
        
        # Sort all scales
        all_scales = np.array(all_scales)
        sorted_scales = np.sort(all_scales)
        
        print(f"Total scales collected: {len(sorted_scales)}")
        print(f"Scale range: {sorted_scales[0]:.3f} to {sorted_scales[-1]:.3f}")
        print(f"Scale median: {np.median(sorted_scales):.3f}")
        
        # Select middle 20% to remove outliers
        n_scales = len(sorted_scales)
        lower_bound = int(0.4 * n_scales)  # Start of middle 20%
        upper_bound = int(0.6 * n_scales)  # End of middle 20%
        
        if upper_bound <= lower_bound:
            # If too few scales, use all of them
            middle_scales = sorted_scales
            print(f"Too few scales ({n_scales}), using all scales")
        else:
            middle_scales = sorted_scales[lower_bound:upper_bound]
            print(f"Using middle 20%: {len(middle_scales)} scales from index {lower_bound} to {upper_bound}")
        
        # Compute average of middle 20%
        global_scale = np.mean(middle_scales)
        
        print(f"Middle 20% range: {middle_scales[0]:.3f} to {middle_scales[-1]:.3f}")
        print(f"Computed global scale: {global_scale:.3f}")
        
        return float(global_scale)
    
    def _compute_pointwise_scales(self, depth_map, confidence_map, image_id):
        """
        Compute point-wise scales between COLMAP depths and LiDAR depths for a single camera.
        
        Args:
            depth_map: (H, W) LiDAR depth map
            confidence_map: (H, W) confidence map
            image_id: COLMAP image ID
            
        Returns:
            scales: List of scale values for points with non-zero confidence
        """
        # Get camera information
        camera = self.reconstruction.get_image_camera(image_id)
        K = camera.calibration_matrix()
        camera_pose = self.reconstruction.get_image_cam_from_world(image_id).matrix()
        
        # Ensure camera_pose is 4x4
        if camera_pose.shape == (3, 4):
            bottom_row = np.array([[0, 0, 0, 1]])
            camera_pose = np.vstack([camera_pose, bottom_row])
        
        # Get original image size from camera
        image_height = camera.height
        image_width = camera.width
        depth_height, depth_width = depth_map.shape
        
        # Get COLMAP 3D points visible in this image
        min_track_length = 3
        points_3d, points_2d, point_ids = self.reconstruction.get_visible_3d_points(image_id, min_track_length)
        
        if len(points_3d) == 0:
            return []
        
        # Generate sparse depth prior from COLMAP points at image resolution
        colmap_depth_prior, colmap_sparse_mask = self.project_3d_points_to_depth_prior(
            points_3d, camera_pose, K, (image_width, image_height)
        )
        
        # Scale LiDAR depth map to image resolution for comparison
        if depth_width != image_width or depth_height != image_height:
            depth_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            confidence_resized = cv2.resize(confidence_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth_map
            confidence_resized = confidence_map
        
        # Find valid overlapping points with non-zero confidence
        lidar_valid_mask = (depth_resized > 0) & (confidence_resized > 0)
        overlap_mask = colmap_sparse_mask & lidar_valid_mask
        
        if not np.any(overlap_mask):
            return []
        
        # Extract overlapping depth values
        colmap_depths = colmap_depth_prior[overlap_mask]
        lidar_depths = depth_resized[overlap_mask]
        confidences = confidence_resized[overlap_mask]
        
        # Compute point-wise scales: scale = colmap_depth / lidar_depth
        # Only include points with valid (non-zero) depths and confidence
        valid_points = (colmap_depths > 0) & (lidar_depths > 0) & (confidences > 0)
        
        if not np.any(valid_points):
            return []
        
        scales = colmap_depths[valid_points] / lidar_depths[valid_points]
        
        # Filter out unreasonable scales (likely errors)
        reasonable_scales = scales[(scales > 0.1) & (scales < 10.0)]
        
        return reasonable_scales.tolist()
    
    def load_lidar_data(self, depth_path, confidence_path):
        """
        Load LiDAR depth map and confidence map.
        
        Args:
            depth_path: Path to depth TIFF file
            confidence_path: Path to confidence TIFF file
            
        Returns:
            depth_map: (H, W) depth map
            confidence_map: (H, W) confidence map
        """
        # Load depth map
        depth_map = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if depth_map is None:
            raise ValueError(f"Failed to load depth map: {depth_path}")
        
        # Load confidence map
        confidence_map = cv2.imread(str(confidence_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if confidence_map is None:
            raise ValueError(f"Failed to load confidence map: {confidence_path}")
        
        # Convert to float if needed
        if depth_map.dtype != np.float32:
            depth_map = depth_map.astype(np.float32)
        if confidence_map.dtype != np.float32:
            confidence_map = confidence_map.astype(np.float32)
        
        # Handle multi-channel images by taking first channel
        if len(depth_map.shape) > 2:
            depth_map = depth_map[:, :, 0]
        if len(confidence_map.shape) > 2:
            confidence_map = confidence_map[:, :, 0]
            
        return depth_map, confidence_map
    
    def project_3d_points_to_depth_prior(self, points_3d, camera_pose, K, image_size):
        """
        Project 3D points to create depth prior map.
        
        Args:
            points_3d: (N, 3) world coordinates
            camera_pose: 4x4 camera pose matrix (cam_from_world)
            K: (3, 3) intrinsic matrix  
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
        points_proj = (K @ points_cam.T).T
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
    
    def apply_global_scale(self, depth_map):
        """
        Apply the precomputed global scale factor to a depth map.
        
        Args:
            depth_map: (H, W) LiDAR depth map
            
        Returns:
            scaled_depth: (H, W) globally scaled depth map
        """
        return depth_map * self.global_scale
    
    def get_individual_output_filename(self, depth_filename, processing_mode="simple"):
        """
        Generate individual point cloud filename from depth filename.
        
        Args:
            depth_filename: Original depth filename (e.g., "depthmap_00001.tiff")
            processing_mode: "simple", "completed", or "prior"
            
        Returns:
            Path to individual point cloud file
        """
        # Extract the base name without extension (e.g., "depthmap_00001")
        base_name = Path(depth_filename).stem
        
        # Create output filename based on processing mode
        if processing_mode == "completed":
            output_filename = f"{base_name}_completed.ply"
        elif processing_mode == "prior":
            output_filename = f"{base_name}_prior.ply"
        else:
            output_filename = f"{base_name}_simple.ply"
        
        return self.output_folder / output_filename
    
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
    
    def combine_lidar_colmap_priors(self, scaled_lidar_depth, confidence_map, colmap_depth_prior, colmap_sparse_mask, image_size):
        """
        Combine LiDAR depth and COLMAP depth priors into a unified sparse depth map.
        
        Args:
            scaled_lidar_depth: (H, W) scaled LiDAR depth map
            confidence_map: (H, W) confidence map for LiDAR
            colmap_depth_prior: (H, W) COLMAP sparse depth map
            colmap_sparse_mask: (H, W) mask for COLMAP depths
            image_size: (width, height) target image size
            
        Returns:
            combined_depth: (H, W) combined sparse depth map
            combined_mask: (H, W) mask indicating valid sparse depths
        """
        height, width = image_size[1], image_size[0]
        
        print(f"    DEBUG combine_lidar_colmap_priors:")
        print(f"      Target size: {width}x{height}")
        print(f"      LiDAR depth shape: {scaled_lidar_depth.shape}")
        print(f"      COLMAP prior shape: {colmap_depth_prior.shape}")
        
        # Initialize combined depth map
        combined_depth = np.zeros((height, width), dtype=np.float32)
        combined_mask = np.zeros((height, width), dtype=bool)
        
        # Resize LiDAR data to target size if needed
        if scaled_lidar_depth.shape != (height, width):
            print(f"      Resizing LiDAR from {scaled_lidar_depth.shape} to {height}x{width}")
            lidar_resized = cv2.resize(scaled_lidar_depth, (width, height), interpolation=cv2.INTER_LINEAR)
            confidence_resized = cv2.resize(confidence_map, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Debug: Check if resizing affected depth values significantly
            if np.any(scaled_lidar_depth > 0):
                orig_valid_depths = scaled_lidar_depth[scaled_lidar_depth > 0]
                resized_valid_depths = lidar_resized[lidar_resized > 0]
                print(f"      Original LiDAR depth range: {orig_valid_depths.min():.3f} to {orig_valid_depths.max():.3f}")
                print(f"      Resized LiDAR depth range: {resized_valid_depths.min():.3f} to {resized_valid_depths.max():.3f}")
        else:
            lidar_resized = scaled_lidar_depth
            confidence_resized = confidence_map
            print(f"      No LiDAR resizing needed")
        
        # Add LiDAR depths where confidence is high
        lidar_valid = (lidar_resized > 0) & (confidence_resized >= self.args.confidence_threshold)
        combined_depth[lidar_valid] = lidar_resized[lidar_valid]
        combined_mask[lidar_valid] = True
        
        print(f"      LiDAR valid pixels: {np.sum(lidar_valid)}")
        
        # Add COLMAP depths (they take priority over LiDAR where both exist)
        combined_depth[colmap_sparse_mask] = colmap_depth_prior[colmap_sparse_mask]
        combined_mask[colmap_sparse_mask] = True
        
        print(f"      COLMAP valid pixels: {np.sum(colmap_sparse_mask)}")
        print(f"      Combined valid pixels: {np.sum(combined_mask)}")
        
        return combined_depth, combined_mask
    
    def prepare_lidar_prior(self, scaled_lidar_depth, confidence_map, image_size):
        """
        Prepare LiDAR-only depth prior for completion.
        
        Args:
            scaled_lidar_depth: (H, W) scaled LiDAR depth map
            confidence_map: (H, W) confidence map for LiDAR
            image_size: (width, height) target image size
            
        Returns:
            lidar_depth_prior: (H, W) LiDAR sparse depth map
            lidar_sparse_mask: (H, W) mask indicating valid sparse depths
        """
        height, width = image_size[1], image_size[0]
        
        print(f"    DEBUG prepare_lidar_prior:")
        print(f"      Target size: {width}x{height}")
        print(f"      LiDAR depth shape: {scaled_lidar_depth.shape}")
        
        # Resize LiDAR data to target size if needed
        if scaled_lidar_depth.shape != (height, width):
            print(f"      Resizing LiDAR from {scaled_lidar_depth.shape} to {height}x{width}")
            lidar_resized = cv2.resize(scaled_lidar_depth, (width, height), interpolation=cv2.INTER_LINEAR)
            confidence_resized = cv2.resize(confidence_map, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Debug: Check if resizing affected depth values significantly
            if np.any(scaled_lidar_depth > 0):
                orig_valid_depths = scaled_lidar_depth[scaled_lidar_depth > 0]
                resized_valid_depths = lidar_resized[lidar_resized > 0]
                print(f"      Original LiDAR depth range: {orig_valid_depths.min():.3f} to {orig_valid_depths.max():.3f}")
                print(f"      Resized LiDAR depth range: {resized_valid_depths.min():.3f} to {resized_valid_depths.max():.3f}")
        else:
            lidar_resized = scaled_lidar_depth
            confidence_resized = confidence_map
            print(f"      No LiDAR resizing needed")
        
        # Create LiDAR prior based on confidence threshold
        lidar_valid = (lidar_resized > 0) & (confidence_resized >= self.args.confidence_threshold)
        
        # Initialize depth prior
        lidar_depth_prior = np.zeros((height, width), dtype=np.float32)
        lidar_depth_prior[lidar_valid] = lidar_resized[lidar_valid]
        
        print(f"      LiDAR valid pixels: {np.sum(lidar_valid)}")
        
        return lidar_depth_prior, lidar_valid
    
    def complete_depth_with_priors(self, image, combined_depth_prior, combined_sparse_mask):
        """
        Complete depth using Prior Depth Anything with LiDAR-only priors.
        
        Args:
            image: (H, W, 3) uint8 image
            combined_depth_prior: (H, W) LiDAR sparse depth map
            combined_sparse_mask: (H, W) mask indicating valid prior depths
            
        Returns:
            completed_depth: (H, W) completed depth map
        """
        print(f"DEBUG: Input shapes - image: {image.shape}, lidar_depth_prior: {combined_depth_prior.shape}, lidar_sparse_mask: {combined_sparse_mask.shape}")
        
        if not np.any(combined_sparse_mask):
            print("Warning: No valid LiDAR depth priors found, cannot complete")
            return np.zeros_like(combined_depth_prior)
        
        print(f"DEBUG: Found {np.sum(combined_sparse_mask)} valid LiDAR sparse points")
        
        # Prepare tensors for Prior Depth Anything
        image_tensor = torch.from_numpy(image.astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_prior_tensor = torch.from_numpy(combined_depth_prior.astype(np.float32)).unsqueeze(0).to(self.device)
        sparse_mask_tensor = torch.from_numpy(combined_sparse_mask.astype(bool)).unsqueeze(0).unsqueeze(1).to(self.device)
        
        print(f"DEBUG: Tensor shapes - image_tensor: {image_tensor.shape}, depth_prior_tensor: {depth_prior_tensor.shape}, sparse_mask_tensor: {sparse_mask_tensor.shape}")
        
        try:
            print("DEBUG: Calling Prior Depth Anything...")
            # Use Prior Depth Anything for depth completion with combined priors
            result = self.prior_depth_anything(
                images=image_tensor,
                sparse_depths=depth_prior_tensor,
                sparse_masks=sparse_mask_tensor,
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
        completed_depth = result.squeeze().detach().cpu().numpy().astype(np.float32)
        print(f"DEBUG: Final completed depth shape: {completed_depth.shape}")
        
        # Debug: Check if the completion preserved the input priors
        if np.any(combined_sparse_mask):
            input_prior_values = combined_depth_prior[combined_sparse_mask]
            output_prior_positions = completed_depth[combined_sparse_mask]
            
            # Check how much the priors changed
            if len(input_prior_values) > 0 and len(output_prior_positions) > 0:
                prior_diff = np.abs(input_prior_values - output_prior_positions)
                print(f"DEBUG: LiDAR prior preservation check:")
                print(f"  Input LiDAR prior range: {input_prior_values.min():.3f} to {input_prior_values.max():.3f}")
                print(f"  Output at LiDAR prior positions: {output_prior_positions.min():.3f} to {output_prior_positions.max():.3f}")
                print(f"  Mean absolute difference: {prior_diff.mean():.3f}")
                print(f"  Max absolute difference: {prior_diff.max():.3f}")
        
        return completed_depth
    
    def generate_point_cloud_from_completed_depth(self, image, depth_map, scaled_K, camera_pose, image_id):
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
    
    def process_frame_simple(self, pair):
        """
        Process a single frame without depth completion (simple LiDAR point cloud generation).
        
        Args:
            pair: Dictionary containing image and LiDAR file information
            
        Returns:
            points_3d: (N, 3) 3D points in world coordinates
            colors: (N, 3) RGB colors for points
        """
        # Load LiDAR data
        depth_map, confidence_map = self.load_lidar_data(
            pair['depth_path'], pair['confidence_path']
        )
        
        print(f"Processing {pair['image_name']} -> {pair['depth_path'].name} (simple)")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Valid depth pixels: {np.sum(depth_map > 0)}")
        print(f"  High confidence pixels: {np.sum(confidence_map >= self.args.confidence_threshold)}")
        
        # Apply global scale to LiDAR depth
        scaled_depth = self.apply_global_scale(depth_map)
        
        # Convert to point cloud using scaled depth
        points_3d, colors = self.depth_map_to_point_cloud(
            scaled_depth, confidence_map, pair['image_id']
        )
        
        print(f"  Generated {len(points_3d)} 3D points from LiDAR")
        
        # Save individual point cloud
        individual_output_path = self.get_individual_output_filename(
            pair['depth_path'].name, "simple"
        )
        if len(points_3d) > 0:
            self.save_point_cloud(points_3d, colors, individual_output_path)
            print(f"  Saved individual point cloud: {individual_output_path}")
            
            # Debug: Check simple point cloud bounds for comparison
            if len(points_3d) > 100:  # Only if we have enough points
                simple_bounds = {
                    'x_min': points_3d[:, 0].min(), 'x_max': points_3d[:, 0].max(),
                    'y_min': points_3d[:, 1].min(), 'y_max': points_3d[:, 1].max(),
                    'z_min': points_3d[:, 2].min(), 'z_max': points_3d[:, 2].max()
                }
                print(f"  Simple point cloud bounds:")
                print(f"    X: {simple_bounds['x_min']:.3f} to {simple_bounds['x_max']:.3f}")
                print(f"    Y: {simple_bounds['y_min']:.3f} to {simple_bounds['y_max']:.3f}")
                print(f"    Z: {simple_bounds['z_min']:.3f} to {simple_bounds['z_max']:.3f}")
        
        return points_3d, colors
    
    def process_frame_with_completion(self, pair):
        """
        Process a single frame with depth completion using LiDAR-only priors.
        
        Args:
            pair: Dictionary containing image and LiDAR file information
            
        Returns:
            points_3d: (N, 3) 3D points in world coordinates
            colors: (N, 3) RGB colors for points
        """
        # Load LiDAR data
        depth_map, confidence_map = self.load_lidar_data(
            pair['depth_path'], pair['confidence_path']
        )
        
        print(f"Processing {pair['image_name']} -> {pair['depth_path'].name} (with completion)")
        print(f"  Depth map shape: {depth_map.shape}")
        print(f"  Valid depth pixels: {np.sum(depth_map > 0)}")
        print(f"  High confidence pixels: {np.sum(confidence_map >= self.args.confidence_threshold)}")
        
        # Load and resize image
        image_pil = Image.open(pair['image_path']).convert('RGB')
        original_size = image_pil.size  # (width, height)
        
        # Get camera information
        camera = self.reconstruction.get_image_camera(pair['image_id'])
        camera_pose = self.reconstruction.get_image_cam_from_world(pair['image_id']).matrix()
        
        # Resize image and scale calibration
        new_size, scaled_K, crop_offset_y = self.resize_and_scale_calibration(
            camera, original_size, self.target_size
        )
        
        # Resize image
        orig_w, orig_h = original_size
        resized_w = self.target_size
        resized_h = round(orig_h * (resized_w / orig_w) / 14) * 14
        
        print(f"  Image resizing: {original_size} -> {resized_w}x{resized_h}")
        print(f"  Crop offset y: {crop_offset_y}")
        
        image_resized = image_pil.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        
        # Center crop if needed
        if resized_h > self.target_size:
            start_y = (resized_h - self.target_size) // 2
            image_resized = image_resized.crop((0, start_y, resized_w, start_y + self.target_size))
            print(f"  Applied center crop: y={start_y} to y={start_y + self.target_size}")
        
        image_array = np.array(image_resized)
        print(f"  Final image array shape: {image_array.shape}")
        print(f"  Target processing size: {new_size}")
        
        # Debug: Check camera calibration matrix
        print(f"  Original camera calibration matrix:")
        print(f"    K = {camera.calibration_matrix()}")
        print(f"  Scaled camera calibration matrix:")
        print(f"    scaled_K = {scaled_K}")
        
        # Apply global scale to LiDAR depth
        scaled_lidar_depth = self.apply_global_scale(depth_map)
        
        # Ensure camera_pose is 4x4
        if camera_pose.shape == (3, 4):
            bottom_row = np.array([[0, 0, 0, 1]])
            camera_pose = np.vstack([camera_pose, bottom_row])
        
        # Use only LiDAR as prior (no COLMAP points)
        lidar_depth_prior, lidar_sparse_mask = self.prepare_lidar_prior(
            scaled_lidar_depth, confidence_map, new_size
        )
        
        print(f"  LiDAR prior has {np.sum(lidar_sparse_mask)} valid pixels")
        
        # Debug: Check depth value ranges
        if np.any(lidar_sparse_mask):
            valid_prior_depths = lidar_depth_prior[lidar_sparse_mask]
            print(f"  Prior depth range: {valid_prior_depths.min():.3f} to {valid_prior_depths.max():.3f}")
            print(f"  Prior depth mean: {valid_prior_depths.mean():.3f}")
        
        # Complete depth using only LiDAR priors
        completed_depth = self.complete_depth_with_priors(
            image_array, lidar_depth_prior, lidar_sparse_mask
        )
        
        # Debug: Check completed depth value ranges
        if np.any(completed_depth > 0):
            valid_completed_depths = completed_depth[completed_depth > 0]
            print(f"  Completed depth range: {valid_completed_depths.min():.3f} to {valid_completed_depths.max():.3f}")
            print(f"  Completed depth mean: {valid_completed_depths.mean():.3f}")
            print(f"  Completed valid pixels: {np.sum(completed_depth > 0)}")
        else:
            print(f"  WARNING: No valid completed depths!")
        
        # Debug: Check scaling consistency
        print(f"  Image array shape: {image_array.shape}")
        print(f"  LiDAR prior shape: {lidar_depth_prior.shape}")
        print(f"  Completed depth shape: {completed_depth.shape}")
        print(f"  Target size: {new_size}")
        
        # Generate point cloud from completed depth
        points_3d, colors = self.generate_point_cloud_from_completed_depth(
            image_array, completed_depth, scaled_K, camera_pose, pair['image_id']
        )
        
        print(f"  Generated {len(points_3d)} 3D points from completed depth")
        
        # Generate and save prior point cloud for debugging
        prior_points_3d, prior_colors = self.generate_point_cloud_from_completed_depth(
            image_array, lidar_depth_prior, scaled_K, camera_pose, pair['image_id']
        )
        
        prior_output_path = self.get_individual_output_filename(
            pair['depth_path'].name, "prior"
        )
        if len(prior_points_3d) > 0:
            self.save_point_cloud(prior_points_3d, prior_colors, prior_output_path)
            print(f"  Generated {len(prior_points_3d)} 3D points from prior")
            print(f"  Saved prior point cloud: {prior_output_path}")
            
            # Debug: Compare prior point cloud bounds with simple LiDAR
            if len(prior_points_3d) > 100:  # Only if we have enough points
                prior_bounds = {
                    'x_min': prior_points_3d[:, 0].min(), 'x_max': prior_points_3d[:, 0].max(),
                    'y_min': prior_points_3d[:, 1].min(), 'y_max': prior_points_3d[:, 1].max(),
                    'z_min': prior_points_3d[:, 2].min(), 'z_max': prior_points_3d[:, 2].max()
                }
                print(f"  Prior point cloud bounds:")
                print(f"    X: {prior_bounds['x_min']:.3f} to {prior_bounds['x_max']:.3f}")
                print(f"    Y: {prior_bounds['y_min']:.3f} to {prior_bounds['y_max']:.3f}")
                print(f"    Z: {prior_bounds['z_min']:.3f} to {prior_bounds['z_max']:.3f}")
        
        # Save individual completed point cloud
        individual_output_path = self.get_individual_output_filename(
            pair['depth_path'].name, "completed"
        )
        if len(points_3d) > 0:
            self.save_point_cloud(points_3d, colors, individual_output_path)
            print(f"  Saved individual point cloud: {individual_output_path}")
            
            # Debug: Compare completed point cloud bounds with prior
            if len(points_3d) > 100:  # Only if we have enough points
                completed_bounds = {
                    'x_min': points_3d[:, 0].min(), 'x_max': points_3d[:, 0].max(),
                    'y_min': points_3d[:, 1].min(), 'y_max': points_3d[:, 1].max(),
                    'z_min': points_3d[:, 2].min(), 'z_max': points_3d[:, 2].max()
                }
                print(f"  Completed point cloud bounds:")
                print(f"    X: {completed_bounds['x_min']:.3f} to {completed_bounds['x_max']:.3f}")
                print(f"    Y: {completed_bounds['y_min']:.3f} to {completed_bounds['y_max']:.3f}")
                print(f"    Z: {completed_bounds['z_min']:.3f} to {completed_bounds['z_max']:.3f}")
        
        return points_3d, colors
    
    def depth_map_to_point_cloud(self, depth_map, confidence_map, image_id):
        """
        Convert depth map to point cloud using COLMAP calibration.
        
        Args:
            depth_map: (H, W) depth map
            confidence_map: (H, W) confidence map
            image_id: COLMAP image ID
            
        Returns:
            points_3d: (N, 3) 3D points in world coordinates
            colors: (N, 3) RGB colors for points (if image available)
        """
        depth_height, depth_width = depth_map.shape
        
        # Apply confidence threshold
        valid_mask = (depth_map > 0) & (confidence_map >= self.args.confidence_threshold)
        
        if not np.any(valid_mask):
            print(f"Warning: No valid depth points after confidence filtering for image {image_id}")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # Get camera calibration and image information
        camera = self.reconstruction.get_image_camera(image_id)
        K = camera.calibration_matrix()
        camera_pose = self.reconstruction.get_image_cam_from_world(image_id).matrix()
        
        # Get original image size from camera
        image_height = camera.height
        image_width = camera.width
        
        # Calculate scaling factors from depth map to image coordinates
        scale_u = image_width / depth_width
        scale_v = image_height / depth_height
        
        print(f"  Depth map size: {depth_width}x{depth_height}, Image size: {image_width}x{image_height}")
        print(f"  Scaling factors: u={scale_u:.3f}, v={scale_v:.3f}")
        
        # Create coordinate grids for depth map
        u_coords, v_coords = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
        
        # Get valid coordinates and depths
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask] 
        depth_valid = depth_map[valid_mask]
        
        # Scale coordinates to match image size for camera calibration
        u_scaled = u_valid * scale_u
        v_scaled = v_valid * scale_v
        
        # Unproject to camera coordinates using scaled coordinates
        K_inv = np.linalg.inv(K)
        
        # Create homogeneous pixel coordinates (using scaled coordinates)
        pixel_coords = np.stack([u_scaled, v_scaled, np.ones_like(u_scaled)], axis=0)
        
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
        
        # Try to get colors from image if available
        colors = np.ones((len(world_coords), 3)) * 0.5  # Default gray color
        
        # Load image for colors if it exists
        pair_info = next((p for p in self.matching_pairs if p['image_id'] == image_id), None)
        if pair_info and pair_info['image_path'].exists():
            try:
                image = Image.open(pair_info['image_path']).convert('RGB')
                image_array = np.array(image)
                
                # Ensure image matches expected camera dimensions
                actual_height, actual_width = image_array.shape[:2]
                if actual_width != image_width or actual_height != image_height:
                    print(f"  Warning: Image size mismatch - Camera: {image_width}x{image_height}, Actual: {actual_width}x{actual_height}")
                    # Resize image to match camera dimensions
                    image_pil = Image.fromarray(image_array)
                    image_pil = image_pil.resize((image_width, image_height), Image.Resampling.BILINEAR)
                    image_array = np.array(image_pil)
                
                # Sample colors from image using scaled coordinates
                # Clamp coordinates to valid image bounds
                u_img = np.clip(np.round(u_scaled).astype(int), 0, image_width - 1)
                v_img = np.clip(np.round(v_scaled).astype(int), 0, image_height - 1)
                
                colors = image_array[v_img, u_img] / 255.0  # Normalize to [0, 1]
            except Exception as e:
                print(f"Warning: Could not load colors from image {pair_info['image_path']}: {e}")
        
        return world_coords, colors
    
    def process_all_pairs(self):
        """Process all matching image-LiDAR pairs and create individual point clouds."""
        frame_point_clouds = []
        
        print(f"Processing {len(self.matching_pairs)} image-LiDAR pairs...")
        
        # Determine processing mode
        processing_desc = "with depth completion" if self.args.enable_depth_completion else "simple LiDAR"
        
        for pair in tqdm(self.matching_pairs, desc=f"Processing {processing_desc}"):
            try:
                # Choose processing method based on depth completion setting
                if self.args.enable_depth_completion:
                    points_3d, colors = self.process_frame_with_completion(pair)
                else:
                    points_3d, colors = self.process_frame_simple(pair)
                
                if len(points_3d) > 0:
                    frame_point_clouds.append({
                        'points': points_3d,
                        'colors': colors,
                        'frame_name': pair['image_name']
                    })
                else:
                    print(f"  No valid points generated for {pair['image_name']}")
                    
            except Exception as e:
                print(f"Error processing {pair['image_name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(frame_point_clouds) == 0:
            raise ValueError("No point clouds were generated from LiDAR data")
        
        return frame_point_clouds
    
    def merge_point_clouds(self, frame_point_clouds):
        """
        Merge individual point clouds from all frames.
        
        Args:
            frame_point_clouds: List of dictionaries containing point clouds per frame
            
        Returns:
            merged_points: (N, 3) merged 3D points
            merged_colors: (N, 3) merged colors
        """
        print(f"Merging {len(frame_point_clouds)} point clouds...")
        
        all_points = []
        all_colors = []
        
        total_points = 0
        for frame_pc in frame_point_clouds:
            points = frame_pc['points']
            colors = frame_pc['colors']
            frame_name = frame_pc['frame_name']
            
            all_points.append(points)
            all_colors.append(colors)
            total_points += len(points)
            
            print(f"  {frame_name}: {len(points)} points")
        
        # Merge all point clouds
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        print(f"Total merged points: {len(merged_points)}")
        
        return merged_points, merged_colors
    
    def save_point_cloud(self, points_3d, colors, output_path):
        """Save point cloud using Open3D."""
        if len(points_3d) == 0:
            print(f"Warning: No points to save")
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
            print(f"Saved merged point cloud with {len(points_3d)} points to {output_path}")
        else:
            raise RuntimeError(f"Failed to save point cloud to {output_path}")
    
    def run(self):
        """Run the complete pipeline."""
        if self.args.enable_depth_completion:
            print("Starting LiDAR point cloud generation with depth completion...")
            print(f"Target image size: {self.target_size}")
            print(f"Device: {self.device}")
        else:
            print("Starting LiDAR point cloud generation (simple mode)...")
        
        print(f"Using global scale factor: {self.global_scale:.3f}")
        print(f"Depth completion: {'enabled' if self.args.enable_depth_completion else 'disabled'}")
        
        # Process all pairs and get individual point clouds
        frame_point_clouds = self.process_all_pairs()
        
        # Merge all point clouds
        merged_points, merged_colors = self.merge_point_clouds(frame_point_clouds)
        
        # Save merged point cloud with appropriate name
        if self.args.enable_depth_completion:
            merged_output_path = self.output_folder / "merged_completed.ply"
        else:
            merged_output_path = self.output_folder / "merged_lidar.ply"
        
        self.save_point_cloud(merged_points, merged_colors, merged_output_path)
        
        print(f"\nComplete! Output saved to folder: {self.output_folder}")
        print(f"Individual point clouds: {len(frame_point_clouds)} files")
        print(f"Merged point cloud: {merged_output_path}")
        print(f"Used global scale factor: {self.global_scale:.3f}")
        
        if self.args.enable_depth_completion:
            print(f"Used LiDAR-only priors for depth completion")
        else:
            print(f"Generated point clouds directly from registered LiDAR data")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run the point cloud generator
    generator = LidarPointCloudGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
