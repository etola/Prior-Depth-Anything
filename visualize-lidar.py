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

# COLMAP utilities
from colmap_utils import ColmapReconstruction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LiDAR depth maps to point cloud using COLMAP calibration"
    )
    parser.add_argument(
        "-s", "--scene_folder", 
        type=str, 
        required=True,
        help="Path to scene folder containing images/, sparse/, and lidar/ subdirectories"
    )
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=0.5,
        help="Minimum confidence threshold for LiDAR points (default: 0.5)"
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="merged_lidar_pointcloud.ply",
        help="Output PLY filename (default: merged_lidar_pointcloud.ply)"
    )
    
    return parser.parse_args()


class LidarPointCloudGenerator:
    """Main class for converting LiDAR depth maps to point clouds using COLMAP calibration."""
    
    def __init__(self, args):
        self.args = args
        
        # Initialize scene paths
        self.scene_folder = Path(args.scene_folder)
        self.images_folder = self.scene_folder / "images"
        self.sparse_folder = self.scene_folder / "sparse"
        self.lidar_folder = self.scene_folder / "lidar"
        
        # Validate input paths
        self._validate_paths()
        
        # Load COLMAP reconstruction
        self._load_colmap_reconstruction()
        
        # Find matching image-lidar pairs
        self._find_matching_pairs()
        
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
    
    def calc_scale_shift(self, sparse_depths, pred_depths, valid_mask):
        """
        Calculate scale and shift for depth registration using least squares.
        
        Args:
            sparse_depths: (H, W) sparse depth map from COLMAP
            pred_depths: (H, W) predicted depth map (LiDAR)
            valid_mask: (H, W) mask indicating valid overlapping points
            
        Returns:
            scale: scalar scale factor
            shift: scalar shift factor
        """
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        # Get valid depth values
        sparse_valid = sparse_depths[valid_mask]
        pred_valid = pred_depths[valid_mask]
        
        if len(sparse_valid) < 2:
            return 1.0, 0.0
        
        # Add small noise to avoid numerical issues
        pred_valid = pred_valid + np.random.rand(*pred_valid.shape) * 1e-5
        
        # Set up least squares: sparse_depths = scale * pred_depths + shift
        # X = [pred_depths, ones], solution = [scale, shift]
        X = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
        
        try:
            # Solve least squares
            solution = np.linalg.lstsq(X, sparse_valid, rcond=None)[0]
            scale, shift = solution[0], solution[1]
            
            # Sanity check: scale should be positive and reasonable
            if scale <= 0 or scale > 100 or np.isnan(scale) or np.isnan(shift):
                print(f"  Warning: Invalid scale/shift computed: scale={scale:.3f}, shift={shift:.3f}, using defaults")
                return 1.0, 0.0
                
            return float(scale), float(shift)
            
        except np.linalg.LinAlgError:
            print("  Warning: Failed to compute scale/shift, using defaults")
            return 1.0, 0.0
    
    def register_depth_to_colmap(self, depth_map, confidence_map, image_id):
        """
        Register LiDAR depth map to COLMAP sparse points by computing scale and shift.
        
        Args:
            depth_map: (H, W) LiDAR depth map
            confidence_map: (H, W) confidence map
            image_id: COLMAP image ID
            
        Returns:
            registered_depth: (H, W) registered depth map
            registration_mask: (H, W) mask indicating successful registration area
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
        
        # Calculate scaling factors from depth map to image coordinates
        scale_u = image_width / depth_width
        scale_v = image_height / depth_height
        
        # Get COLMAP 3D points visible in this image
        min_track_length = 3
        points_3d, points_2d, point_ids = self.reconstruction.get_visible_3d_points(image_id, min_track_length)
        
        if len(points_3d) == 0:
            print(f"  Warning: No COLMAP 3D points found for registration in image {image_id}")
            return depth_map, np.ones_like(depth_map, dtype=bool)
        
        print(f"  Found {len(points_3d)} COLMAP 3D points for registration")
        
        # Generate sparse depth prior from COLMAP points at image resolution
        colmap_depth_prior, colmap_sparse_mask = self.project_3d_points_to_depth_prior(
            points_3d, camera_pose, K, (image_width, image_height)
        )
        
        # Scale LiDAR depth map to image resolution for comparison
        if depth_width != image_width or depth_height != image_height:
            # Resize depth map and confidence map to image resolution
            depth_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            confidence_resized = cv2.resize(confidence_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth_map
            confidence_resized = confidence_map
        
        # Create valid mask: both COLMAP and LiDAR have valid depths
        lidar_valid_mask = (depth_resized > 0) & (confidence_resized >= self.args.confidence_threshold)
        overlap_mask = colmap_sparse_mask & lidar_valid_mask
        
        print(f"  COLMAP sparse points: {np.sum(colmap_sparse_mask)}")
        print(f"  Valid LiDAR points: {np.sum(lidar_valid_mask)}")
        print(f"  Overlapping points: {np.sum(overlap_mask)}")
        
        if np.sum(overlap_mask) < 10:  # Need minimum points for reliable registration
            print(f"  Warning: Insufficient overlapping points ({np.sum(overlap_mask)}) for registration")
            return depth_map, np.ones_like(depth_map, dtype=bool)
        
        # Calculate scale and shift
        scale, shift = self.calc_scale_shift(
            colmap_depth_prior, depth_resized, overlap_mask
        )
        
        print(f"  Computed registration: scale={scale:.3f}, shift={shift:.3f}")
        
        # Apply scale and shift to original depth map
        registered_depth = depth_map * scale + shift
        
        # Create registration mask (areas with valid depth)
        registration_mask = depth_map > 0
        
        return registered_depth, registration_mask
    
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
        """Process all matching image-LiDAR pairs and create point clouds."""
        all_points = []
        all_colors = []
        
        print(f"Processing {len(self.matching_pairs)} image-LiDAR pairs...")
        
        for pair in tqdm(self.matching_pairs, desc="Converting depth maps"):
            try:
                # Load LiDAR data
                depth_map, confidence_map = self.load_lidar_data(
                    pair['depth_path'], pair['confidence_path']
                )
                
                print(f"Processing {pair['image_name']} -> {pair['depth_path'].name}")
                print(f"  Depth map shape: {depth_map.shape}")
                print(f"  Valid depth pixels: {np.sum(depth_map > 0)}")
                print(f"  High confidence pixels: {np.sum(confidence_map >= self.args.confidence_threshold)}")
                
                # Register LiDAR depth to COLMAP sparse points
                registered_depth, registration_mask = self.register_depth_to_colmap(
                    depth_map, confidence_map, pair['image_id']
                )
                
                # Convert to point cloud using registered depth
                points_3d, colors = self.depth_map_to_point_cloud(
                    registered_depth, confidence_map, pair['image_id']
                )
                
                if len(points_3d) > 0:
                    all_points.append(points_3d)
                    all_colors.append(colors)
                    print(f"  Generated {len(points_3d)} 3D points")
                else:
                    print(f"  No valid points generated")
                    
            except Exception as e:
                print(f"Error processing {pair['image_name']}: {e}")
                continue
        
        if len(all_points) == 0:
            raise ValueError("No point clouds were generated from LiDAR data")
        
        # Merge all point clouds
        print("Merging point clouds...")
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
        print("Starting LiDAR point cloud generation...")
        
        # Process all pairs and merge point clouds
        merged_points, merged_colors = self.process_all_pairs()
        
        # Save merged point cloud
        output_path = self.scene_folder / self.args.output_name
        self.save_point_cloud(merged_points, merged_colors, output_path)
        
        print(f"\nComplete! Merged point cloud saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run the point cloud generator
    generator = LidarPointCloudGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
