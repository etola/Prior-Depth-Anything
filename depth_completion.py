import argparse
import os
import sys
from typing import List

from densification import DensificationConfig, DensificationProblem

import torch
from prior_depth_anything import PriorDepthAnything

from vggt.models.vggt import VGGT


import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate point cloud using MapAnything with COLMAP calibrations"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use for computation"
    )
    parser.add_argument(
        "-sf", "--scene_folder",
        type=str,
        required=True,
        help="Scene folder containing 'images' and 'sparse' subdirectories",
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="output",
        help="Output folder for results (default: scene_folder/output/)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-m", "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=518,
        help="Resolution for MapAnything model inference (default: 518)",
    )
    parser.add_argument(
        "-e", "--export_resolution",
        type=int,
        default=0,
        help="Resolution to export 3dn depthmap format (default: 0 -> disables export)",
    )
    parser.add_argument(
        "-c", "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for depth filtering (default: 0.0)",
    )
    parser.add_argument(
        "-p", "--max_points",
        type=int,
        default=1000000,
        help="Maximum number of points in output point cloud (default: 1000000)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=8,
        help="Number of images to process in each batch to manage memory usage (default: 8)",
    )
    parser.add_argument(
        "--smart_batching",
        action="store_true",
        default=False,
        help="Use COLMAP reconstruction quality for intelligent batch formation (default: True)",
    )
    parser.add_argument(
        "--sequential_batching",
        action="store_true",
        help="Use simple sequential batching instead of smart batching",
    )
    parser.add_argument(
        "-R", "--reference_reconstruction",
        type=str,
        default=None,
        help="Path to reference lidar fused cloud in COLMAP reconstruction format for prior depth information (default: None)",
    )
    parser.add_argument("--use_sfm_prior", action="store_true", default=False, help="Use SfM depth for prior depth information")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output and save colorized prior and predicted depth maps",
    )
    parser.add_argument("--min_track_length", "-t", type=int, default=3, help="Minimum track length for SfM depth to be used for prior depth information")
    return parser.parse_args()

def _init_model(config: DensificationConfig, init_vggt_model: bool = False):
    """Initialize Prior Depth Anything model."""
    print("Initializing Prior Depth Anything...")

    config.device = torch.device(config.device)

    vggt_model = None
    dtype = None
    if init_vggt_model:
        print("Initializing VGGT model...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(config.device)

    # Initialize PriorDepthAnything which handles model downloads automatically
    pda_model = PriorDepthAnything(
        device=str(torch.device(config.device)),
        coarse_only=False  # We want both coarse and fine processing
    )
    return pda_model, vggt_model, dtype

def _run_depth_completion(model, problem: DensificationProblem, config: DensificationConfig, image_ids: List[int]):
  
    for img_id in image_ids:

        depth_data = problem.get_depth_data(img_id)

        dmap_prior = depth_data['prior_depth_map']
        sparse_mask = dmap_prior > 0

        # Prepare tensors for Prior Depth Anything
        image_tensor = torch.from_numpy(depth_data['scaled_image'].astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(config.device)  # Use uint8 for image
        depth_prior_tensor = torch.from_numpy(dmap_prior.astype(np.float32)).unsqueeze(0).to(config.device)
        sparse_mask_tensor = torch.from_numpy(sparse_mask.astype(bool)).unsqueeze(0).unsqueeze(1).to(config.device)  # Add channel dimension

        # print(f"DEBUG: Tensor shapes - image_tensor: {image_tensor.shape}, depth_prior_tensor: {depth_prior_tensor.shape}, sparse_mask_tensor: {sparse_mask_tensor.shape}")

        try:
            # print("DEBUG: Calling Prior Depth Anything...")
            # Use Prior Depth Anything for depth completion WITHOUT geometric depths
            # This will use the coarse depth estimator for initial prediction, then refine with priors
            result = model(
                images=image_tensor,
                sparse_depths=depth_prior_tensor,
                sparse_masks=sparse_mask_tensor,
                # No geometric_depths parameter - Prior Depth Anything will use its own coarse estimator
            )
            # print(f"DEBUG: Prior Depth Anything result shape: {result.shape}")
            
        except Exception as e:
            print(f"ERROR in Prior Depth Anything call: {e}")
            print(f"ERROR: Input tensor shapes were:")
            print(f"  - images: {image_tensor.shape}")
            print(f"  - sparse_depths: {depth_prior_tensor.shape}")
            print(f"  - sparse_masks: {sparse_mask_tensor.shape}")
            raise e

        completed_depth = result.squeeze().detach().cpu().numpy().astype(np.float32)  # Squeeze all single dimensions

        problem.update_depth_data(img_id, completed_depth, None)

    return

def estimate_depth_with_vggt(vggt_model, dtype, image_tensor):
    """
    Estimate depth using VGGT model.
    
    Args:
        image_tensor: (1, 3, H, W) preprocessed image tensor
        
    Returns:
        depth_map: (H, W) estimated depth map  
        confidence: (H, W) confidence map
    """
    with torch.no_grad():
        # with torch.amp.autocast(device_type='cuda', dtype=dtype):
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = vggt_model(image_tensor)

    depth_map = predictions['depth'].squeeze().cpu()
    confidence = predictions['depth_conf'].squeeze().cpu()
    
    return depth_map, confidence

def _run_depth_refinement(pda_model, vggt_model, dtype, problem: DensificationProblem, config: DensificationConfig, image_ids: List[int]):
  
    for img_id in image_ids:

        depth_data = problem.get_depth_data(img_id)

        dmap_prior = depth_data['prior_depth_map']
        prior_mask = dmap_prior > 0

        # Prepare tensors for Prior Depth Anything
        image_tensor = torch.from_numpy(depth_data['scaled_image'].astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(config.device)  # Use uint8 for image
        depth_prior_tensor = torch.from_numpy(dmap_prior.astype(np.float32)).unsqueeze(0).to(config.device)
        prior_mask_tensor = torch.from_numpy(prior_mask.astype(bool)).unsqueeze(0).to(config.device)  # Add channel dimension

        vggt_depth, vggt_confidence = estimate_depth_with_vggt(vggt_model, dtype, image_tensor)
        vggt_depth_np = vggt_depth if isinstance(vggt_depth, np.ndarray) else vggt_depth.numpy()
        vggt_depth_tensor = torch.from_numpy(vggt_depth_np.astype(np.float32)).unsqueeze(0).to(config.device)

        from geometric_utility import colorize_heatmap

        drgb = colorize_heatmap(vggt_depth_np, colormap='plasma', data_range=depth_data['depth_range'], )
        from PIL import Image
        Image.fromarray(drgb).save(config.output_folder + f"/vggt_depth_{img_id:04d}.png")


        # Use Prior Depth Anything's completion.forward with correct parameters:
        # - sparse_depths: COLMAP depth priors  
        # - sparse_masks: mask for COLMAP priors
        # - geometric_depths: VGGT depth estimate
        # - ret: 'knn' for KNN alignment
        result = pda_model.completion.forward(
            images=image_tensor,
            sparse_depths=depth_prior_tensor,
            sparse_masks=prior_mask_tensor,
            geometric_depths=vggt_depth_tensor,
            ret='knn'
        )

        refined_depth = result.squeeze(0).cpu().numpy().astype(np.float32)

        problem.update_depth_data(img_id, refined_depth, None)


def run_pda_depth_completion(problem: DensificationProblem, config: DensificationConfig, image_batches: List[List[int]]):
    pda_model, vggt_model, dtype = _init_model(config)
    for batch_idx, batch_image_ids in enumerate(image_batches):
        print(f"Processing batch {batch_idx}/{len(image_batches)} with {len(batch_image_ids)} images")
        print(f"Batch image ids: {batch_image_ids}")
        with torch.no_grad():
            _run_depth_completion(pda_model, problem, config, batch_image_ids)
            # _run_depth_refinement(pda_model, vggt_model, dtype, problem, config, batch_image_ids)
            torch.cuda.empty_cache()

def main():
    """Main function."""
    args = parse_args()

    # Print configuration
    print("Arguments:", vars(args))

    config = DensificationConfig()
    config.parse(args)

    config.run_depth_completion = run_pda_depth_completion

    problem = DensificationProblem(config)
    problem.run_densification()



if __name__ == "__main__":
    main()
