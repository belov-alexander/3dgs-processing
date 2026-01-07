#!/usr/bin/env python3
"""
3DGS COLMAP -> Brush Pipeline

This script automates the 3DGS (Gaussian Splat) creation process.
It performs the following steps:
1) COLMAP feature_extractor (auto-uses masks if present)
2) COLMAP exhaustive_matcher
3) COLMAP mapper (sparse SfM)
4) COLMAP image_undistorter -> creates dense/0/images + dense/0/sparse
5) (Optional) copies masks into dense/0/masks if provided
6) Brush training + export

Platform: Cross-platform (Windows, macOS, Linux)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[PIPELINE] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def run_command(cmd: List[str], check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Runs a shell command and logs it."""
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"Running: {cmd_str}")
    
    try:
        # On Windows, shell=True might be needed for some commands if they are built-ins,
        # but for executables it's usually better False. 
        # However, for cross-platform compatibility with how users invoke things, 
        # keeping shell=False is safer but we must ensure executable is in PATH.
        return subprocess.run(cmd, check=check, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {cmd_str}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        sys.exit(1)

def count_images(directory: Path) -> int:
    """Counts common image extensions in a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in extensions and p.is_file())

def main():
    parser = argparse.ArgumentParser(description="3DGS COLMAP -> Brush Pipeline")

    # Path arguments
    parser.add_argument("--project_dir", type=Path, required=True, help="Path to project directory")
    parser.add_argument("--images_dir", type=Path, required=True, help="Path to original images")
    parser.add_argument("--colmap_bin", type=str, default=config.COLMAP_BIN, help="Name or path of COLMAP binary")
    parser.add_argument("--brush_bin", type=str, default=config.BRUSH_BIN, help="Name or path of Brush binary")
    
    # Optional Masks
    parser.add_argument("--masks_dir", type=Path, default=None, 
                        help="Directory containing masks for original images (for COLMAP)")
    parser.add_argument("--mask_ext", type=str, default=config.MASK_EXT, help="Extension of mask files")
    parser.add_argument("--dense_masks_dir", type=Path, default=None,
                        help="Directory containing masks for undistorted images (for Brush)")

    # Preset Vars (COLMAP)
    parser.add_argument("--sfm_max_image_size", type=int, default=config.SFM_MAX_IMAGE_SIZE)
    parser.add_argument("--sift_max_num_features", type=int, default=config.SIFT_MAX_NUM_FEATURES)
    parser.add_argument("--undistort_max_image_size", type=int, default=config.UNDISTORT_MAX_IMAGE_SIZE)
    parser.add_argument("--camera_model", type=str, default=config.CAMERA_MODEL)
    parser.add_argument("--single_camera", type=int, default=config.SINGLE_CAMERA)
    parser.add_argument("--min_num_matches", type=int, default=config.MIN_NUM_MATCHES)
    parser.add_argument("--refine_focal_length", type=int, default=config.REFINE_FOCAL_LENGTH)
    parser.add_argument("--refine_extra_params", type=int, default=config.REFINE_EXTRA_PARAMS)
    parser.add_argument("--refine_principal_point", type=int, default=config.REFINE_PRINCIPAL_POINT)

    # Preset Vars (Brush)
    parser.add_argument("--run_brush", type=int, default=config.RUN_BRUSH, choices=[0, 1], help="Whether to run Brush training (1=yes, 0=no)")
    parser.add_argument("--brush_total_steps", type=int, default=config.BRUSH_TOTAL_STEPS)
    parser.add_argument("--brush_max_splats", type=int, default=config.BRUSH_MAX_SPLATS)
    parser.add_argument("--brush_export_every", type=int, default=config.BRUSH_EXPORT_EVERY)
    parser.add_argument("--brush_eval_split_every", type=int, default=config.BRUSH_EVAL_SPLIT_EVERY)
    parser.add_argument("--brush_export_name", type=str, default=config.BRUSH_EXPORT_NAME)
    parser.add_argument("--cubecl_default_device", type=str, default=config.CUBECL_DEFAULT_DEVICE, help="Optional GPU device ID for Brush")

    args = parser.parse_args()

    # Derived Vars
    brush_max_resolution = args.undistort_max_image_size

    # -----------------------------
    # CHECKS & SETUP
    # -----------------------------
    if not args.images_dir.exists():
        logger.error(f"IMAGES_DIR does not exist: {args.images_dir}")
        sys.exit(1)

    args.project_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = args.project_dir / "database.db"
    sparse_dir = args.project_dir / "sparse"
    dense_dir = args.project_dir / "dense"
    
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)

    brush_export_dir = args.project_dir / "brush_exports"

    logger.info(f"Project: {args.project_dir}")
    logger.info(f"Images:  {args.images_dir}")

    # -----------------------------
    # AUTO MASK DETECTION (COLMAP)
    # -----------------------------
    colmap_mask_args = []
    
    if args.masks_dir and args.masks_dir.exists():
        img_count = count_images(args.images_dir)
        # Count masks
        mask_count = sum(1 for p in args.masks_dir.iterdir() 
                         if p.is_file() and p.suffix.lower() == f".{args.mask_ext}".lower())
        
        if mask_count > 0:
            colmap_mask_args = ["--ImageReader.mask_path", str(args.masks_dir)]
            logger.info(f"Masks detected for COLMAP: {mask_count} mask(s). Using masks for feature extraction.")
            if img_count > 0 and mask_count < img_count:
                logger.warning("Fewer masks than images. Images without masks will be processed without masking.")
        else:
            logger.warning(f"MASKS_DIR exists ({args.masks_dir}) but no files with extension .{args.mask_ext} found.")
    else:
        logger.info("No COLMAP masks configured/found. Continuing without masks.")


    # -----------------------------
    # 1) Feature Extraction
    # -----------------------------
    logger.info("1/5 COLMAP: feature extraction")
    cmd_feature = [
        args.colmap_bin, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(args.images_dir),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.camera_model,
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_image_size", str(args.sfm_max_image_size),
        "--SiftExtraction.max_num_features", str(args.sift_max_num_features)
    ]
    cmd_feature.extend(colmap_mask_args)
    run_command(cmd_feature)

    # -----------------------------
    # 2) Matching
    # -----------------------------
    logger.info("2/5 COLMAP: exhaustive matching")
    cmd_match = [
        args.colmap_bin, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1"
    ]
    run_command(cmd_match)

    # -----------------------------
    # 3) Sparse Reconstruction
    # -----------------------------
    logger.info("3/5 COLMAP: mapper (sparse SfM)")
    cmd_mapper = [
        args.colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(args.images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.min_num_matches", str(args.min_num_matches),
        "--Mapper.ba_refine_focal_length", str(args.refine_focal_length),
        "--Mapper.ba_refine_extra_params", str(args.refine_extra_params),
        "--Mapper.ba_refine_principal_point", str(args.refine_principal_point)
    ]
    run_command(cmd_mapper)

    # Check mapping result
    if not (sparse_dir / "0").exists():
        logger.error(f"Sparse model not found at {sparse_dir / '0'}. Mapper may have failed.")
        sys.exit(1)

    # -----------------------------
    # 4) Undistort for 3DGS/Brush
    # -----------------------------
    logger.info("4/5 COLMAP: image_undistorter (creates dense/0/images + dense/0/sparse)")
    cmd_undistort = [
        args.colmap_bin, "image_undistorter",
        "--image_path", str(args.images_dir),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(args.undistort_max_image_size)
    ]
    run_command(cmd_undistort)

    # Brush Data Paths
    brush_data_path = dense_dir / "0"
    brush_images_path = brush_data_path / "images"
    brush_sparse_path = brush_data_path / "sparse"
    brush_masks_path = brush_data_path / "masks"

    if not brush_images_path.exists() or not brush_sparse_path.exists():
        logger.error(f"Expected {brush_images_path} and {brush_sparse_path} to exist after undistort.")
        sys.exit(1)

    logger.info(f"Dataset for Brush: {brush_data_path}")

    # -----------------------------
    # 4b) Optional: Provide dense masks
    # -----------------------------
    if args.dense_masks_dir:
        if args.dense_masks_dir.exists():
            dense_masks = list(args.dense_masks_dir.glob(f"*.{args.mask_ext}"))
            if dense_masks:
                logger.info(f"Copying undistorted masks into dataset for Brush -> {brush_masks_path}")
                brush_masks_path.mkdir(exist_ok=True)
                for mask_file in dense_masks:
                    shutil.copy2(mask_file, brush_masks_path / mask_file.name)
                logger.info(f"Brush masks ready: {len(dense_masks)} file(s) copied.")
            else:
                logger.warning(f"DENSE_MASKS_DIR provided but no *.{args.mask_ext} files found.")
        else:
            logger.warning(f"DENSE_MASKS_DIR set but directory not found: {args.dense_masks_dir}")
    else:
        logger.info("No undistorted masks configured for Brush (DENSE_MASKS_DIR empty).")

    # -----------------------------
    # 5) Brush training + export
    # -----------------------------
    if args.run_brush == 1:
        logger.info("5/5 BRUSH: train + export")
        
        # Check if brush binary exists (simple check if not an absolute path with no path env)
        # shutil.which checks PATH variable
        if not shutil.which(args.brush_bin) and not Path(args.brush_bin).exists():
             logger.error(f"BRUSH_BIN not found: {args.brush_bin}. Ensure it is in PATH or provide full path.")
             sys.exit(1)

        brush_export_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if args.cubecl_default_device:
            env["CUBECL_DEFAULT_DEVICE"] = args.cubecl_default_device
            logger.info(f"Using GPU device via CUBECL_DEFAULT_DEVICE={args.cubecl_default_device}")

        # Construct Brush Command
        # Note: flags based on script provided.
        cmd_brush = [
            args.brush_bin, str(brush_data_path),
            "--total-steps", str(args.brush_total_steps),
            "--max-resolution", str(brush_max_resolution),
            "--max-splats", str(args.brush_max_splats),
            "--eval-split-every", str(args.brush_eval_split_every),
            "--export-every", str(args.brush_export_every),
            "--export-path", str(brush_export_dir),
            "--export-name", args.brush_export_name,
            "--eval-every", str(args.brush_export_every),
            "--eval-save-to-disk"
        ]

        run_command(cmd_brush, env=env)
        logger.info(f"Brush exports: {brush_export_dir}")

    logger.info("DONE")

if __name__ == "__main__":
    main()
