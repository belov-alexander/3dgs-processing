# 3DGS-COLMAP-BRUSH Pipeline

This project automates the creation of 3D Gaussian Splats (3DGS) from a set of images using COLMAP for Structure-from-Motion (SfM) and Brush for training/exporting the splats.

## Overview

The pipeline performs the following steps:

1. **COLMAP Feature Extraction**: Extracts features from images. Auto-detects masks if provided.
2. **COLMAP Matching**: Matches features across images.
3. **COLMAP Mapper**: Performs sparse SfM reconstruction.
4. **COLMAP Undistortion**: Undistorts images and prepares them for 3DGS.
5. **Mask Handling (Optional)**: Can use masks for feature extraction (COLMAP) and/or copy undistorted masks for learning (Brush).
6. **Brush Training**: Trains the 3DGS model using `brush`.

## Requirements

1. **Python 3.6+**
2. **COLMAP**: Must be installed and available in your system `PATH` (or specified in `config.py` / command line).
3. **Brush**: Must be installed and available in your system `PATH` (or specified in `config.py` / command line).

## Configuration

Default values are stored in `config.py`. You can edit this file to change defaults permanently, or override them via command-line arguments.

**Key Configurable Parameters in `config.py`:**

* `COLMAP_BIN`: Path to COLMAP binary.
* `BRUSH_BIN`: Path to Brush binary.
* `SFM_MAX_IMAGE_SIZE`, `SIFT_MAX_NUM_FEATURES`: Reconstruction quality settings.
* `BRUSH_TOTAL_STEPS`: Number of training iterations.
* `CUBECL_DEFAULT_DEVICE`: GPU device ID (if needed).

## Usage

Run the script from the command line:

```bash
python 3dgs-colmap-brush.py --project_dir <PATH_TO_PROJECT_OUTPUT> --images_dir <PATH_TO_IMAGES>
```

### Examples

**Basic Run:**

```bash
python 3dgs-colmap-brush.py \
    --project_dir ./my_project \
    --images_dir ./my_images
```

**With Masks for Sky/Object Removal:**
If you have masks (black=ignore, white=keep) for your source images:

```bash
python 3dgs-colmap-brush.py \
    --project_dir ./my_project \
    --images_dir ./my_images \
    --masks_dir ./my_masks
```

**Customizing Training Steps:**

```bash
python 3dgs-colmap-brush.py \
    --project_dir ./my_project \
    --images_dir ./my_images \
    --brush_total_steps 15000
```

## Setup on a New Machine

1. Clone/Copy this repository.
2. Ensure `colmap` and `brush` are installed.
3. Run the python script.

## Notes

* **Masks**:
  * `masks_dir` is for the *input* images (affects SfM).
  * `dense_masks_dir` is optional. If you have masks that match the *undistorted* images, you can provide directory with `dense_masks_dir` to copy them into the Brush dataset folder.
