#!/usr/bin/env bash
set -euo pipefail

#############################################
# COLMAP -> (optional sky masks) -> Undistort -> Brush (3DGS)
#
# What it does:
# 1) COLMAP feature_extractor (auto-uses masks if present)
# 2) COLMAP exhaustive_matcher
# 3) COLMAP mapper (sparse SfM)
# 4) COLMAP image_undistorter -> creates dense/0/images + dense/0/sparse
# 5) (optional) copies masks into dense/0/masks if provided
# 6) Brush training + export
#
# Notes:
# - Masks are applied to COLMAP feature extraction only (if MASKS_DIR has masks).
# - For Brush, masks must match UNDISTORTED images. Easiest is to provide masks
#   already made for dense/0/images in DENSE_MASKS_DIR.
#############################################

########################
# USER PATH VARIABLES
########################
PROJECT_DIR="/path/to/project"
IMAGES_DIR="/path/to/images"

COLMAP_BIN="colmap"

# Brush binary path/name (must be executable / in PATH)
BRUSH_BIN="brush"

########################
# OPTIONAL MASKS
########################
# Masks for COLMAP feature extraction (same base name as image, PNG recommended):
#   masks/IMG_0001.png  matches images/IMG_0001.jpg
MASKS_DIR="/path/to/masks_for_originals"   # set "" to disable
MASK_EXT="png"

# Masks for Brush training (MUST match dense/0/images pixel-to-pixel):
# If you already have masks for UNDISTORTED images, put them here and they will be copied to dense/0/masks.
DENSE_MASKS_DIR=""  # e.g. "/path/to/masks_for_undistorted" or "" to disable

########################
# PRESET VARS (COLMAP)
########################
SFM_MAX_IMAGE_SIZE=4096
SIFT_MAX_NUM_FEATURES=8192

UNDISTORT_MAX_IMAGE_SIZE=2400

CAMERA_MODEL="OPENCV"
SINGLE_CAMERA=1

REFINE_FOCAL_LENGTH=1
REFINE_EXTRA_PARAMS=1
REFINE_PRINCIPAL_POINT=0

MIN_NUM_MATCHES=32

########################
# PRESET VARS (BRUSH)
########################
RUN_BRUSH=1

BRUSH_TOTAL_STEPS=30000
BRUSH_MAX_RESOLUTION="${UNDISTORT_MAX_IMAGE_SIZE}"
BRUSH_MAX_SPLATS=6000000

BRUSH_EXPORT_EVERY=5000
BRUSH_EVAL_SPLIT_EVERY=10

BRUSH_EXPORT_DIR="${PROJECT_DIR}/brush_exports"
BRUSH_EXPORT_NAME='export_{iter}.ply'

# Optional GPU selection (if your Brush build supports it)
# Example: export CUBECL_DEFAULT_DEVICE=0 before running, or set below.
CUBECL_DEFAULT_DEVICE="${CUBECL_DEFAULT_DEVICE:-}"

########################
# DERIVED PATHS
########################
DB_PATH="${PROJECT_DIR}/database.db"
SPARSE_DIR="${PROJECT_DIR}/sparse"
DENSE_DIR="${PROJECT_DIR}/dense"

# Brush expects COLMAP dataset folder containing images/ and sparse/
BRUSH_DATA_PATH="${DENSE_DIR}/0"
BRUSH_IMAGES_PATH="${BRUSH_DATA_PATH}/images"
BRUSH_SPARSE_PATH="${BRUSH_DATA_PATH}/sparse"
BRUSH_MASKS_PATH="${BRUSH_DATA_PATH}/masks"

########################
# HELPERS
########################
log() { echo -e "\n[PIPELINE] $*\n"; }
warn() { echo -e "\n[PIPELINE][WARNING] $*\n" >&2; }
die() { echo -e "\n[PIPELINE][ERROR] $*\n" >&2; exit 1; }

count_images() {
  # counts common image extensions in a directory (non-recursive)
  find "$1" -maxdepth 1 -type f \( \
    -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \
  \) | wc -l | tr -d ' '
}

########################
# CHECKS
########################
[[ -d "${IMAGES_DIR}" ]] || die "IMAGES_DIR does not exist: ${IMAGES_DIR}"
mkdir -p "${PROJECT_DIR}" "${SPARSE_DIR}" "${DENSE_DIR}"

log "Project: ${PROJECT_DIR}"
log "Images:  ${IMAGES_DIR}"

########################
# AUTO MASK DETECTION (COLMAP)
########################
USE_MASKS=0
MASK_ARG=()

if [[ -n "${MASKS_DIR}" && -d "${MASKS_DIR}" ]]; then
  IMG_COUNT=$(count_images "${IMAGES_DIR}")
  MSK_COUNT=$(find "${MASKS_DIR}" -maxdepth 1 -type f -iname "*.${MASK_EXT}" | wc -l | tr -d ' ')

  if [[ "${MSK_COUNT}" -gt 0 ]]; then
    USE_MASKS=1
    MASK_ARG=( --ImageReader.mask_path "${MASKS_DIR}" )
    log "Masks detected for COLMAP: ${MSK_COUNT} mask(s) (expected up to ${IMG_COUNT}). Using masks for feature extraction."
    if [[ "${IMG_COUNT}" -gt 0 && "${MSK_COUNT}" -lt "${IMG_COUNT}" ]]; then
      warn "Fewer masks than images. Images without masks will be processed without masking."
    fi
  else
    warn "MASKS_DIR exists but no *.${MASK_EXT} files found. Not using masks."
  fi
else
  log "No COLMAP masks configured/found. Continuing without masks."
fi

########################
# 1) Feature Extraction
########################
log "1/5 COLMAP: feature extraction"
"${COLMAP_BIN}" feature_extractor \
  --database_path "${DB_PATH}" \
  --image_path "${IMAGES_DIR}" \
  "${MASK_ARG[@]}" \
  --ImageReader.single_camera "${SINGLE_CAMERA}" \
  --ImageReader.camera_model "${CAMERA_MODEL}" \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.max_image_size "${SFM_MAX_IMAGE_SIZE}" \
  --SiftExtraction.max_num_features "${SIFT_MAX_NUM_FEATURES}"

########################
# 2) Matching
########################
log "2/5 COLMAP: exhaustive matching"
"${COLMAP_BIN}" exhaustive_matcher \
  --database_path "${DB_PATH}" \
  --SiftMatching.use_gpu 1

########################
# 3) Sparse Reconstruction
########################
log "3/5 COLMAP: mapper (sparse SfM)"
"${COLMAP_BIN}" mapper \
  --database_path "${DB_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --output_path "${SPARSE_DIR}" \
  --Mapper.min_num_matches "${MIN_NUM_MATCHES}" \
  --Mapper.ba_refine_focal_length "${REFINE_FOCAL_LENGTH}" \
  --Mapper.ba_refine_extra_params "${REFINE_EXTRA_PARAMS}" \
  --Mapper.ba_refine_principal_point "${REFINE_PRINCIPAL_POINT}"

[[ -d "${SPARSE_DIR}/0" ]] || die "Sparse model not found at ${SPARSE_DIR}/0. Mapper may have failed."

########################
# 4) Undistort for 3DGS/Brush
########################
log "4/5 COLMAP: image_undistorter (creates dense/0/images + dense/0/sparse)"
"${COLMAP_BIN}" image_undistorter \
  --image_path "${IMAGES_DIR}" \
  --input_path "${SPARSE_DIR}/0" \
  --output_path "${DENSE_DIR}" \
  --output_type COLMAP \
  --max_image_size "${UNDISTORT_MAX_IMAGE_SIZE}"

[[ -d "${BRUSH_IMAGES_PATH}" && -d "${BRUSH_SPARSE_PATH}" ]] || \
  die "Expected ${BRUSH_IMAGES_PATH} and ${BRUSH_SPARSE_PATH} to exist after undistort."

log "Dataset for Brush:"
echo "  ${BRUSH_DATA_PATH}"
echo "  images: ${BRUSH_IMAGES_PATH}"
echo "  sparse: ${BRUSH_SPARSE_PATH}"

########################
# 4b) Optional: Provide masks to Brush (must match undistorted images)
########################
if [[ -n "${DENSE_MASKS_DIR}" ]]; then
  if [[ -d "${DENSE_MASKS_DIR}" ]]; then
    DMSK_COUNT=$(find "${DENSE_MASKS_DIR}" -maxdepth 1 -type f -iname "*.${MASK_EXT}" | wc -l | tr -d ' ')
    if [[ "${DMSK_COUNT}" -gt 0 ]]; then
      mkdir -p "${BRUSH_MASKS_PATH}"
      # Copy masks (overwrite to keep in sync)
      log "Copying undistorted masks into dataset for Brush -> ${BRUSH_MASKS_PATH}"
      find "${DENSE_MASKS_DIR}" -maxdepth 1 -type f -iname "*.${MASK_EXT}" -exec cp -f {} "${BRUSH_MASKS_PATH}/" \;
      log "Brush masks ready: ${DMSK_COUNT} file(s) copied."
    else
      warn "DENSE_MASKS_DIR provided but no *.${MASK_EXT} files found. Brush will run without masks."
    fi
  else
    warn "DENSE_MASKS_DIR set but directory not found: ${DENSE_MASKS_DIR}. Brush will run without masks."
  fi
else
  log "No undistorted masks configured for Brush (DENSE_MASKS_DIR empty)."
fi

########################
# 5) Brush training + export
########################
if [[ "${RUN_BRUSH}" -eq 1 ]]; then
  log "5/5 BRUSH: train + export"

  if ! command -v "${BRUSH_BIN}" >/dev/null 2>&1; then
    die "BRUSH_BIN not found in PATH: ${BRUSH_BIN}. Set BRUSH_BIN to full path, e.g. BRUSH_BIN=\"/path/to/brush\"."
  fi

  mkdir -p "${BRUSH_EXPORT_DIR}"

  if [[ -n "${CUBECL_DEFAULT_DEVICE}" ]]; then
    export CUBECL_DEFAULT_DEVICE
    log "Using GPU device via CUBECL_DEFAULT_DEVICE=${CUBECL_DEFAULT_DEVICE}"
  fi

  # NOTE: Brush CLI options can vary by version/build.
  # If your Brush binary uses different flags, run: ${BRUSH_BIN} --help
  "${BRUSH_BIN}" "${BRUSH_DATA_PATH}" \
    --total-steps "${BRUSH_TOTAL_STEPS}" \
    --max-resolution "${BRUSH_MAX_RESOLUTION}" \
    --max-splats "${BRUSH_MAX_SPLATS}" \
    --eval-split-every "${BRUSH_EVAL_SPLIT_EVERY}" \
    --export-every "${BRUSH_EXPORT_EVERY}" \
    --export-path "${BRUSH_EXPORT_DIR}" \
    --export-name "${BRUSH_EXPORT_NAME}" \
    --eval-every "${BRUSH_EXPORT_EVERY}" \
    --eval-save-to-disk

  log "Brush exports:"
  echo "  ${BRUSH_EXPORT_DIR}"
fi

log "DONE"
