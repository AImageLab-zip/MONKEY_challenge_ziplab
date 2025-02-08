#!/usr/bin/env python
import os
import shutil
from glob import glob
from pathlib import Path

import torch
from cellvit.training.evaluate.inference_cellvit_wsi_single import (
    CellViTInfExpDetection,
    create_test_dataset,
)

MPP_LEVEL0_VALUE = 0.24199951445730394

# Set up paths
INPUT_PATH = Path("test")  # Simulated /input
OUTPUT_PATH = Path("test_output")  # Simulated /output
MODEL_PATH = Path("test_model")  # Simulated /opt/ml/model
RESOURCES_PATH = Path("resources")  # Simulated /opt/ml/resources

# Ensure directories exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESOURCES_PATH.mkdir(parents=True, exist_ok=True)

# Find WSI and mask files
wsi_files = sorted(
    glob(
        str(
            INPUT_PATH
            / "input"
            / "images"
            / "kidney-transplant-biopsy-wsi-pas"
            / "*.tif"
        )
    )
)
mask_files = sorted(
    glob(str(INPUT_PATH / "input" / "images" / "tissue-mask" / "*.tif"))
)

if not wsi_files:
    raise FileNotFoundError(
        f"No WSI files found in {INPUT_PATH / 'images/kidney-transplant-biopsy-wsi-pas'}"
    )
if not mask_files:
    raise FileNotFoundError(
        f"No mask files found in {INPUT_PATH / 'images/tissue-mask'}"
    )

# Select the first available image and mask
wsi_path = wsi_files[0]
mask_path = mask_files[0]
print(f"Found WSI: {wsi_path}")
print(f"Found Mask: {mask_path}")

# Set CPU count
cpus = max(1, os.cpu_count() - 1)

# Create test dataset
test_dataset_dir = create_test_dataset(
    wsi_path=wsi_path,
    mask_path=mask_path,
    output_dir="/work/grana_urologia/MONKEY_challenge/data/monkey_inference_test",  # RESOURCES_PATH / "temp_dataset",
    patch_shape=(256, 256, 3),
    spacings=(0.25,),
    overlap=(0, 0),
    offset=(0, 0),
    center=False,
    cpus=cpus,
)

# Handle external model checkpoint
external_models = sorted(MODEL_PATH.glob("*.pth"))
if not external_models:
    raise FileNotFoundError(f"No external model checkpoint found in {MODEL_PATH}")
external_model_file = external_models[0]
print(f"Found external model: {external_model_file}")

# Copy model to `model_and_info/checkpoints`
dest_dir = Path("model_and_info") / "checkpoints"
dest_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(external_model_file, dest_dir)
print(f"Copied external model to {dest_dir}")

# Use "model_and_info" as logdir
logdir = str(Path("model_and_info"))
# Fixed backbone model
fixed_backbone = str(Path("backbone") / "CellViT-SAM-H-x40-AMP.pth")

# Select GPU (or CPU if unavailable)
gpu = 0 if torch.cuda.is_available() else "cpu"

# Instantiate experiment
experiment = CellViTInfExpDetection(
    logdir=logdir,
    cellvit_path=fixed_backbone,
    dataset_path=str(test_dataset_dir),
    normalize_stains=False,
    gpu=gpu,
    input_shape=(256, 256),
    output_path=OUTPUT_PATH,
    mpp_value=MPP_LEVEL0_VALUE,
)

# Run inference
experiment.run_inference()

print("✅ Inference completed. JSON files written to:", OUTPUT_PATH)
