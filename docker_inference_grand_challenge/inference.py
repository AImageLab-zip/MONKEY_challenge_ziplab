#!/usr/bin/env python
import os
import uuid
from glob import glob
from pathlib import Path

from cellvit.training.evaluate.inference_cellvit_wsi_single import (
    CellViTInfExpDetection,
    create_test_dataset,
)

# Set constants for WSI info and patch extraction
MPP_LEVEL0_VALUE = 0.24199951445730394
INPUT_SHAPE_2D = (256, 256)
INPUT_SHAPE_3D = (256, 256, 3)
SPACINGS = (0.25,)
OVERLAP = (0, 0)
OFFSET = (0, 0)
CENTER = False

# Set GPU
GPU = 0

# Set CPU count
CPUS = max(1, os.cpu_count() - 1)

# Set up paths
INPUT_PATH = Path("test")  # Simulated /input
OUTPUT_PATH = Path("test_output")  # Simulated /output
MODEL_PATH = Path("example_model")  # Simulated /opt/ml/model
RESOURCES_PATH = Path("resources")  # Simulated /opt/ml/resources

# Ensure directories exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESOURCES_PATH.mkdir(parents=True, exist_ok=True)

# set fixed model backbones paths
SAM_H_BACKBONE_PATH = os.path.join(
    RESOURCES_PATH, "backbones", "SAM-H", "CellViT-SAM-H-x40-AMP.pth"
)

VIT_256_BACKBONE_PATH = os.path.join(
    RESOURCES_PATH, "backbones", "VIT-256", "CellViT-256-x40-AMP.pth"
)

# set fallback model path
FALLBACK_MODEL_CLF_PATH = os.path.join(RESOURCES_PATH, "models", "model_best_vit.pth")

# The external model (provided via a mounted volume) is found under MODEL_PATH.
external_models = sorted(MODEL_PATH.glob("*.pth"))
if not external_models:
    raise FileNotFoundError(f"No external model checkpoint found in {MODEL_PATH}")
external_model_file = external_models[0]

print(f"Found external model: {external_model_file}")
external_model_path = os.path.join(MODEL_PATH, external_model_file)


if "sam" in external_model_file.lower():
    print("Using SAM-H backbone")
    fixed_backbone_path = SAM_H_BACKBONE_PATH

elif "vit" in external_model_file.lower():
    print("Using VIT-256 backbone")
    fixed_backbone_path = VIT_256_BACKBONE_PATH
else:
    print(
        "Defaulting to internal fine-tuned model with VIT-256 backbone\nResults may be less accurate..."
    )
    fixed_backbone_path = VIT_256_BACKBONE_PATH
    # set the default vit-256 fine-tuned model classifier path
    external_model_file = FALLBACK_MODEL_CLF_PATH


# Set a logdir
logdir = os.path.join(RESOURCES_PATH, "logs")
# Fixed backbone model


# get a unique id for the test dataset and create the directory
dataset_path = os.path.join(RESOURCES_PATH, f"temp_dataset_{uuid.uuid4()}")
os.makedirs(dataset_path, exist_ok=True)

# Find WSI and mask files
wsi_files = glob(
    os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif")
)
mask_files = glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif"))

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

# Create test dataset
create_test_dataset(
    wsi_path=wsi_path,
    mask_path=mask_path,
    output_dir=dataset_path,
    patch_shape=INPUT_SHAPE_3D,
    spacings=SPACINGS,
    overlap=OVERLAP,
    offset=OFFSET,
    center=CENTER,
    cpus=CPUS,
)

# Instantiate experiment
experiment = CellViTInfExpDetection(
    logdir=logdir,
    cellvit_path=fixed_backbone_path,
    model_path=external_model_path,
    dataset_path=dataset_path,
    normalize_stains=False,
    gpu=GPU,
    input_shape=INPUT_SHAPE_2D,
    output_path=OUTPUT_PATH,
    mpp_value=MPP_LEVEL0_VALUE,
)

# Run inference
experiment.run_inference()

print("✅ Inference completed. JSON files written to:", OUTPUT_PATH)
