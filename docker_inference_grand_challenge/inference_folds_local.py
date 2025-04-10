#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import importlib.util
import os
import re
import subprocess
import uuid
from glob import glob
from pathlib import Path

from cellvit.training.evaluate.inference_cellvit_wsi_single import (
    CellViTInfExpDetection,
    create_test_dataset,
)

# Set constants for WSI info and patch extraction
MPP_LEVEL0_VALUE = 0.24199951445730394
FILTERING_THRESHOLD = 5  # threshold im micrometers to filter out eventual overlapping detections - WAS 3.5
PROB_THRESHOLD = 0  # threshold for filtering out low probability detections - WAS 0.5
INPUT_SHAPE_2D = (256, 256)
INPUT_SHAPE_3D = (256, 256, 3)
SPACINGS = (0.25,)
OVERLAP = (0, 0)
OFFSET = (0, 0)
CENTER = False

DOCKER_INFERENCE = False  # NOTE: Set to False if running locally without Docker

# Set GPU
GPU = 0

# Set CPU count
CPUS = max(1, os.cpu_count() - 1)

INPUT_PATH = Path("test")  # Simulated /input
OUTPUT_PATH = Path("test_output")  # Simulated /output
MODEL_PATH = Path("example_model")  # Simulated /opt/ml/model

# Resources folder path (included in the Docker image: includes backbones and models)
RESOURCES_PATH = Path("resources")  # Simulated resources (internal folder)

# Ensure directories exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESOURCES_PATH.mkdir(parents=True, exist_ok=True)

# set fixed model backbones paths
FIXED_BACKBONE = os.path.join(
    RESOURCES_PATH, "backbones", "SAM-H", "CellViT-SAM-H-x40-AMP.pth"
)


def get_args():
    parser = argparse.ArgumentParser(description="CellVit Eval Preds Configuration")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--balance_split_by",
        type=str,
        default=None,
        help="Column name for stratified split, or None for random split",
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/json_mm",
        help="Directory with ground truth JSONs",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/work/grana_urologia/MONKEY_challenge/data/monkey-data/images/tissue-masks",
        help="Directory with tissue masks",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/grana_urologia/MONKEY_challenge/outputs/cellvit_finetune/finetune_challenge_pipeline/json_preds",
        help="Directory to save output JSON predictions",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="/work/grana_urologia/MONKEY_challenge/outputs/cellvit_finetune/finetune_challenge_pipeline/scores",
        help="Directory to save evaluation scores",
    )
    parser.add_argument(
        "--metadata_dataset_path",
        type=str,
        default="/work/grana_urologia/MONKEY_challenge/data/dataset_metadata_df.csv",
        help="Path to dataset metadata CSV file",
    )

    return parser.parse_args()


def run_pred_single_wsi(wsi_path=None, mask_path=None, fold_model_path=None, output_path=None):
    # set an id for temp files
    temp_id = str(uuid.uuid4())
    # Set a temp directory path with the unique id
    temp_dir_path = os.path.join(RESOURCES_PATH, "temp", temp_id)
    os.makedirs(temp_dir_path, exist_ok=True)
    # Set a logdir in the temp directory
    logdir = os.path.join(temp_dir_path, "logs")
    # Set a dataset path in the temp directory
    dataset_path = os.path.join(temp_dir_path, "temp_test_dataset")
    os.makedirs(dataset_path, exist_ok=True)

    # Find WSI and mask files

    print(f"Processing WSI: {wsi_path}")
    print(f"with associated mask: {mask_path}")

    print("Creating patchified WSI dataset...")
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

    print("Creating experiment for inference...")
    # Instantiate experiment
    experiment = CellViTInfExpDetection(
        logdir=logdir,
        cellvit_path=FIXED_BACKBONE,
        model_paths=[fold_model_path],
        dataset_path=dataset_path,
        roi_mask_path=mask_path,
        normalize_stains=False,
        gpu=GPU,
        input_shape=INPUT_SHAPE_2D,
        output_path=output_path,
        mpp_value=MPP_LEVEL0_VALUE,
        thresh_filtering=FILTERING_THRESHOLD,
        prob_threshold=PROB_THRESHOLD,
    )

    print("Running inference...")
    # Run inference
    experiment.run_inference()

    print("✅ Inference completed. JSON files written to:", output_path)
    saved_json_files = glob(os.path.join(output_path, "*.json"))
    print("Saved JSON files:", saved_json_files)

    print("🧹 Cleaning dataset temporary files...")
    # Clean up temporary files inside the temp directory
    os.system(f"rm -rf {temp_dir_path}")
    print("🧹 Temporary files cleaned up.")

    print("All done!")

    return 0


def run_preds_folds():
    pass


if __name__ == "__main__":
    run_preds_folds()
