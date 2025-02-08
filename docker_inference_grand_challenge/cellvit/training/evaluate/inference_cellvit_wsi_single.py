# -*- coding: utf-8 -*-
# Detection Inference Code for Test Data
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)

import argparse
import csv
import glob
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycm
import torch
import tqdm
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)

# Import patch iterator related classes
from wholeslidedata.iterators import PatchConfiguration, create_patch_iterator

from cellvit.config.config import CELL_IMAGE_SIZES
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.training.evaluate.ocelot_eval_metrics import (
    _calc_scores,
    _preprocess_distance_and_confidence,
)
from cellvit.training.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
)
from cellvit.training.utils.tools import pair_coordinates

# ID -> name map
CLASS_MAP = {0: "monocytes", 1: "lymphocytes", 2: "other"}


def px_to_mm(px: float, spacing: float = 0.25) -> float:
    """
    Convert pixel coordinates to millimeters, given a micrometers-per-pixel spacing.
    E.g. if spacing=0.25 (µm/px), then 1 px = 0.25 µm = 0.00025 mm.
    """
    return px * spacing / 1000.0


def parse_patch_basename(patch_basename: str) -> Tuple[str, float, float]:
    """
    Parse a patch basename that contains x and y offsets.
    For example, if patch_basename = "A_P000001_PAS_CPG_x9984_y87808_105",
    it extracts:
      - slide_id: "A_P000001_PAS_CPG" (all parts before the first part starting with 'x')
      - patch_x: 9984.0
      - patch_y: 87808.0
    """
    parts = patch_basename.split("_")
    slide_parts = []
    patch_x = None
    patch_y = None
    for part in parts:
        if part.startswith("x"):
            try:
                patch_x = float(part.lstrip("x"))
            except ValueError:
                raise ValueError(f"Error parsing x coordinate in: {patch_basename}")
        elif part.startswith("y"):
            try:
                patch_y = float(part.lstrip("y"))
            except ValueError:
                raise ValueError(f"Error parsing y coordinate in: {patch_basename}")
        else:
            # Append parts until we encounter a part starting with "x"
            if patch_x is None:
                slide_parts.append(part)
    slide_id = "_".join(slide_parts)
    if patch_x is None or patch_y is None:
        raise ValueError(f"Could not parse x,y from patch basename: {patch_basename}")
    return slide_id, patch_x, patch_y


def create_test_dataset(
    wsi_path: str,
    mask_path: str,
    output_dir: str,
    patch_shape: tuple = (1024, 1024, 3),
    spacings: tuple = (0.25,),
    overlap: tuple = (0, 0),
    offset: tuple = (0, 0),
    center: bool = False,
    cpus: int = 4,
) -> Path:
    """
    Create a CellViT-compatible test dataset from a single WSI.
    Patches are saved under output_dir/test/images and empty CSV files under output_dir/test/labels.
    Each patch filename embeds the slide id, global x and y coordinates and a unique patch index.

    Args:
        wsi_path (str): Path to the WSI image.
        mask_path (str): Path to the WSI mask (optional; if not present, pass an empty string).
        output_dir (str): Root directory for the test dataset.
        patch_shape (tuple): Shape of each patch (H, W, C).
        spacings (tuple): Spacing value(s) for patch extraction.
        overlap (tuple): Overlap in pixels along x and y.
        offset (tuple): Offset in pixels.
        center (bool): Whether to center the patches.
        cpus (int): Number of CPU cores to use.

    Returns:
        Path: Path to the test dataset folder.
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    images_train_dir = train_dir / "images"
    labels_train_dir = train_dir / "labels"
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)

    splits_dir = output_dir / "splits"
    os.makedirs(splits_dir, exist_ok=True)

    test_dir = output_dir / "test"
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Create the patch configuration
    patch_config = PatchConfiguration(
        patch_shape=patch_shape,
        spacings=spacings,
        overlap=overlap,
        offset=offset,
        center=center,
    )

    # Create the patch iterator.
    patch_iterator = create_patch_iterator(
        image_path=wsi_path,
        mask_path=mask_path if mask_path and os.path.isfile(mask_path) else None,
        patch_configuration=patch_config,
        cpus=cpus,
        backend="asap",
    )

    slide_id = Path(
        wsi_path
    ).stem  # Use the WSI filename (without extension) as slide id.
    pbar = tqdm.tqdm(patch_iterator, desc="Creating test patches")
    idx_patch = 0
    for patch_data, mask_data, info in pbar:
        # Get the patch image as a numpy array.
        patch_np = patch_data.squeeze().astype(np.uint8)  # shape (H, W, 3)
        H, W, _ = info["tile_shape"]
        # Global coordinates for this patch (assumed provided in info).
        patch_x = info.get("x", 0)
        patch_y = info.get("y", 0)

        # Build a patch basename that includes the slide id, patch_x, patch_y, and a unique index.
        patch_basename = f"{slide_id}_x{patch_x}_y{patch_y}_{idx_patch}"

        # Save the patch image.
        img_path = images_dir / f"{patch_basename}.png"
        plt.imsave(str(img_path), patch_np)

        # Create an empty CSV file for annotations.
        csv_path = labels_dir / f"{patch_basename}.csv"
        with open(csv_path, mode="w", newline="") as cf:
            writer = csv.writer(cf)
            # (Empty file, as no annotations are available.)

        idx_patch += 1
    pbar.close()
    print(f"Test dataset created at: {test_dir}")
    return test_dir


def _convert_coords(
    x_pixel: float, y_pixel: float, micrometers_per_pixel: float, output_unit: str
) -> Tuple[float, float]:
    """
    Convert pixel (x, y) coordinates to the desired output_unit:
      - "pixel" -> No conversion needed, return as is
      - "um"    -> Convert from pixels to micrometers
      - "mm"    -> Convert from pixels to millimeters
    """
    if output_unit == "pixel":
        return x_pixel, y_pixel  # Already in pixels
    elif output_unit == "um":
        x_um = x_pixel * micrometers_per_pixel  # Convert px to µm
        y_um = y_pixel * micrometers_per_pixel
        return x_um, y_um
    elif output_unit == "mm":
        x_mm = (x_pixel * micrometers_per_pixel) / 1000.0  # Convert px to mm
        y_mm = (y_pixel * micrometers_per_pixel) / 1000.0
        return x_mm, y_mm
    else:
        raise ValueError(
            f"Unknown output_unit '{output_unit}'. Must be 'pixel', 'um', or 'mm'."
        )


def _build_annotation_json(
    points: List[Tuple[float, float, float]],
    annotation_name: str,
    fixed_z_value: float = 0.0,
) -> dict:
    """
    Build the annotation dictionary in the specified format:
      {
        "name": "inflammatory-cells" (etc.),
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [
           {
             "name": "Point 0",
             "point": [x, y, z],
             "probability": ...
           },
           ...
        ]
      }

    :param points: list of (x, y, probability)
    :param annotation_name: e.g. 'inflammatory-cells'
    :param fixed_z_value: in your original example, you might store 0.25 or 0.241999, etc.
    :return: dict suitable for JSON serialization
    """
    annotation_dict = {
        "name": annotation_name,
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    for idx, (x_val, y_val, prob) in enumerate(points):
        annotation_dict["points"].append(
            {
                "name": f"Point {idx}",
                "point": [x_val, y_val, fixed_z_value],
                "probability": prob,  # remove if you really don't want it
            }
        )

    return annotation_dict


def generate_inflammatory_annotation_dicts(
    global_cell_pred_dict: Dict,
    micrometers_per_pixel: float = 0.25,
    output_unit: str = "mm",
    fixed_z_value: float = 0.25,
) -> Tuple[dict, dict, dict]:
    """
    Process the given global_cell_pred_dict (already in memory) and generate
    three annotation dictionaries:
      1. "inflammatory-cells" (combined monocytes + lymphocytes)
      2. "monocytes"
      3. "lymphocytes"

    Args:
        global_cell_pred_dict (Dict): Output of the CellViT pipeline, keyed by patch, containing cells.
        micrometers_per_pixel (float): MPP scale (µm/pixel).
        output_unit (str): "pixel", "um", or "mm".
        fixed_z_value (float): Fixed Z coordinate value.

    Returns:
        Tuple[dict, dict, dict]: (inflammatory_dict, monocytes_dict, lymphocytes_dict)
    """
    monocytes_points = []
    lymphocytes_points = []

    for _, cells_in_patch in global_cell_pred_dict.items():
        for _, cell_info in cells_in_patch.items():
            cell_type = cell_info["type"]
            if cell_type == 2:  # "other" => skip
                continue

            global_x, global_y = cell_info["global_centroid"]
            probability = cell_info.get("type_prob", 1.0)

            # Convert coords using `micrometers_per_pixel`
            converted_x, converted_y = _convert_coords(
                global_x, global_y, micrometers_per_pixel, output_unit
            )

            if cell_type == 0:  # monocytes
                monocytes_points.append((converted_x, converted_y, probability))
            elif cell_type == 1:  # lymphocytes
                lymphocytes_points.append((converted_x, converted_y, probability))

    # Combine for inflammatory-cells
    inflammatory_points = monocytes_points + lymphocytes_points

    inflammatory_dict = _build_annotation_json(
        inflammatory_points, "inflammatory-cells", fixed_z_value
    )
    monocytes_dict = _build_annotation_json(
        monocytes_points, "monocytes", fixed_z_value
    )
    lymphocytes_dict = _build_annotation_json(
        lymphocytes_points, "lymphocytes", fixed_z_value
    )

    return inflammatory_dict, monocytes_dict, lymphocytes_dict


def save_annotation_json(annotation_dict: dict, filepath: Path) -> None:
    """
    Helper function to save annotation JSON files.
    """
    with open(filepath, "w") as f:
        json.dump(annotation_dict, f, indent=2)


class CellViTInfExpDetection(CellViTClassifierInferenceExperiment):
    """Inference Experiment for CellViT with a Classifier Head on Detection Data

    Args:
        logdir (Union[Path, str]): Log directory with the trained classifier
        cellvit_path (Union[Path, str]): Path to pretrained CellViT model
        dataset_path (Union[Path, str]): Path to the dataset (parent path, not the fold path)
        input_shape (List[int]): Input shape of images before beeing feed to the model.
        normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        gpu (int, optional): GPU to use. Defaults to 0.
        comment (str, optional): Comment for storing. Defaults to None.

    Additional Attributes (besides the ones from the parent class):
        input_shape (List[int]): Input shape of images before beeing feed to the model.

    Overwritten Methods:
        _load_inference_transforms(normalize_settings_default: dict, transform_settings: dict = None) -> Callable
            Load inference transformations
        _load_dataset(transforms: Callable, normalize_stains: bool) -> Dataset
            Load Detection Dataset
        _extract_tokens(cell_pred_dict: dict, predictions: dict, image_size: int) -> List
            Extract cell tokens associated to cells
        _get_cellvit_result(images: torch.Tensor, cell_gt_batch: List, types_batch: List, image_names: List, postprocessor: DetectionCellPostProcessorCupy) -> Tuple[List[dict], List[dict], dict[dict], List[float], List[float], List[float]
            Retrieve CellViT Inference results from a batch of patches
        _get_global_classifier_scores(predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float, float, float, float, float]
            Calculate global metrics for the classification head, *without* taking quality of the detection model into account
        _plot_confusion_matrix(predictions: torch.Tensor, gt: torch.Tensor, test_result_dir: Union[Path, str]) -> None
            Plot and save the confusion matrix (normalized and non-normalized)
        update_cell_dict_with_predictions(cell_dict: dict, predictions: np.ndarray, probabilities: np.ndarray, metadata: List[Tuple[float, float, str]]) -> dict
            Update the cell dictionary with the predictions from the classifier
        _calculate_pipeline_scores(cell_dict: dict) -> Tuple[dict, dict, dict]
            Calculate the final pipeline scores, use the TIA evaluation metrics
        run_inference() -> None
            Run Inference on Test Dataset
    """

    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        dataset_path: Union[Path, str],
        input_shape: List[int],
        normalize_stains: bool = False,
        gpu: int = 0,
        comment: str = None,
        output_path: Union[Path, str] = None,  # Added output path argument
        mpp_value: float = 0.24199951445730394,  # Added micrometers per pixel
    ) -> None:
        assert len(input_shape) == 2, "Input shape must have a length of 2."
        for in_sh in input_shape:
            assert in_sh in CELL_IMAGE_SIZES, "Shape entries must be divisible by 32."

        self.input_shape = input_shape
        self.output_path = (
            Path(output_path) if output_path else None
        )  # Store output path
        self.mpp_value = mpp_value  # Store micrometers per pixel

        super().__init__(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=normalize_stains,
            gpu=gpu,
            comment=comment,
        )

    def _load_inference_transforms(
        self,
        normalize_settings_default: dict,
        transform_settings: dict = None,
    ) -> Callable:
        """Load inference transformations

        Args:
            normalize_settings_default (dict): Setting of cellvit model
            transform_settings (dict, optional): Alternative to overwrite. Defaults to None.

        Returns:
            Callable: Transformations
        """
        self.logger.info("Loading inference transformations")

        if transform_settings is not None and "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = normalize_settings_default["mean"]
            std = normalize_settings_default["std"]

        inference_transform = A.Compose(
            [
                A.PadIfNeeded(
                    self.input_shape[0],
                    self.input_shape[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                ),
                A.CenterCrop(
                    self.input_shape[0], self.input_shape[1], always_apply=True
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )
        return inference_transform

    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load Detection Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: Detection Dataset
        """
        dataset = DetectionDataset(
            dataset_path=self.dataset_path,
            split="test",
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset

    def _extract_tokens(
        self, cell_pred_dict: dict, predictions: dict, image_size: List[int]
    ) -> List:
        """Extract cell tokens associated to cells

        Args:
            cell_pred_dict (dict): Cell prediction dict
            predictions (dict): Prediction dict
            image_size (List[int]): Image size of the input image (H, W)

        Returns:
            List: List of topkens for each patch
        """
        if hasattr(self.cellvit_model, "patch_size"):
            patch_size = self.cellvit_model.patch_size
        else:
            patch_size = 16

        if patch_size == 16:
            rescaling_factor = 1
        else:
            if image_size[0] == image_size[1]:
                if image_size[0] in self.cellvit_model.input_rescale_dict:
                    rescaling_factor = (
                        self.cellvit_model.input_rescale_dict[image_size[0]]
                        / image_size[0]
                    )
                else:
                    self.logger.error(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
                    raise RuntimeError(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
            else:
                self.logger.error(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )
                raise RuntimeError(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )

        batch_tokens = []
        for patch_idx, patch_cell_pred_dict in enumerate(cell_pred_dict):
            extracted_cell_tokens = []
            patch_tokens = predictions["tokens"][patch_idx]
            for cell in patch_cell_pred_dict.values():
                bbox = rescaling_factor * cell["bbox"]
                bb_index = bbox / patch_size
                bb_index[0, :] = np.floor(bb_index[0, :])
                bb_index[1, :] = np.ceil(bb_index[1, :])
                bb_index = bb_index.astype(np.uint8)
                cell_token = patch_tokens[
                    :, bb_index[0, 0] : bb_index[1, 0], bb_index[0, 1] : bb_index[1, 1]
                ]
                cell_token = torch.mean(
                    rearrange(cell_token, "D H W -> (H W) D"), dim=0
                )
                extracted_cell_tokens.append(cell_token.detach().cpu())
            batch_tokens.append(extracted_cell_tokens)

        return batch_tokens

    def _get_cellvit_result(
        self,
        images: torch.Tensor,
        cell_gt_batch: List,
        types_batch: List,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
    ) -> Tuple[List[dict], List[dict], dict, List[float], List[float], List[float]]:
        """
        Retrieve CellViT Inference results. Modified to allow empty ground truth.
        """
        extracted_cells_matching = []
        overall_extracted_cells = []
        image_pred_dict = {}
        f1s = []
        precs = []
        recs = []

        image_size = images.shape[2]
        images = images.to(self.device)
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = self.cellvit_model.forward(images, retrieve_tokens=True)
        else:
            predictions = self.cellvit_model.forward(images, retrieve_tokens=True)

        predictions = self._apply_softmax_reorder(predictions)
        _, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self._extract_tokens(cell_pred_dict, predictions, self.input_shape)

        # For test inference, ground truth is empty.
        for pred_dict, true_centroids, cell_types, patch_token, image_name in zip(
            cell_pred_dict, cell_gt_batch, types_batch, tokens, image_names
        ):
            image_pred_dict[image_name] = {}
            pred_centroids = [v["centroid"] for v in pred_dict.values()]
            pred_centroids = np.array(pred_centroids)
            # Instead of requiring both GT and predictions, use only predictions.
            if len(pred_centroids) > 0:
                for cell_idx in range(len(pred_centroids)):
                    overall_extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": pred_centroids[cell_idx],
                            "type": 0,  # default type value when no GT
                            "token": patch_token[cell_idx],
                        }
                    )
                    image_pred_dict[image_name][cell_idx + 1] = pred_dict[cell_idx + 1]
                # (Pairing is skipped when GT is empty; you may optionally add pairing logic.)
        return (
            extracted_cells_matching,
            overall_extracted_cells,
            image_pred_dict,
            f1s,
            precs,
            recs,
        )

    def _get_global_classifier_scores(
        self, predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor
    ):
        return None

    def update_cell_dict_with_predictions(
        self,
        cell_dict: dict,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        metadata: List[Tuple[float, float, str]],
    ) -> dict:
        """Update the cell dictionary with the predictions from the classifier

        Args:
            cell_dict (dict): Cell dictionary with CellViT default predictions
            predictions (np.ndarray): Classifier predictions of the class
            probabilities (np.ndarray): Classifier output probabilities
            metadata (List[Tuple[float, float, str]]): Cell metadata

        Returns:
            dict: Updated cell dictionary, be careful about the ordering -> Types start with the index 0
        """
        self.logger.info("Updating PanNuke-cell-preds with dataset specific classes")
        for pred, prob, inform in zip(predictions, probabilities, metadata):
            cell_found = False
            image_name = inform[2]
            image_cell_dict = cell_dict[image_name]
            row_pred, col_pred = inform[:2]
            row_pred = float(f"{row_pred:.0f}")
            col_pred = float(f"{col_pred:.0f}")

            for cell_idx, properties in image_cell_dict.items():
                row, col = properties["centroid"]
                row = float(f"{row:.0f}")
                col = float(f"{col:.0f}")
                if row == row_pred and col == col_pred:
                    cell_dict[image_name][cell_idx]["type"] = int(pred)
                    cell_dict[image_name][cell_idx]["type_prob"] = float(
                        prob[int(pred)]
                    )
                    cell_dict[image_name][cell_idx]["bbox"] = cell_dict[image_name][
                        cell_idx
                    ]["bbox"].tolist()
                    cell_dict[image_name][cell_idx]["centroid"] = cell_dict[image_name][
                        cell_idx
                    ]["centroid"].tolist()
                    cell_dict[image_name][cell_idx]["contour"] = cell_dict[image_name][
                        cell_idx
                    ]["contour"].tolist()
                    cell_found = True
            assert cell_found, "Not all cells have predictions"

        return cell_dict

    def run_inference(self):
        """Run inference without ground truth, classify cells, convert patch-local predictions to global WSI coordinates, and save only the JSON results."""

        extracted_cells = []  # All detected cells
        image_pred_dict = {}  # Dictionary with all detected cells

        postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
        cellvit_dl = DataLoader(
            self.inference_dataset,
            batch_size=4,
            num_workers=8,
            shuffle=False,
            collate_fn=self.inference_dataset.collate_batch,
        )

        # Step 1: Extract cells with CellViT (GT is empty in test mode)
        with torch.no_grad():
            for _, (images, _, _, image_names) in tqdm.tqdm(
                enumerate(cellvit_dl), total=len(cellvit_dl)
            ):
                _, overall_extracted_cells, batch_pred_dict, _, _, _ = (
                    self._get_cellvit_result(
                        images=images,
                        cell_gt_batch=[],  # Empty GT in test mode
                        types_batch=[],  # Dummy values
                        image_names=image_names,
                        postprocessor=postprocessor,
                    )
                )
                image_pred_dict.update(batch_pred_dict)
                extracted_cells.extend(overall_extracted_cells)

        # Step 2: Ensure extracted_cells contains data
        if not extracted_cells:
            self.logger.warning(
                "❌ No cells detected! Check model weights and input data."
            )
            return

        # Step 3: Classify cells using extracted tokens
        classification_results = self._get_classifier_result(extracted_cells)
        classification_results.pop("gt", None)  # No ground truth in test mode
        classification_results["predictions"] = (
            classification_results["predictions"].numpy().tolist()
        )
        classification_results["probabilities"] = (
            classification_results["probabilities"].numpy().tolist()
        )

        # Step 4: Update cell dictionary with classifier predictions
        cell_pred_dict = self.update_cell_dict_with_predictions(
            cell_dict=image_pred_dict,
            predictions=np.array(classification_results["predictions"]),
            probabilities=np.array(classification_results["probabilities"]),
            metadata=classification_results["metadata"],
        )

        # Step 5: Convert patch-local cell centroids to global WSI coordinates (in mm)
        global_cell_pred_dict = {}
        for patch_name, cells in cell_pred_dict.items():
            try:
                slide_id, patch_x, patch_y = parse_patch_basename(patch_name)
            except Exception as e:
                self.logger.error(f"❌ Error parsing patch name {patch_name}: {e}")
                continue

            global_cells = {}
            for cell_idx, cell in cells.items():
                local_centroid = cell.get("centroid", [0, 0])

                # Compute global coordinates
                x_global_px = local_centroid[0] + patch_x
                y_global_px = local_centroid[1] + patch_y
                global_centroid_px = [x_global_px, y_global_px]

                # Ensure required fields exist
                if "type" not in cell:
                    self.logger.error(
                        f"❌ Missing 'type' for cell in patch {patch_name}. Skipping."
                    )
                    continue

                # Convert to JSON serializable format
                cell_cleaned = {
                    key: (
                        value.tolist()
                        if isinstance(value, (np.ndarray, torch.Tensor))
                        else value
                    )
                    for key, value in cell.items()
                }
                cell_cleaned["global_centroid"] = global_centroid_px
                global_cells[cell_idx] = cell_cleaned

            global_cell_pred_dict[patch_name] = global_cells

        # Step 6: Generate annotation JSONs for monocytes, lymphocytes, and inflammatory cells
        inflammatory_dict, monocytes_dict, lymphocytes_dict = (
            generate_inflammatory_annotation_dicts(
                global_cell_pred_dict,
                micrometers_per_pixel=self.mpp_value,  # Now using class variable
                output_unit="mm",
                fixed_z_value=self.mpp_value,
            )
        )

        # Step 7: Save JSON files to specified output directory
        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)
            save_annotation_json(
                inflammatory_dict, self.output_path / "detected-inflammatory-cells.json"
            )
            save_annotation_json(
                monocytes_dict, self.output_path / "detected-monocytes.json"
            )
            save_annotation_json(
                lymphocytes_dict, self.output_path / "detected-lymphocytes.json"
            )
            self.logger.info(f"✅ JSON results saved in {self.output_path}")


# if __name__ == "__main__":
#     # configuration_parser = CellViTInfExpDetectionParser()
#     # configuration = configuration_parser.parse_arguments()

#     create_test_dataset(
#         wsi_path="/work/grana_urologia/MONKEY_challenge/source/sota_architectures/test_data/A_P000001_PAS_CPG.tif",
#         mask_path="/work/grana_urologia/MONKEY_challenge/data/monkey-data/images/tissue-masks/A_P000001_mask.tif",
#         output_dir="/work/grana_urologia/MONKEY_challenge/data/monkey_inference_test",
#         patch_shape=(256, 256, 3),
#         spacings=(0.25,),
#         overlap=(0, 0),
#         offset=(0, 0),
#         center=False,
#         cpus=4,
#     )

#     experiment = CellViTInfExpDetection(
#         logdir="/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/model_test/2025-02-08T145641_cellvit++ sam-h finetuning",  # clf needs to be in the path: "logdir/checkpoints/model_best.pth"
#         cellvit_path="/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/checkpoints/SAM/CellViT-SAM-H-x40-AMP.pth",
#         dataset_path="/work/grana_urologia/MONKEY_challenge/data/monkey_inference_test",
#         normalize_stains=False,
#         gpu="0",
#         input_shape=(256, 256),
#     )
#     experiment.run_inference()
