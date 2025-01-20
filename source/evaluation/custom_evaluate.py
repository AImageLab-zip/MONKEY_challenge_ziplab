import json
import os
from pathlib import Path
from pprint import pprint

import numpy as np
from monai.metrics import compute_froc_curve_data
from scipy.spatial import distance

# Constants for evaluation
SPACING_LEVEL0 = 0.24199951445730394  # um/pixel
GT_MM = True  # Ground truth and predictions are in millimeters

# Radius constants in millimeters
LYMPHOCYTE_RADIUS_MM = 0.004  # 4 um
MONOCYTE_RADIUS_MM = 0.005  # 5 um
IMMUNE_CELL_RADIUS_MM = 0.005  # 5 um


def load_json_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def compute_froc(gt_coords, predictions, probs, radius, area_mm2):
    """Compute FROC metrics for a single slide."""
    true_positives, false_negatives, false_positives, tp_probs, fp_probs = (
        match_coordinates(gt_coords, predictions, probs, radius)
    )
    total_pos = len(gt_coords)
    sensitivity, fp_per_mm2, froc_score = compute_froc_curve_data(
        fp_probs, tp_probs, total_pos, area_mm2
    )
    return {
        "sensitivity": list(sensitivity),
        "fp_per_mm2": list(fp_per_mm2),
        "froc_score": froc_score,
        "total_positives": total_pos,
        "area_mm2": area_mm2,
    }


def match_coordinates(gt_coords, pred_coords, pred_probs, margin):
    """Match ground truth and prediction coordinates within a margin."""
    gt_array = np.array(gt_coords)
    pred_array = np.array(pred_coords)
    dist_matrix = distance.cdist(gt_array, pred_array)
    matched_gt = set()
    matched_pred = set()
    while True:
        min_dist = np.min(dist_matrix)
        if min_dist > margin or min_dist == np.inf:
            break
        gt_idx, pred_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        dist_matrix[gt_idx, :] = np.inf
        dist_matrix[:, pred_idx] = np.inf
    tp_probs = [pred_probs[i] for i in matched_pred]
    fp_probs = [pred_probs[i] for i in range(len(pred_coords)) if i not in matched_pred]
    return (
        len(matched_gt),
        len(gt_coords) - len(matched_gt),
        len(pred_coords) - len(matched_gt),
        tp_probs,
        fp_probs,
    )


def evaluate_predictions(predictions_folder, ground_truth_folder):
    predictions_folder = Path(predictions_folder)
    ground_truth_folder = Path(ground_truth_folder)
    metrics = {}

    for patient_folder in predictions_folder.iterdir():
        if not patient_folder.is_dir():
            continue
        patient_id = patient_folder.name
        detected_lymphocytes = load_json_file(
            patient_folder / "detected-lymphocytes.json"
        )
        detected_monocytes = load_json_file(patient_folder / "detected-monocytes.json")
        detected_inflammatory_cells = load_json_file(
            patient_folder / "detected-inflammatory-cells.json"
        )

        gt_lymphocytes = load_json_file(
            ground_truth_folder / f"{patient_id}_lymphocytes.json"
        )
        gt_monocytes = load_json_file(
            ground_truth_folder / f"{patient_id}_monocytes.json"
        )
        gt_inflammatory_cells = load_json_file(
            ground_truth_folder / f"{patient_id}_inflammatory-cells.json"
        )

        area_mm2 = (
            SPACING_LEVEL0 * SPACING_LEVEL0 * gt_lymphocytes["area_rois"] / 1_000_000
        )

        lymph_metrics = compute_froc(
            gt_coords=[p["point"] for p in gt_lymphocytes["points"]],
            predictions=[p["point"] for p in detected_lymphocytes["points"]],
            probs=[p["probability"] for p in detected_lymphocytes["points"]],
            radius=LYMPHOCYTE_RADIUS_MM,
            area_mm2=area_mm2,
        )

        mono_metrics = compute_froc(
            gt_coords=[p["point"] for p in gt_monocytes["points"]],
            predictions=[p["point"] for p in detected_monocytes["points"]],
            probs=[p["probability"] for p in detected_monocytes["points"]],
            radius=MONOCYTE_RADIUS_MM,
            area_mm2=area_mm2,
        )

        inflam_metrics = compute_froc(
            gt_coords=[p["point"] for p in gt_inflammatory_cells["points"]],
            predictions=[p["point"] for p in detected_inflammatory_cells["points"]],
            probs=[p["probability"] for p in detected_inflammatory_cells["points"]],
            radius=IMMUNE_CELL_RADIUS_MM,
            area_mm2=area_mm2,
        )

        metrics[patient_id] = {
            "lymphocytes": lymph_metrics,
            "monocytes": mono_metrics,
            "inflammatory_cells": inflam_metrics,
        }

    return metrics


def eval_metrics(predictions_folder, ground_truth_folder, save_path, filename="metrics.json"):
    metrics = evaluate_predictions(predictions_folder, ground_truth_folder)

    output_file = os.path.join(save_path, filename)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    pprint(metrics)


def main():
    predictions_folder = "path/to/predictions"
    ground_truth_folder = "path/to/ground_truth"

    metrics = evaluate_predictions(predictions_folder, ground_truth_folder)

    output_file = Path("metrics.json")
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    pprint(metrics)


if __name__ == "__main__":
    main()
