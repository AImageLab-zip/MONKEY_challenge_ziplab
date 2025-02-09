import json
import os
from pathlib import Path
from pprint import pformat, pprint

import monai.metrics as mm
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import auc
from tqdm import tqdm

# ---------------------------------------------------------------------
# Constants -- tweak to your needs
# ---------------------------------------------------------------------
SPACING_LEVEL0 = 0.24199951445730394
GT_MM = True  # Whether ground-truth coords are in mm


# ---------------------------------------------------------------------
# Core evaluation workflow
# ---------------------------------------------------------------------
def eval_metrics(
    predictions_folder, ground_truth_folder, save_path, filename="metrics.json"
):
    """
    Evaluate predictions vs. ground truths, then save the metrics JSON.
    """
    metrics = evaluate_predictions(predictions_folder, ground_truth_folder)
    output_file = os.path.join(save_path, filename)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # pprint(metrics)
    # print overall froc scores for each cell type

    return metrics


def evaluate_predictions(predictions_folder, ground_truth_folder):
    """
    Loops through all patient subfolders in `predictions_folder`, processes each,
    and returns a metrics dictionary with per-slide and aggregated metrics.
    """
    predictions_path = Path(predictions_folder)
    gt_path = Path(ground_truth_folder)

    # Collect results for each patient
    results = []
    progress_bar = tqdm(predictions_path.iterdir(), desc="Evaluating WSIs...")
    for subfolder in progress_bar:
        if subfolder.is_dir():
            patient_id = subfolder.name  # e.g. "A_P000001"
            progress_bar.set_postfix_str(f"Patient {subfolder.name}")
            result = process_patient(patient_id, predictions_path, gt_path)
            if result is not None:
                results.append(result)

    # Convert patient-wise results to the final metrics dict
    file_ids = [r[0] for r in results]
    metrics_per_slide = [r[1] for r in results]
    metrics = {"per_slide": {}}

    # Insert each patient's metrics
    for i, file_id in enumerate(file_ids):
        metrics["per_slide"][file_id] = metrics_per_slide[i]

    # Build aggregated metrics
    lymphocytes_metrics = format_metrics_for_aggr(metrics_per_slide, "lymphocytes")
    monocytes_metrics = format_metrics_for_aggr(metrics_per_slide, "monocytes")
    inflammatory_cells_metrics = format_metrics_for_aggr(
        metrics_per_slide, "inflammatory-cells"
    )

    aggregated_metrics = {
        "lymphocytes": get_aggr_froc(lymphocytes_metrics),
        "monocytes": get_aggr_froc(monocytes_metrics),
        "inflammatory-cells": get_aggr_froc(inflammatory_cells_metrics),
    }
    metrics["aggregates"] = aggregated_metrics

    # Optionally remove unneeded verbose info from per-slide
    for _, file_metrics in metrics["per_slide"].items():
        for cell_type in ["lymphocytes", "monocytes", "inflammatory-cells"]:
            for k in ["fp_probs_slide", "tp_probs_slide", "total_pos_slide"]:
                file_metrics[cell_type].pop(k, None)

    return metrics


def process_patient(patient_id, predictions_path, gt_path):
    """
    Processes a single patient's predictions, compares them with ground-truth files,
    and returns (patient_id, patient_metrics_dict).
    """
    # Paths to predictions
    location_detected_lymphocytes = (
        predictions_path / patient_id / "detected-lymphocytes.json"
    )
    location_detected_monocytes = (
        predictions_path / patient_id / "detected-monocytes.json"
    )
    location_detected_inflammatory_cells = (
        predictions_path / patient_id / "detected-inflammatory-cells.json"
    )

    # If one of these doesn't exist, skip
    if not (
        location_detected_lymphocytes.exists()
        and location_detected_monocytes.exists()
        and location_detected_inflammatory_cells.exists()
    ):
        print(f"WARNING: Missing predictions for {patient_id}, skipping.")
        return None

    # Load predictions
    result_detected_lymphocytes = load_json_file(location_detected_lymphocytes)
    result_detected_monocytes = load_json_file(location_detected_monocytes)
    result_detected_inflammatory_cells = load_json_file(
        location_detected_inflammatory_cells
    )

    # Convert mm to pixel if needed
    if not GT_MM:
        result_detected_inflammatory_cells = convert_mm_to_pixel(
            result_detected_inflammatory_cells
        )
        result_detected_monocytes = convert_mm_to_pixel(result_detected_monocytes)
        result_detected_lymphocytes = convert_mm_to_pixel(result_detected_lymphocytes)

    # For ground-truth, assume files named: A_P000001_lymphocytes.json, etc.
    # If your folder uses a different pattern, adjust accordingly.
    file_id = patient_id
    gt_lymphocytes = load_json_file(gt_path / f"{file_id}_lymphocytes.json")
    gt_monocytes = load_json_file(gt_path / f"{file_id}_monocytes.json")
    gt_inflammatory_cells = load_json_file(
        gt_path / f"{file_id}_inflammatory-cells.json"
    )

    # Radii used for matching predictions to GT
    radius_lymph = 0.004 if GT_MM else int(4 / SPACING_LEVEL0)
    radius_mono = 0.005 if GT_MM else int(5 / SPACING_LEVEL0)
    radius_infl = 0.005 if GT_MM else int(5 / SPACING_LEVEL0)

    # Compute FROC metrics for each cell type
    lymphocytes_froc = get_froc_vals(
        gt_lymphocytes, result_detected_lymphocytes, radius=radius_lymph
    )
    monocytes_froc = get_froc_vals(
        gt_monocytes, result_detected_monocytes, radius=radius_mono
    )
    inflamm_froc = get_froc_vals(
        gt_inflammatory_cells, result_detected_inflammatory_cells, radius=radius_infl
    )

    # Return the metrics for this patient
    metrics_dict = {
        "lymphocytes": lymphocytes_froc,
        "monocytes": monocytes_froc,
        "inflammatory-cells": inflamm_froc,
    }

    return (file_id, metrics_dict)


# ---------------------------------------------------------------------
# FROC logic, same as your original
# ---------------------------------------------------------------------
def get_froc_vals(gt_dict, result_dict, radius: int):
    """
    Computes the Free-Response Receiver Operating Characteristic (FROC) values for given ground truth and result data.
    Using https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    Args:
        gt_dict (dict): Ground truth data containing points and regions of interest (ROIs).
        result_dict (dict): Result data containing detected points and their probabilities.
        radius (int): The maximum distance in pixels for considering a detection as a true positive.

    Returns:
        dict: A dictionary containing FROC metrics such as sensitivity, false positives per mm²,
              true positive probabilities, false positive probabilities, total positives,
              area in mm², and FROC score.
    """
    if len(result_dict["points"]) == 0:
        return {
            "sensitivity_slide": [0],
            "fp_per_mm2_slide": [0],
            "fp_probs_slide": [0],
            "tp_probs_slide": [0],
            "total_pos_slide": 0,
            "area_mm2_slide": 0,
            "froc_score_slide": 0,
        }
    if len(gt_dict["points"]) == 0:
        return {}

    gt_coords = [i["point"] for i in gt_dict["points"]]
    # rois not used heavily, but we keep the logic
    gt_rois = [i["polygon"] for i in gt_dict["rois"]]
    area_mm2 = SPACING_LEVEL0 * SPACING_LEVEL0 * gt_dict["area_rois"] / 1000000

    result_prob = [i["probability"] for i in result_dict["points"]]
    result_coords = [[i["point"][0], i["point"][1]] for i in result_dict["points"]]

    true_positives, false_negatives, false_positives, tp_probs, fp_probs = (
        match_coordinates(gt_coords, result_coords, result_prob, radius)
    )
    total_pos = len(gt_coords)
    sensitivity, fp_per_mm2_slide, froc_score = get_froc_score(
        fp_probs, tp_probs, total_pos, area_mm2
    )

    return {
        "sensitivity_slide": list(sensitivity),
        "fp_per_mm2_slide": list(fp_per_mm2_slide),
        "fp_probs_slide": list(fp_probs),
        "tp_probs_slide": list(tp_probs),
        "total_pos_slide": total_pos,
        "area_mm2_slide": area_mm2,
        "froc_score_slide": float(froc_score),
    }


def match_coordinates(ground_truth, predictions, pred_prob, margin):
    """
    Matches predicted coordinates to ground truth coordinates within a certain distance margin
    and computes the associated probabilities for true positives and false positives.

    Args:
        ground_truth (list of tuples): List of ground truth coordinates as (x, y).
        predictions (list of tuples): List of predicted coordinates as (x, y).
        pred_prob (list of floats): List of probabilities associated with each predicted coordinate.
        margin (float): The maximum distance for considering a prediction as a true positive.

    Returns:
        true_positives (int): Number of correctly matched predictions.
        false_negatives (int): Number of ground truth coordinates not matched by any prediction.
        false_positives (int): Number of predicted coordinates not matched by any ground truth.
        tp_probs (list of floats): Probabilities of the true positive predictions.
        fp_probs (list of floats): Probabilities of the false positive predictions.
    """
    if len(ground_truth) == 0 and len(predictions) == 0:
        return 0, 0, 0, np.array([]), np.array([])

    gt_array = np.array(ground_truth)
    pred_array = np.array(predictions)
    pred_prob_array = np.array(pred_prob)

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

    true_positives = len(matched_gt)
    false_negatives = len(ground_truth) - true_positives
    false_positives = len(predictions) - true_positives

    tp_probs = [pred_prob_array[i] for i in matched_pred]
    fp_probs = [
        pred_prob_array[i] for i in range(len(predictions)) if i not in matched_pred
    ]

    return (
        true_positives,
        false_negatives,
        false_positives,
        np.array(tp_probs),
        np.array(fp_probs),
    )


def get_froc_score(fp_probs, tp_probs, total_pos, area_mm2):
    eval_thresholds = (10, 20, 50, 100, 200, 300)

    fp_per_mm2, sensitivity = mm.compute_froc_curve_data(
        fp_probs, tp_probs, total_pos, area_mm2
    )
    if len(fp_per_mm2) == 0 and len(sensitivity) == 0:
        return sensitivity, fp_per_mm2, 0

    if len(sensitivity) == 1:
        # Only one TP => handle edge case
        sensitivity = [1]
        fp_per_mm2 = [len(fp_probs) / area_mm2]
        froc_score = np.mean([int(fp_per_mm2[0] < i) for i in eval_thresholds])
    else:
        froc_score = mm.compute_froc_score(
            fp_per_mm2, sensitivity, eval_thresholds=eval_thresholds
        )

    return sensitivity, fp_per_mm2, froc_score


# ---------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------
def format_metrics_for_aggr(metrics_list, cell_type):
    """
    Formats each slide’s metrics into a structure used by get_aggr_froc().
    """
    aggr = {}
    for d in [m[cell_type] for m in metrics_list]:
        for key, value in d.items():
            if key not in aggr:
                aggr[key] = []
            if isinstance(value, list):
                # remove None if any
                value = [x for x in value if x is not None]
            aggr[key].append(value)
    return aggr


def get_aggr_froc(metrics_dict):
    if len(metrics_dict) == 0:
        return {
            "sensitivity_aggr": [0],
            "fp_per_mm2_aggr": [0],
            "area_mm2_aggr": 0,
            "froc_score_aggr": 0,
        }
    fp_probs = np.array(
        [item for sublist in metrics_dict["fp_probs_slide"] for item in sublist]
    )
    tp_probs = np.array(
        [item for sublist in metrics_dict["tp_probs_slide"] for item in sublist]
    )
    total_pos = sum(metrics_dict["total_pos_slide"])
    area_mm2 = sum(metrics_dict["area_mm2_slide"])
    if total_pos == 0:
        return {
            "sensitivity_aggr": [0],
            "fp_per_mm2_aggr": [0],
            "area_mm2_aggr": area_mm2,
            "froc_score_aggr": 0,
        }

    sensitivity_overall, fp_per_mm2, froc_score_overall = get_froc_score(
        fp_probs, tp_probs, total_pos, area_mm2
    )
    return {
        "sensitivity_aggr": list(sensitivity_overall),
        "fp_per_mm2_aggr": list(fp_per_mm2),
        "area_mm2_aggr": area_mm2,
        "froc_score_aggr": float(froc_score_overall),
    }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_json_file(location):
    with open(location, "r") as f:
        return json.load(f)


def convert_mm_to_pixel(data_dict, spacing=SPACING_LEVEL0):
    # Convert coordinates in mm to pixels
    points_pixels = []
    for d in data_dict["points"]:
        if len(d["point"]) == 2:
            d["point"] = [
                mm_to_pixel(d["point"][0], spacing),
                mm_to_pixel(d["point"][1], spacing),
                0,
            ]
        else:
            d["point"] = [
                mm_to_pixel(d["point"][0], spacing),
                mm_to_pixel(d["point"][1], spacing),
                mm_to_pixel(d["point"][2], spacing),
            ]
        points_pixels.append(d)
    data_dict["points"] = points_pixels
    return data_dict


def mm_to_pixel(dist, spacing=SPACING_LEVEL0):
    spacing_px = spacing / 1000
    return int(round(dist / spacing_px))


if __name__ == "__main__":
    # Example usage
    eval_metrics(
        predictions_folder="/work/grana_urologia/MONKEY_challenge/data/eval_test_cellvit/adjusted_tresh_plus_clf_tresh/",
        ground_truth_folder="/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/json_mm",
        save_path="/work/grana_urologia/MONKEY_challenge/data/eval_test_cellvit/adjusted_tresh_plus_clf_tresh/",
    )
