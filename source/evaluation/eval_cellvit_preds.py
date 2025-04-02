import json
import os
from pathlib import Path

# from pprint import pprint
from typing import Any, Dict, List, Tuple

import numpy as np
import openslide
from eval_plot_preds import compute_offline_metrics
from sklearn.model_selection import KFold  # , StratifiedKFold
from tqdm import tqdm

SPACING_LEVEL0 = 0.24199951445730394


def match_preds_gts(
    preds_dir: str, gt_dir: str, mask_dir: str
) -> Dict[str, Dict[str, str]]:
    """
    Match ground truth, prediction, and mask files based on patient IDs.

    Returns:
        Mapping from patient ID to a dictionary of GT, prediction, and mask file paths.
    """
    patient_data: Dict[str, Dict[str, str]] = {}
    patients_list: List[str] = []

    # Process ground truth files
    gt_path = Path(gt_dir)
    for file in gt_path.glob("*.json"):
        patient_id = "_".join(file.name.split("_")[:2])
        patients_list.append(patient_id)
        if patient_id not in patient_data:
            patient_data[patient_id] = {}
        if "inflammatory" in file.name:
            patient_data[patient_id]["gt_inflammatory"] = str(file.resolve())
        elif "lymphocytes" in file.name:
            patient_data[patient_id]["gt_lymphocyte"] = str(file.resolve())
        elif "monocytes" in file.name:
            patient_data[patient_id]["gt_monocyte"] = str(file.resolve())
        else:
            print(f"[WARNING] Unknown class in the filename: {file.name}")

    patients_set = set(patients_list)

    # Process mask files
    unmatched_masks: List[str] = []
    mask_path = Path(mask_dir)
    for file in mask_path.glob("*.tif"):
        if file.is_file():
            matched = False
            for patient_id in patients_set:
                if patient_id in file.name:
                    matched = True
                    if patient_id not in patient_data:
                        patient_data[patient_id] = {}
                    patient_data[patient_id]["mask"] = str(file.resolve())
                    break
            if not matched:
                unmatched_masks.append(file.name)
    if unmatched_masks:
        print("\n[WARNING] Mask files with unmatched patient IDs:")
        for f in unmatched_masks:
            print(f"- {f}")
    missing_masks = [
        pid for pid in patients_set if "mask" not in patient_data.get(pid, {})
    ]
    if missing_masks:
        print("\n[WARNING] Patients with GT but missing masks:")
        for pid in missing_masks:
            print(f"- {pid}")

    # Process prediction files
    unmatched_preds: List[str] = []
    preds_path = Path(preds_dir)
    for file in preds_path.glob("*.json"):
        if "cells" in file.name:
            matched = False
            for patient_id in patients_set:
                if patient_id in file.name:
                    matched = True
                    if patient_id not in patient_data:
                        patient_data[patient_id] = {}
                    patient_data[patient_id]["cellvit_preds"] = str(file.resolve())
                    break
            if not matched:
                unmatched_preds.append(file.name)
    if unmatched_preds:
        print("\n[WARNING] Prediction files with unmatched patient IDs:")
        for f in unmatched_preds:
            print(f"- {f}")
    missing_preds = [
        pid for pid in patients_set if "cellvit_preds" not in patient_data.get(pid, {})
    ]
    if missing_preds:
        print("\n[WARNING] Patients with GT but missing predictions:")
        for pid in missing_preds:
            print(f"- {pid}")

    return patient_data


def _build_annotation_json(
    points: List[Tuple[float, float, float]],
    annotation_name: str,
    fixed_z_value: float = SPACING_LEVEL0,
) -> dict:
    """
    Build the annotation dictionary with coordinates in millimeters.
    The input points are pixel coordinates, and fixed_z_value (in micrometers per pixel)
    is used to convert these to millimeters.

    The output dictionary is structured as:
      {
        "name": annotation_name,
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [
          {"name": "Point 0", "point": [x_mm, y_mm, z_mm], "probability": ...},
          ...
        ]
      }

    :param points: List of tuples (x_pixel, y_pixel, probability)
    :param annotation_name: Name of the annotation (e.g. "monocytes")
    :param fixed_z_value: Spacing level 0 in micrometers per pixel.
    :return: Annotation dictionary with coordinates converted to millimeters.
    """
    # Compute conversion factor (micrometers per pixel -> millimeters per pixel)
    conversion_factor = fixed_z_value / 1000.0

    annotation_dict = {
        "name": annotation_name,
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    for idx, (x_val, y_val, prob) in enumerate(points):
        # Convert x and y coordinates from pixels to mm using the conversion factor
        x_mm = x_val * conversion_factor
        y_mm = y_val * conversion_factor
        # Convert the fixed z value to mm
        z_mm = fixed_z_value / 1000.0
        annotation_dict["points"].append(
            {
                "name": f"Point {idx}",
                "point": [x_mm, y_mm, z_mm],
                "probability": prob,
            }
        )
    return annotation_dict


def create_annotations(
    filtered_preds: Dict[str, List[Tuple[float, float, float]]],
    only_inflammatory: bool = False,
    fixed_z: float = SPACING_LEVEL0,
) -> Tuple[dict, dict, dict]:
    """
    Create three annotation JSONs with names "monocytes", "lymphocytes", and "inflammatory-cells".

    If only_inflammatory is True, the same points (from the "inflammatory" key)
    are used for all three annotations. Otherwise, separate keys are used and the
    inflammatory annotation is the union of monocytes and lymphocytes.
    """
    if only_inflammatory:
        points = filtered_preds.get("inflammatory", [])
        monocyte_annotation = _build_annotation_json(points, "monocytes", fixed_z)
        lymphocyte_annotation = _build_annotation_json(points, "lymphocytes", fixed_z)
        inflammatory_annotation = _build_annotation_json(
            points, "inflammatory-cells", fixed_z
        )
        return monocyte_annotation, lymphocyte_annotation, inflammatory_annotation
    else:
        monocyte_points = filtered_preds.get("monocytes", [])
        lymphocyte_points = filtered_preds.get("lymphocytes", [])
        inflammatory_points = monocyte_points + lymphocyte_points

        monocyte_annotation = _build_annotation_json(
            monocyte_points, "monocytes", fixed_z
        )
        lymphocyte_annotation = _build_annotation_json(
            lymphocyte_points, "lymphocytes", fixed_z
        )
        inflammatory_annotation = _build_annotation_json(
            inflammatory_points, "inflammatory-cells", fixed_z
        )
        return monocyte_annotation, lymphocyte_annotation, inflammatory_annotation


def filter_points_openslide(
    point_dict: Dict[str, List[Tuple[float, float, float]]],
    mask_path: str,
    region_size: int = 3,
    position: int = 1,  # Position for the nested progress bar
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Filters points (with probabilities) using OpenSlide.
    For each (x, y, prob) tuple, reads a region (region_size x region_size) from the mask at level 0.
    Checks the center pixel of that region to decide if the point is inside the ROI.

    Expects a dictionary with keys mapping to lists of (x, y, prob) tuples.
    Returns a new dictionary with only those tuples whose (x, y) fall inside the mask.
    """
    slide = openslide.OpenSlide(mask_path)
    half_region = region_size // 2
    filtered_dict = {}

    # Initialize the nested progress bar
    total_points = sum(len(points) for points in point_dict.values())
    with tqdm(
        total=total_points, desc="Filtering points", position=position, leave=False
    ) as pbar:
        for label, points in point_dict.items():
            filtered_points = []
            for x, y, prob in points:
                region_x = int(round(x)) - half_region
                region_y = int(round(y)) - half_region
                region = slide.read_region(
                    (region_x, region_y), 0, (region_size, region_size)
                )
                center_pixel = region.getpixel((half_region, half_region))
                # Consider the point inside if the red channel is nonzero (adjust threshold if needed)
                if center_pixel[0] != 0:
                    filtered_points.append((x, y, prob))
                pbar.update(1)  # Update the progress bar for each point processed
            filtered_dict[label] = filtered_points
    slide.close()
    return filtered_dict


def parse_cells_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Extracts the probability, type (class) and centroid position (x, y) of each cell
    from the JSON file.

    Expected JSON structure:
    {
      "wsi_metadata": {...},
      "type_map": {"1": "Neoplastic", "2": "Inflammatory", ...},
      "cells": [
         {
           "bbox": [...],
           "centroid": [x, y],
           "contour": [...],
           "type_prob": <float>,
           "type": <int>,
           ...
         },
         ...
      ]
    }

    Returns:
        A list of dictionaries with keys:
          - "class": the cell class (converted using type_map)
          - "probability": the cell's type probability
          - "centroid": the [x, y] position
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    type_map = data.get("type_map", {})
    cells = data.get("cells", [])
    extracted = []
    for cell in cells:
        cell_type = cell.get("type")
        cell_class = type_map.get(str(cell_type), "Unknown")
        probability = cell.get("type_prob")
        centroid = cell.get("centroid", [None, None])
        extracted.append(
            {
                "class": cell_class,
                "probability": probability,
                "centroid": centroid,
            }
        )
    return extracted


def process_predictions(
    patient_data: Dict[str, Dict[str, str]],
    output_dir: str,
    only_inflammatory: bool = False,
    fixed_z: float = SPACING_LEVEL0,
) -> None:
    """
    For each patient, process the predictions:
      - Parse the cell JSON to extract probability, type and centroid.
      - Optionally filter points using the corresponding tissue mask (via OpenSlide).
      - Create three annotation JSONs: "detected-monocytes.json", "detected-lymphocytes.json", and "detected-inflammatory-cells.json".
    """
    with tqdm(
        total=len(patient_data), desc="Processing patients", position=0
    ) as main_pbar:
        for patient_id, data in patient_data.items():
            # Update the description to show the current patient ID
            main_pbar.set_description(f"Processing patient: {patient_id}")

            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)

            data = patient_data[patient_id]
            roi_mask_path = data.get("mask")
            preds_path = data.get("cellvit_preds")

            cells = parse_cells_json(preds_path)
            print(f"\nProcessing {patient_id} with {len(cells)} cells")

            preds_points: Dict[str, List[Tuple[float, float, float]]] = {}
            if only_inflammatory:
                inflammatory_points = [
                    (cell["centroid"][0], cell["centroid"][1], cell["probability"])
                    for cell in cells
                    if cell["class"].lower() == "inflammatory"
                ]
                preds_points["inflammatory"] = inflammatory_points
            else:
                preds_points["monocytes"] = [
                    (cell["centroid"][0], cell["centroid"][1], cell["probability"])
                    for cell in cells
                    if cell["class"].lower() == "monocytes"
                ]
                preds_points["lymphocytes"] = [
                    (cell["centroid"][0], cell["centroid"][1], cell["probability"])
                    for cell in cells
                    if cell["class"].lower() == "lymphocytes"
                ]
                preds_points["inflammatory"] = (
                    preds_points["monocytes"] + preds_points["lymphocytes"]
                )

            if roi_mask_path:
                preds_points = filter_points_openslide(
                    preds_points, roi_mask_path, region_size=3
                )

            if only_inflammatory:
                ann_mon, ann_lym, ann_infl = create_annotations(
                    preds_points,
                    only_inflammatory=True,
                    fixed_z=fixed_z,
                )
            else:
                ann_mon, ann_lym, ann_infl = create_annotations(
                    preds_points,
                    only_inflammatory=False,
                    fixed_z=fixed_z,
                )

            for json_data, filename in zip(
                [ann_mon, ann_lym, ann_infl],
                [
                    "detected-monocytes.json",
                    "detected-lymphocytes.json",
                    "detected-inflammatory-cells.json",
                ],
            ):
                out_path = os.path.join(patient_output_dir, f"{patient_id}_{filename}")
                with open(out_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                print(f"Saved {out_path}")

            main_pbar.update(
                1
            )  # Update the main progress bar after processing each patient


def main():
    SEED = 42
    N_FOLDS = 5

    preds_dir = "/work/grana_urologia/MONKEY_challenge/outputs/cellvit_baseline/predictions_sam-h_baseline_all_dataset"
    gt_dir = (
        "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/json_mm"
    )
    mask_dir = (
        "/work/grana_urologia/MONKEY_challenge/data/monkey-data/images/tissue-masks"
    )
    output_dir = (
        "/work/grana_urologia/MONKEY_challenge/outputs/cellvit_baseline/json_preds"
    )

    metrics_dir = (
        "/work/grana_urologia/MONKEY_challenge/outputs/cellvit_baseline/scores"
    )

    print("Matching ground truth, predictions, and masks...")
    patient_data = match_preds_gts(preds_dir, gt_dir, mask_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("Matching done.\nProcessing predictions...")

    # Evaluate the predictions on the n folds of the dataset

    # 1. split the dataset into n folds
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # extract the patient IDs
    patient_ids = list(patient_data.keys())

    # 2. for each fold, compute the metrics, plot the FROC curve, and save the results
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patient_ids)):
        print(f"\nProcessing fold {fold_idx}...")
        # create a new metrics directory for the fold
        metrics_fold_dir = os.path.join(metrics_dir, f"fold_{fold_idx}")
        os.makedirs(metrics_fold_dir, exist_ok=True)

        # create a new directory for the current fold
        fold_preds_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_preds_dir, exist_ok=True)

        # get the patient IDs for the current fold
        # train_patients = [patient_ids[idx] for idx in train_idx]
        test_patients = [patient_ids[idx] for idx in test_idx]
        fold_patient_data = {pid: patient_data[pid] for pid in test_patients}

        # 2a. process the predictions for the current fold
        # Set only_inflammatory to True if predictions contain only the "Inflammatory" label.
        process_predictions(fold_patient_data, fold_preds_dir, only_inflammatory=True)

        print(f"Processing metrics for fold {fold_idx}...")

        # 2b. compute the scores and FROC curves based on the current fold and save the results
        compute_offline_metrics(
            preds_dir=fold_preds_dir,
            ground_truth_dir=gt_dir,
            save_dir=metrics_fold_dir,
            metrics_filename=f"metrics_fold_{fold_idx}.json",
            plot_froc=True,
            plot_froc_single_wsis=True,
            froc_plot_filename=f"froc_curve_fold_{fold_idx}.png",
        )

        print(f"Metrics and FROC curve for fold {fold_idx} saved in {metrics_fold_dir}")

    # 3. aggregate the results from all folds
    # compute the mean and standard deviation of the FROC scores
    froc_scores_infl = []
    froc_scores_mon = []
    froc_scores_lym = []
    for fold_idx in range(N_FOLDS):
        metrics_fold_dir = os.path.join(metrics_dir, f"fold_{fold_idx}")
        metrics_file_path = os.path.join(
            metrics_fold_dir, f"metrics_fold_{fold_idx}.json"
        )
        with open(metrics_file_path, "r") as f:
            metrics = json.load(f)
        froc_score_inflammatory = metrics["aggregates"]["inflammatory-cells"][
            "froc_score_aggr"
        ]
        froc_score_monocytes = metrics["aggregates"]["monocytes"]["froc_score_aggr"]
        froc_score_lymphocytes = metrics["aggregates"]["lymphocytes"]["froc_score_aggr"]
        froc_scores_infl.append(froc_score_inflammatory)
        froc_scores_mon.append(froc_score_monocytes)
        froc_scores_lym.append(froc_score_lymphocytes)

    mean_froc_inf = np.mean(froc_scores_infl)
    std_froc_inf = np.std(froc_scores_infl)
    mean_froc_mon = np.mean(froc_scores_mon)
    std_froc_mon = np.std(froc_scores_mon)
    mean_froc_lym = np.mean(froc_scores_lym)
    std_froc_lym = np.std(froc_scores_lym)

    print(f"Mean FROC scores for inflammatory-cells: {mean_froc_inf} ± {std_froc_inf}")
    print(f"Mean FROC scores for monocytes: {mean_froc_mon} ± {std_froc_mon}")
    print(f"Mean FROC scores for lymphocytes: {mean_froc_lym} ± {std_froc_lym}")

    # save the results in a json file in the metrics_dir
    froc_scores = {
        "inflammatory_cells": {"mean": mean_froc_inf, "std": std_froc_inf},
        "monocytes": {"mean": mean_froc_mon, "std": std_froc_mon},
        "lymphocytes": {"mean": mean_froc_lym, "std": std_froc_lym},
    }
    froc_scores_path = os.path.join(metrics_dir, "froc_scores_overall_folds.json")
    with open(froc_scores_path, "w") as f:
        json.dump(froc_scores, f, indent=2)

    print(f"FROC scores fold aggregation saved in {froc_scores_path}")


if __name__ == "__main__":
    main()
