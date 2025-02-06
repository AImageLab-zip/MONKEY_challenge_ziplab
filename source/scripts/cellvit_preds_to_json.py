import json
import os

import ijson
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask


###############################################################################
# 1) STREAM-PARSE POINTS FROM (CUSTOM) JSON OR GEOJSON
###############################################################################
def parse_points_stream(
    json_path, mode="json", monocyte_label="monocytes", lymphocyte_label="lymphocytes"
):
    monocyte_points = []
    lymphocyte_points = []

    if mode.lower() == "geojson":
        with open(json_path, "r") as f:
            parser = ijson.items(f, "features.item")
            for feature in parser:
                props = feature.get("properties", {}).get("classification", {})
                class_name = props.get("name", "").lower()
                coords = feature.get("geometry", {}).get("coordinates", [])

                if feature.get("geometry", {}).get("type") == "MultiPoint":
                    for x_px, y_px in coords:
                        x_px = float(x_px)
                        y_px = float(y_px)
                        if class_name == monocyte_label.lower():
                            monocyte_points.append((x_px, y_px))
                        elif class_name == lymphocyte_label.lower():
                            lymphocyte_points.append((x_px, y_px))

    else:
        with open(json_path, "r") as f_all:
            pre_data = json.load(f_all)
            type_map = pre_data.get("type_map", {})

        with open(json_path, "r") as f:
            parser = ijson.items(f, "cells.item")
            for cell in parser:
                label_str = type_map.get(str(cell["type"]), "").lower()
                x_px, y_px = cell.get("centroid", [0, 0])

                x_px = float(x_px)
                y_px = float(y_px)

                if label_str == monocyte_label.lower():
                    monocyte_points.append((x_px, y_px))
                elif label_str == lymphocyte_label.lower():
                    lymphocyte_points.append((x_px, y_px))

    return monocyte_points, lymphocyte_points


###############################################################################
# 2) LOAD MASK USING RASTERIO AT DOWNSAMPLED LEVEL
###############################################################################
def load_mask_rasterio(mask_path, downsample_factor=1):
    """
    Reads a pyramidal TIFF using Rasterio at the specified downsample factor.
    Returns a 2D NumPy mask array and a transform for coordinate conversion.
    """
    with rasterio.open(mask_path) as src:
        new_width = int(src.width / downsample_factor)
        new_height = int(src.height / downsample_factor)
        mask_array = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.nearest,
        )
        transform = src.transform * src.transform.scale(
            (src.width / new_width), (src.height / new_height)
        )
    return mask_array, transform


###############################################################################
# 3) FILTER POINTS BASED ON RASTERIO MASK
###############################################################################
def filter_points_in_mask(points_px, mask_array, transform):
    """
    Filters points based on whether they fall in the mask.
    Assumes (x, y) are in pixel coordinates if no valid transform exists.
    """
    kept_points = []

    # Check if the transform is an identity matrix (meaning no georeferencing)
    is_identity_transform = (
        transform.a == 1
        and transform.b == 0
        and transform.c == 0
        and transform.d == 0
        and transform.e == 1
        and transform.f == 0
    )

    for point in points_px:
        x, y = point  # Ensure tuple unpacking

        if is_identity_transform:
            col, row = int(round(x)), int(round(y))  # Direct pixel indexing
        else:
            # Transform from image coordinates to mask indices
            row, col = ~transform * (x, y)
            row, col = int(round(row)), int(round(col))

        if 0 <= row < mask_array.shape[0] and 0 <= col < mask_array.shape[1]:
            if mask_array[row, col] != 0:
                kept_points.append((x, y))

    return kept_points


###############################################################################
# 4) MAIN FUNCTION: parse_inflammatory_data
###############################################################################
def parse_inflammatory_data(
    json_path,
    mode="json",
    real_mpp=0.24199951445730394,
    z_spacing=0.25,
    monocyte_label="Inflammatory",
    lymphocyte_label="lymphocytes",
    mask_tif_path=None,
    downsample_factor=4,
):
    """
    1. Parse points from JSON/GeoJSON.
    2. Load downsampled mask via Rasterio and filter points.
    3. Convert pixel coordinates to mm and return final dicts.
    """
    # 1) Parse points
    monocyte_pts_px, lymphocyte_pts_px = parse_points_stream(
        json_path=json_path,
        mode=mode,
        monocyte_label=monocyte_label,
        lymphocyte_label=lymphocyte_label,
    )

    # 2) Load mask and filter points
    if mask_tif_path:
        mask_array, transform = load_mask_rasterio(mask_tif_path, downsample_factor)
        monocyte_pts_filtered = filter_points_in_mask(
            monocyte_pts_px, mask_array, transform
        )
        lymphocyte_pts_filtered = filter_points_in_mask(
            lymphocyte_pts_px, mask_array, transform
        )
    else:
        monocyte_pts_filtered = monocyte_pts_px
        lymphocyte_pts_filtered = lymphocyte_pts_px

    # 3) Convert pixel coords => mm coords
    px_to_mm = real_mpp / 1000.0
    convert_to_mm = lambda pts: [
        {"point": [x * px_to_mm, y * px_to_mm, z_spacing]} for x, y in pts
    ]

    return {
        "monocytes": {
            "type": "Multiple points",
            "points": convert_to_mm(monocyte_pts_filtered),
        },
        "lymphocytes": {
            "type": "Multiple points",
            "points": convert_to_mm(lymphocyte_pts_filtered),
        },
        "inflammatory_cells": {
            "type": "Multiple points",
            "points": convert_to_mm(monocyte_pts_filtered + lymphocyte_pts_filtered),
        },
    }


###############################################################################
# 6) EXAMPLE USAGE
###############################################################################
if __name__ == "__main__":
    # Example parameters (edit paths to your environment):
    mode = "json"  # or "geojson"
    mask_tif_path = "/work/grana_urologia/MONKEY_challenge/data/monkey-data/images/tissue-masks/A_P000001_mask.tif"
    json_file_path = "/work/grana_urologia/MONKEY_challenge/source/sota_architectures/test_data/outputs/A_P000001_PAS_CPG_cell_detection.json"
    save_dir = "/work/grana_urologia/MONKEY_challenge/source/sota_architectures/transformed_preds/A_P000001"
    os.makedirs(save_dir, exist_ok=True)

    # Parse with OpenSlide-level-based approach
    results = parse_inflammatory_data(
        json_path=json_file_path,
        mode=mode,
        real_mpp=0.24199951445730394,
        z_spacing=0.25,
        monocyte_label="Inflammatory",
        lymphocyte_label="lymphocytes",
        mask_tif_path=mask_tif_path,
        downsample_factor=4,
    )

    for key, data in results.items():
        with open(os.path.join(save_dir, f"{key}.json"), "w") as f:
            json.dump(data, f, indent=2)

    print("✅ Done! Results saved in:", save_dir)
