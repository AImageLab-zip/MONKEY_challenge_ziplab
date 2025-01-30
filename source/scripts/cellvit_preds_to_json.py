import json
import os

import ijson
import numpy as np
import openslide  # pip install openslide-python
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Allows handling very large images if needed


###############################################################################
# 1) STREAM-PARSE POINTS FROM (CUSTOM) JSON OR GEOJSON
###############################################################################
def parse_points_stream(
    json_path,
    mode="json",
    monocyte_label="monocytes",
    lymphocyte_label="lymphocytes",
):
    """
    Memory-efficiently parse monocyte & lymphocyte detections from:
      - 'custom JSON' mode => uses "type_map" + "cells"
      - 'geojson' mode => uses "features" with "MultiPoint" coords
    Returns two lists of (x_px, y_px).
    """
    monocyte_points = []
    lymphocyte_points = []

    if mode.lower() == "geojson":
        # Example GeoJSON structure:
        # {
        #   "type": "FeatureCollection",
        #   "features": [
        #       {
        #           "properties": { "classification": { "name": "<label>" } },
        #           "geometry": { "type": "MultiPoint", "coordinates": [[x_px, y_px], ...] }
        #       },
        #       ...
        #   ]
        # }
        with open(json_path, "r") as f:
            parser = ijson.items(f, "features.item")
            for feature in parser:
                props = feature.get("properties", {})
                class_dict = props.get("classification", {})
                class_name = class_dict.get("name", "").lower()

                geom = feature.get("geometry", {})
                geom_type = geom.get("type", "")
                coords = geom.get("coordinates", [])

                if geom_type == "MultiPoint":
                    for x_px, y_px in coords:
                        if class_name == monocyte_label.lower():
                            monocyte_points.append((x_px, y_px))
                        elif class_name == lymphocyte_label.lower():
                            lymphocyte_points.append((x_px, y_px))

    else:
        # Custom JSON structure:
        # {
        #   "type_map": { "1": "Inflammatory", "2": "lymphocytes", ... },
        #   "cells": [
        #       { "type": 1, "centroid": [x_px, y_px] },
        #       ...
        #   ]
        # }
        # => We'll ijson-stream the "cells" array, but we first need the type_map
        with open(json_path, "r") as f_all:
            pre_data = json.load(f_all)
            type_map = pre_data.get("type_map", {})

        with open(json_path, "r") as f:
            parser = ijson.items(f, "cells.item")
            for cell in parser:
                numeric_type = str(cell["type"])
                label_str = type_map.get(numeric_type, "").lower()
                x_px, y_px = cell.get("centroid", [0, 0])

                if label_str == monocyte_label.lower():
                    monocyte_points.append((x_px, y_px))
                elif label_str == lymphocyte_label.lower():
                    lymphocyte_points.append((x_px, y_px))

    return monocyte_points, lymphocyte_points


###############################################################################
# 2) LOAD DOWNSAMPLED MASK FROM A PYRAMIDAL TIFF WITH OPENSLIDE
###############################################################################
def load_mask_openslide(mask_path, openslide_level=1):
    """
    Reads a (pyramidal) TIFF using OpenSlide at the specified 'openslide_level'.
    This level is typically 2^level downsampling from the highest-res.

    Returns:
      mask_array: 2D NumPy array of shape (h, w)
      actual_downsample: float factor for converting level-0 coords to level=openslide_level coords
    """
    slide = openslide.OpenSlide(mask_path)
    if openslide_level >= slide.level_count:
        raise ValueError(
            f"Requested level {openslide_level} not available. Max = {slide.level_count-1}."
        )

    w, h = slide.level_dimensions[openslide_level]  # width, height at that level
    actual_downsample = slide.level_downsamples[openslide_level]

    # Read the entire region at the chosen level
    region = slide.read_region((0, 0), openslide_level, (w, h))
    mask_array = np.array(region.convert("L"))  # Convert to grayscale

    slide.close()
    return mask_array, float(actual_downsample)


###############################################################################
# 3) FILTER POINTS BY THE DOWNSAMPLED MASK
###############################################################################
def filter_points_in_downsampled_mask(points_px, mask_array, downsample_factor):
    """
    points_px: list of (x_px, y_px) in level-0 coords (highest res).
    mask_array: 2D array at some downsample level.
    downsample_factor: the float factor to convert from level-0 to this array's coords.

    Returns points that land in a non-zero pixel of mask_array.
    """
    kept_points = []
    h, w = mask_array.shape
    for x, y in points_px:
        xd = int(round(x / downsample_factor))
        yd = int(round(y / downsample_factor))
        if 0 <= xd < w and 0 <= yd < h:
            if mask_array[yd, xd] != 0:
                kept_points.append((x, y))
    return kept_points


###############################################################################
# 4) BUILD FINAL DICTS FOR "monocytes", "lymphocytes", "inflammatory_cells"
###############################################################################
def build_final_dicts(monocyte_points_mm, lymphocyte_points_mm):
    """
    monocyte_points_mm, lymphocyte_points_mm are lists of dicts:
      { "name": "...", "point": [x_mm, y_mm, z_spacing], "probability": 1.0 }
    """
    monocytes_json = {
        "name": "monocytes",
        "type": "Multiple points",
        "points": monocyte_points_mm,
        "version": {"major": 1, "minor": 0},
    }
    lymphocytes_json = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "points": lymphocyte_points_mm,
        "version": {"major": 1, "minor": 0},
    }
    inflammatory_json = {
        "name": "inflammatory cells",
        "type": "Multiple points",
        "points": monocyte_points_mm + lymphocyte_points_mm,
        "version": {"major": 1, "minor": 0},
    }
    return {
        "monocytes": monocytes_json,
        "lymphocytes": lymphocytes_json,
        "inflammatory_cells": inflammatory_json,
    }


###############################################################################
# 5) MAIN FUNCTION: parse_inflammatory_data
###############################################################################
def parse_inflammatory_data(
    json_path,
    mode="json",
    real_mpp=0.24199951445730394,
    z_spacing=0.25,
    monocyte_label="Inflammatory",
    lymphocyte_label="lymphocytes",
    mask_tif_path=None,
    openslide_level=1,
):
    """
    1. Stream-parse points from 'json_path' (custom or GeoJSON).
    2. Load ROI mask via OpenSlide at `openslide_level`, then filter points.
    3. Convert the kept points (px) => mm => final dicts.

    Returns dict with keys: "monocytes", "lymphocytes", "inflammatory_cells".
    """
    # 1) Parse points
    monocyte_pts_px, lymphocyte_pts_px = parse_points_stream(
        json_path=json_path,
        mode=mode,
        monocyte_label=monocyte_label,
        lymphocyte_label=lymphocyte_label,
    )

    # 2) If mask is given, load and filter points
    if mask_tif_path:
        mask_array, actual_downsample = load_mask_openslide(
            mask_tif_path, openslide_level
        )
        monocyte_pts_filtered = filter_points_in_downsampled_mask(
            monocyte_pts_px, mask_array, downsample_factor=actual_downsample
        )
        lymphocyte_pts_filtered = filter_points_in_downsampled_mask(
            lymphocyte_pts_px, mask_array, downsample_factor=actual_downsample
        )
    else:
        monocyte_pts_filtered = monocyte_pts_px
        lymphocyte_pts_filtered = lymphocyte_pts_px

    # 3) Convert final pixel coords => mm coords
    px_to_mm = real_mpp / 1000.0  # e.g. 0.25 microns/px => 0.25 / 1000 mm/px
    monocyte_points_mm = []
    for i, (x_px, y_px) in enumerate(monocyte_pts_filtered, start=1):
        x_mm = x_px * px_to_mm
        y_mm = y_px * px_to_mm
        monocyte_points_mm.append(
            {
                "name": f"Monocyte {i}",
                "point": [x_mm, y_mm, z_spacing],
                "probability": 1.0,
            }
        )

    lymphocyte_points_mm = []
    for i, (x_px, y_px) in enumerate(lymphocyte_pts_filtered, start=1):
        x_mm = x_px * px_to_mm
        y_mm = y_px * px_to_mm
        lymphocyte_points_mm.append(
            {
                "name": f"Lymphocyte {i}",
                "point": [x_mm, y_mm, z_spacing],
                "probability": 1.0,
            }
        )

    # 4) Build final results dict
    return build_final_dicts(monocyte_points_mm, lymphocyte_points_mm)


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
        real_mpp=0.24199951445730394,  # your MPP
        z_spacing=0.25,
        monocyte_label="Inflammatory",
        lymphocyte_label="lymphocytes",
        mask_tif_path=mask_tif_path,
        openslide_level=1,  # Lower = higher resolution, higher = more downsample
    )

    # Save final outputs
    with open(os.path.join(save_dir, "detected-monocytes.json"), "w") as f:
        json.dump(results["monocytes"], f, indent=2)

    with open(os.path.join(save_dir, "detected-lymphocytes.json"), "w") as f:
        json.dump(results["lymphocytes"], f, indent=2)

    with open(os.path.join(save_dir, "detected-inflammatory-cells.json"), "w") as f:
        json.dump(results["inflammatory_cells"], f, indent=2)

    print("✅ Done! Wrote out results to:", save_dir)
