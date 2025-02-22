import json
import os
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon
from tqdm import tqdm


# Function to parse dot annotations (monocytes, lymphocytes)
def parse_asap_dot_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations_root = root.find("Annotations")
    dots = []

    for annotation in annotations_root.findall("Annotation"):
        if annotation.get("Type", "").lower() != "dot":
            continue  # Ignore non-dot annotations

        group_name = annotation.get("PartOfGroup", "unknown").lower()
        label_id = 2  # Default "other"
        if group_name == "monocytes":
            label_id = 0
        elif group_name == "lymphocytes":
            label_id = 1

        coords_root = annotation.find("Coordinates")
        if coords_root is None:
            continue

        for coordinate in coords_root.findall("Coordinate"):
            x = float(coordinate.get("X"))
            y = float(coordinate.get("Y"))
            dots.append((x, y, label_id))

    return dots


# Function to parse ROI regions
def parse_roi_regions(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations_root = root.find("Annotations")
    rois = []

    for annotation in annotations_root.findall("Annotation"):
        if annotation.get("PartOfGroup") != "ROI":
            continue  # Only consider ROI polygons

        coords_root = annotation.find("Coordinates")
        if coords_root is None:
            continue

        polygon_points = [
            (float(coord.get("X")), float(coord.get("Y")))
            for coord in coords_root.findall("Coordinate")
        ]
        rois.append(Polygon(polygon_points))  # Convert to Shapely Polygon

    return rois


# Function to parse predicted cell centroids
def parse_predicted_cells(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(
        [(cell["centroid"][0], cell["centroid"][1]) for cell in data["cells"]]
    )


# Function to filter predicted cells inside ROI
def filter_cells_inside_roi(predicted_cells, rois):
    return np.array(
        [
            p
            for p in predicted_cells
            if any(roi.contains(Point(p[0], p[1])) for roi in rois)
        ]
    )


# Function to assign labels and remove duplicate monocyte/lymphocyte points
def assign_labels_deduplicated(
    xml_annotations, predicted_cells, threshold_microns=7.5, base_mpp=0.25
):
    threshold_pixels = threshold_microns / base_mpp
    xml_coords = np.array([(x, y) for x, y, _ in xml_annotations])
    xml_labels = np.array([label for _, _, label in xml_annotations])

    tree = KDTree(xml_coords) if len(xml_coords) > 0 else None
    assigned_annotations = []

    assigned_annotations.extend(xml_annotations)  # Keep all original annotations

    for x, y in predicted_cells:
        if tree is not None:
            dist, idx = tree.query((x, y))
            if dist < threshold_pixels:
                continue  # Skip this predicted cell to prevent duplication

        assigned_annotations.append((x, y, 2))  # Assign "other" label

    return assigned_annotations


# Function to ensure all groups are defined in the XML
def ensure_annotation_group_exists(root, group_name, color):
    annotation_groups = root.find("AnnotationGroups")
    if annotation_groups is None:
        annotation_groups = ET.SubElement(root, "AnnotationGroups")

    for group in annotation_groups.findall("Group"):
        if group.get("Name") == group_name:
            return  # The group already exists

    ET.SubElement(
        annotation_groups,
        "Group",
        {"Name": group_name, "PartOfGroup": "None", "Color": color},
    )


# Function to process the XML with JSON predictions
def preprocess_xml_with_json(
    xml_path, json_path, output_path, threshold_microns=7.5, base_mpp=0.25
):
    xml_annotations = parse_asap_dot_annotations(xml_path)
    predicted_cells = parse_predicted_cells(json_path)
    rois = parse_roi_regions(xml_path)

    filtered_predicted_cells = filter_cells_inside_roi(
        predicted_cells, rois
    )  # Keep only points inside ROI
    new_annotations = assign_labels_deduplicated(
        xml_annotations, filtered_predicted_cells, threshold_microns, base_mpp
    )  # Assign labels and deduplicate

    save_final_asap_xml(xml_path, new_annotations, output_path)


# Function to save the corrected XML
def save_final_asap_xml(xml_path, new_annotations, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations_root = root.find("Annotations")

    # Ensure "other" group exists
    ensure_annotation_group_exists(root, "other", "#00FF00")

    existing_annotations = annotations_root.findall("Annotation")
    max_existing_number = max(
        [
            int(anno.get("Name").split("Annotation ")[1])
            for anno in existing_annotations
            if anno.get("Name", "").startswith("Annotation ")
        ]
        or [0]
    )

    # Remove only duplicate dot annotations, keep ROI regions
    for annotation in existing_annotations:
        if annotation.get("Type") == "Dot":
            annotations_root.remove(annotation)

    color_map = {
        "monocytes": "#FFFF00",
        "lymphocytes": "#FF0000",
        "other": "#00FF00",
    }

    annotation_index = max_existing_number + 1
    for x, y, label in new_annotations:
        group_name = (
            "monocytes" if label == 0 else "lymphocytes" if label == 1 else "other"
        )
        color = color_map[group_name]

        annotation_el = ET.SubElement(
            annotations_root,
            "Annotation",
            {
                "Name": f"Annotation {annotation_index}",
                "Type": "Dot",
                "PartOfGroup": group_name,
                "Color": color,
            },
        )

        coords_el = ET.SubElement(annotation_el, "Coordinates")
        ET.SubElement(
            coords_el,
            "Coordinate",
            {
                "Order": "0",
                "X": f"{x:.4f}",
                "Y": f"{y:.4f}",
            },
        )

        annotation_index += 1

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


import os
import re

from tqdm import tqdm


def extract_patient_ids(xml_folder):
    """Extract unique patient IDs from filenames using '_' as a separator, removing .xml."""
    patient_ids = set()

    for filename in os.listdir(xml_folder):
        parts = filename.split("_")
        if len(parts) > 1 and filename.endswith(".xml"):
            patient_id = "_".join(parts[:2]).replace(
                ".xml", ""
            )  # Keep only the first underscore, remove .xml
            patient_ids.add(patient_id)

    return patient_ids


def find_matching_files(patient_ids, xml_folder, json_folder):
    """Find matching XML and JSON files for each patient ID."""
    xml_files = {pid: None for pid in patient_ids}
    json_files = {pid: None for pid in patient_ids}

    for filename in os.listdir(xml_folder):
        for pid in patient_ids:
            if pid in filename and filename.endswith(".xml"):
                xml_files[pid] = os.path.join(xml_folder, filename)
                break

    for filename in os.listdir(json_folder):
        for pid in patient_ids:
            if (
                filename.startswith(pid)
                and "cell_detection" in filename
                and filename.endswith(".json")
            ):
                json_files[pid] = os.path.join(json_folder, filename)
                break

    print(f"Found matching {len(xml_files)} XML files and {len(json_files)} JSON files")

    return xml_files, json_files


def process_files(
    xml_files,
    json_files,
    output_folder,
    threshold_microns=5,
    base_mpp=0.24199951445730394,
):
    """Process each XML and JSON file pair using preprocess_xml_with_json."""
    os.makedirs(output_folder, exist_ok=True)

    for pid in tqdm(xml_files.keys(), desc="Processing files"):
        xml_input = xml_files.get(pid)
        json_input = json_files.get(pid)
        output_xml = os.path.join(output_folder, f"{pid}_3class.xml")

        if os.path.exists(output_xml):
            print(f"Skipping {pid}: Output file already exists")
            continue

        if xml_input and json_input:
            preprocess_xml_with_json(
                xml_input, json_input, output_xml, threshold_microns, base_mpp
            )
        else:
            print(f"Skipping {pid}: Missing file(s)")


if __name__ == "__main__":
    # # Set file paths for MONKEY dataset
    # xml_input = (
    #     "/work/grana_urologia/MONKEY_challenge/data/cell_positions_preds/A_P000001.xml"
    # )
    # json_input = "/work/grana_urologia/MONKEY_challenge/data/cell_positions_preds/A_P000001_PAS_CPG_cell_detection.json"
    # output_xml = "/work/grana_urologia/MONKEY_challenge/data/cell_positions_preds/A_P000001_3class.xml"

    # # Run the processing
    # preprocess_xml_with_json(
    #     xml_input,
    #     json_input,
    #     output_xml,
    #     threshold_microns=5,
    #     base_mpp=0.24199951445730394,
    # )

    # Example usage
    xml_folder = (
        "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/xml/"
    )
    json_folder = "/work/grana_urologia/MONKEY_challenge/data/cell_positions_preds/"
    output_folder = "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/xml_3_classes/"

    patient_ids = extract_patient_ids(xml_folder)
    xml_files, json_files = find_matching_files(patient_ids, xml_folder, json_folder)
    process_files(
        xml_files,
        json_files,
        output_folder,
        threshold_microns=5,
        base_mpp=0.24199951445730394,
    )
