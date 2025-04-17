import json
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import ttach as tta
from instanseg import InstanSeg
from instanseg.inference_class import _rescale_to_pixel_size, _to_tensor_float32
from instanseg.utils.pytorch_utils import (
    centroids_from_lab,
    get_masked_patches,
    torch_fastremap,
)
from instanseg.utils.utils import _move_channel_axis, show_images
from skimage.measure import label as sklabel
from tiffslide import TiffSlide
from tiling import get_random_non_empty_tiles, get_random_non_empty_tiles_with_pos
from tqdm import tqdm
from train import PatchClassifier_pl

# WSI VARIABLES
EXTRACTED_TILE_SIZE = 1024
ORIGINAL_MPP = 0.24199951445730394
DEST_MPP = 0.5
PATCH_SIZE = 128
NORMALIZE_HE = False
rescale_output = False if DEST_MPP == 0.5 else True

# MODEL VARIABLES
USE_TTA = True
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
    ]
)
BATCH_SIZE = 128
INSTANSEG_MODEL_PATH = Path(
    "/work/grana_urologia/MONKEY_challenge/checkpoints/instanseg_monkey/instanseg_brightfield_monkey.pt"
)
TRAINED_CLF_IHC_PATH = Path(
    "/work/grana_urologia/MONKEY_challenge/checkpoints/instanseg_monkey/1922985/checkpoints/epoch=249-step=195500.ckpt"
)

# DATASET AND OUTPUT FOLDER VARIABLES
INPUT_PATH = Path("/work/grana_urologia/MONKEY_challenge/data/monkey-data")
OUTPUT_PATH = Path(
    "/work/grana_urologia/MONKEY_challenge/data/instanseg_3_classes_xml_annotations_all_wsi"
)


def normalise_HE(x):
    import torch
    import torchstain
    from instanseg.utils.utils import _move_channel_axis

    device = x.device
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.maxCRef = normalizer.maxCRef.to(device)
    normalizer.HERef = normalizer.HERef.to(device)
    norm = normalizer.normalize(I=x, stains=False, Io=240, beta=0.01)
    norm = torch.clamp(norm[0], 0, 255)
    norm = _move_channel_axis(norm)
    return norm


def scatter_plot(coords, y_hat, image):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(_move_channel_axis(image, to_back=True))
    plt.scatter(coords[y_hat == 0, 1], coords[y_hat == 0, 0], marker="x", color="blue")
    plt.scatter(coords[y_hat == 1, 1], coords[y_hat == 1, 0], marker="x", color="green")
    plt.scatter(coords[y_hat == 2, 1], coords[y_hat == 2, 0], marker="x", color="red")

    legend_labels = {0: "Lymphocytes", 1: "Monocytes", 2: "Other"}
    colors = ["blue", "green", "red"]
    patches = [
        mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in legend_labels
    ]
    plt.legend(handles=patches, loc="upper right")
    plt.show()


def get_patients(gt_dir: str):
    patients_list = []
    gt_path = Path(gt_dir)
    for file in gt_path.glob("*.json"):
        patient_id = "_".join(file.name.split("_")[:2])
        patients_list.append(patient_id)
    return list(set(patients_list))


# ----------------------------------------------------------------------
# XML conversion helper (ASAP format)
# ----------------------------------------------------------------------
SPACING_LEVEL0 = ORIGINAL_MPP


def indent(elem, level=0, space="\t"):
    i = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + space
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent(child, level + 1, space)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


COLOR_MAPPING = {"lymphocytes": "#00F900", "monocytes": "#F90000", "other": "#0000FF"}


# def points_to_xml(points, patient_id, label_mapping, output_dir=".", prob_cutoff=0.0):
#     """
#     Converts a list of point annotations into an XML (ASAP) file.

#     Each point dictionary must have:
#       - "name": name for the point.
#       - "point": list or tuple [x, y] (coordinates in working units).
#       - "probability": a numerical probability.
#       - "label_num": an integer label.
#     """
#     root = ET.Element("ASAP_Annotations")
#     annotations_elem = ET.SubElement(root, "Annotations")

#     for point in points:
#         if point.get("probability", 0) < prob_cutoff:
#             continue
#         label_num = point.get("label_num")
#         if label_num is None:
#             continue
#         label_name = label_mapping.get(label_num, "unknown")
#         annotation_attrib = {
#             "Name": point.get("name", "unnamed"),
#             "Type": "Dot",
#             "PartOfGroup": f"{label_name}",
#             "Color": COLOR_MAPPING.get(label_name, "#000000"),
#         }
#         annotation_elem = ET.SubElement(
#             annotations_elem, "Annotation", annotation_attrib
#         )
#         coords_elem = ET.SubElement(annotation_elem, "Coordinates")
#         # Write coordinates:
#         x = point["point"][0]
#         y = point["point"][1]
#         ET.SubElement(
#             coords_elem, "Coordinate", {"Order": "0", "X": str(x), "Y": str(y)}
#         )

#     groups_elem = ET.SubElement(root, "AnnotationGroups")
#     for label_num, label_name in label_mapping.items():
#         ET.SubElement(
#             groups_elem,
#             "Group",
#             {
#                 "Name": f"detected-{label_name}",
#                 "PartOfGroup": "None",
#                 "Color": COLOR_MAPPING.get(label_name, "#000000"),
#             },
#         )

#     indent(root, space="\t")
#     output_path = Path(output_dir) / f"{patient_id}.xml"
#     tree = ET.ElementTree(root)
#     tree.write(output_path, encoding="utf-8", xml_declaration=True)
#     print(f"Saved XML file: {output_path}")


def points_to_xml(
    points: List[Dict[str, Any]],
    patient_id: str,
    label_mapping: Dict[int, str],
    output_dir: str = ".",
    prob_cutoff: float = 0.0,
) -> None:
    """
    Converts a list of point annotations into an ASAP XML file
    using *pixel* coordinates (no scaling). Ensures the output
    directory exists and writes via a string path.
    """

    # make sure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    root = ET.Element("ASAP_Annotations")
    ann_root = ET.SubElement(root, "Annotations")

    present_groups: Dict[str, str] = {}

    for i, pt in enumerate(points):
        if pt.get("probability", 0.0) < prob_cutoff:
            continue

        label_num = pt.get("label_num")
        label_name = label_mapping.get(label_num, "unknown")
        color = COLOR_MAPPING.get(label_name, "#000000")
        present_groups[label_name] = color

        ann_attrs = OrderedDict(
            [
                ("Name", pt.get("name", f"Point {i}")),
                ("Type", "Dot"),
                ("PartOfGroup", label_name),
                ("Color", color),
            ]
        )
        ann_elem = ET.SubElement(ann_root, "Annotation", ann_attrs)

        coords = ET.SubElement(ann_elem, "Coordinates")
        x, y = float(pt["point"][0]), float(pt["point"][1])
        coord_attrs = OrderedDict(
            [
                ("Order", "0"),
                ("X", f"{x:.4f}"),
                ("Y", f"{y:.4f}"),
            ]
        )
        ET.SubElement(coords, "Coordinate", coord_attrs)

    groups_root = ET.SubElement(root, "AnnotationGroups")
    for gname, gcolor in sorted(present_groups.items()):
        grp_attrs = OrderedDict(
            [
                ("Name", gname),
                ("PartOfGroup", "None"),
                ("Color", gcolor),
            ]
        )
        grp_elem = ET.SubElement(groups_root, "Group", grp_attrs)
        if gname.lower() != "other":
            ET.SubElement(grp_elem, "Attributes")

    # pretty-print with tabs
    indent(root, space="\t")

    out_path = Path(output_dir) / f"{patient_id}.xml"
    ET.ElementTree(root).write(str(out_path), encoding="utf-8", xml_declaration=True)


# ----------------------------------------------------------------------
# Main processing function: process each patient, extract points, and generate XML.
# ----------------------------------------------------------------------
def run():
    print("starting up")
    CLASSIFICATION_DEVICE = "cuda"
    np.random.seed(0)

    instanseg_script = torch.jit.load(INSTANSEG_MODEL_PATH).to(CLASSIFICATION_DEVICE)
    brightfield_nuclei = InstanSeg(instanseg_script, verbosity=0)

    model = PatchClassifier_pl.load_from_checkpoint(
        checkpoint_path=TRAINED_CLF_IHC_PATH, strict=True
    )
    model = model.to(CLASSIFICATION_DEVICE).eval()

    classifier = tta.ClassificationTTAWrapper(
        model, transforms, merge_mode="mean"
    ).eval()

    gt_dir = os.path.join(INPUT_PATH, "annotations/json_mm")
    patients_ids = get_patients(gt_dir=gt_dir)
    label_mapping = {
        0: "lymphocytes",
        1: "monocytes",
        2: "other",
    }  # define your mapping here

    # Will accumulate per patient all point dictionaries.
    for patient_id in tqdm(patients_ids):
        img_pascpg_path = Path(INPUT_PATH) / (
            "images/pas-cpg/" + patient_id + "_PAS_CPG.tif"
        )
        ihc_path = Path(INPUT_PATH) / ("images/ihc/" + patient_id + "_IHC_CPG.tif")
        # patient_output_dir = os.path.join(OUTPUT_PATH, patient_id)
        # os.makedirs(patient_output_dir, exist_ok=True)
        patient_output_dir = OUTPUT_PATH

        slidepascpg = TiffSlide(img_pascpg_path)
        slideihc = TiffSlide(ihc_path)

        all_coords = []
        all_classes = []
        all_confidences = []

        tiles_he, tiles_ihc, positions_tiles = get_random_non_empty_tiles_with_pos(
            slidepascpg, slideihc, num_images=1000, tile_size=EXTRACTED_TILE_SIZE
        )

        for tile_he, tile_ihc, tile_pos in zip(tiles_he, tiles_ihc, positions_tiles):
            labels, input_tensor = brightfield_nuclei.eval_small_image(
                tile_he,
                pixel_size=ORIGINAL_MPP,
                rescale_output=False,
                seed_threshold=0.05,
            )

            tile_bbox = (
                tile_pos,
                (tile_pos[0] + EXTRACTED_TILE_SIZE, tile_pos[1] + EXTRACTED_TILE_SIZE),
            )
            ihc_tensor = (
                _rescale_to_pixel_size(
                    _to_tensor_float32(tile_ihc), ORIGINAL_MPP, DEST_MPP
                )
                .byte()
                .to(CLASSIFICATION_DEVICE)
            )
            he_tensor = (
                _rescale_to_pixel_size(
                    _to_tensor_float32(tile_he), ORIGINAL_MPP, DEST_MPP
                )
                .byte()
                .to(CLASSIFICATION_DEVICE)
            )
            if labels.sum() == 0:
                continue

            assert ihc_tensor.shape[-2:] == he_tensor.shape[-2:]
            assert ihc_tensor.shape[-2:] == labels.shape[-2:]
            labels = torch_fastremap(labels)

            crops, masks = get_masked_patches(
                labels.to(CLASSIFICATION_DEVICE), ihc_tensor, patch_size=PATCH_SIZE
            )
            crops = crops / 255.0
            x_ihc = torch.cat((crops, masks), dim=1)

            crops, masks = get_masked_patches(
                labels.to(CLASSIFICATION_DEVICE), he_tensor, patch_size=PATCH_SIZE
            )
            crops = crops.to(torch.uint8)
            masks = masks.to(torch.uint8)

            with torch.no_grad():
                y_hat = torch.cat(
                    [
                        classifier.forward(
                            x_ihc[i : i + BATCH_SIZE].float().to(CLASSIFICATION_DEVICE)
                        )
                        for i in range(0, len(x_ihc), BATCH_SIZE)
                    ],
                    dim=0,
                )
                y_hat = y_hat[:, -3:]
                y_hat = y_hat.cpu()

            assert y_hat.isnan().sum() == 0

            conf = y_hat.softmax(1)
            y_hat = y_hat.argmax(1)
            y_lymphocytes = y_hat.cpu().numpy()

            coords_local = centroids_from_lab(labels)[0]
            coords_global = coords_local.cpu().numpy()[:, ::-1] * (
                DEST_MPP / ORIGINAL_MPP
            ) + np.array(tile_bbox[0])
            conf_np = conf.cpu().numpy()

            all_coords.extend(coords_global)
            all_classes.extend(y_lymphocytes)
            all_confidences.extend(conf_np)

        all_coords = np.array(all_coords)  # shape (N, ?)
        all_classes = np.array(all_classes)  # shape (N,)
        all_confidences = np.array(all_confidences)  # shape (N, 3)

        # Now, combine these arrays into a list of point dictionaries.
        points_list = []
        for i in range(len(all_coords)):
            label_num = int(all_classes[i])
            # Choose probability based on label: for simplicity, use the corresponding confidence.
            if label_num == 0:
                p = float(all_confidences[i][0])
            elif label_num == 1:
                p = float(all_confidences[i][1])
            elif label_num == 2:
                p = float(all_confidences[i][2])
            else:
                p = 0.0

            point_dict = {
                "name": f"Point {i}",
                "point": all_coords[i][:2].tolist(),  # use x, y only
                "probability": p,
                "label_num": label_num,
            }
            points_list.append(point_dict)

        # Use the helper method to write the XML file for this patient.
        points_to_xml(
            points_list,
            patient_id,
            label_mapping,
            output_dir=patient_output_dir,
            prob_cutoff=0.0,
        )
        print(f"Saved XML file for {patient_id}")

    print("Done")
    return 0


if __name__ == "__main__":
    import xml.etree.ElementTree as ET  # needed for XML functions

    raise SystemExit(run())
