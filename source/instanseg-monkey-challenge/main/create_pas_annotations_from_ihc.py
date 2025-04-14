import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import ttach as tta
from instanseg import InstanSeg
from instanseg.inference_class import (  # fixed import as instanseg.inference_class and not instanseg.instanseg
    _rescale_to_pixel_size,
    _to_tensor_float32,
)
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
OUTPUT_PATH = Path("/work/grana_urologia/MONKEY_challenge/outputs/instanseg")


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

    # plt.figure(figsize=(10,10))
    # plt.imshow(_move_channel_axis(image, to_back=True))
    # plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(_move_channel_axis(image, to_back=True))
    plt.scatter(coords[y_hat == 0, 1], coords[y_hat == 0, 0], marker="x", color="blue")
    plt.scatter(coords[y_hat == 1, 1], coords[y_hat == 1, 0], marker="x", color="green")
    plt.scatter(coords[y_hat == 2, 1], coords[y_hat == 2, 0], marker="x", color="red")

    # Create custom legend
    legend_labels = {0: "Lymphocytes", 1: "Monocytes", 2: "Other"}
    colors = ["blue", "green", "red"]  # Adjust based on your "jet" colormap

    patches = [
        mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in legend_labels
    ]
    plt.legend(handles=patches, loc="upper right")

    plt.show()


def get_patients(gt_dir: str):
    patients_list = []
    patients_set = []

    # Process ground truth files
    gt_path = Path(gt_dir)
    for file in gt_path.glob("*.json"):
        patient_id = "_".join(file.name.split("_")[:2])
        patients_list.append(patient_id)

    patients_set = list(set(patients_list))

    return patients_set


def get_dicts(coords_lymphocytes, y_lymphocytes, conf):
    output_dict = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    counter = 0

    for cc, class_, confidence in zip(coords_lymphocytes, y_lymphocytes, conf):
        x, y = cc

        x = x * ORIGINAL_MPP / 1000
        y = y * ORIGINAL_MPP / 1000

        prediction_record_inflammatory = {
            "name": "Point " + str(counter),
            "point": [
                x,
                y,
                ORIGINAL_MPP,
            ],
            "probability": sum(confidence[:2]),
        }

        prediction_record_monocyte = {
            "name": "Point " + str(counter),
            "point": [
                x,
                y,
                ORIGINAL_MPP,
            ],
            "probability": confidence[1].item(),
        }

        prediction_record_lymphocyte = {
            "name": "Point " + str(counter),
            "point": [
                x,
                y,
                ORIGINAL_MPP,
            ],
            "probability": confidence[0].item(),
        }

        output_dict_inflammatory_cells["points"].append(
            prediction_record_inflammatory
        )  # should be replaced with detected inflammatory_cells

        output_dict["points"].append(prediction_record_lymphocyte)

        output_dict_monocytes["points"].append(
            prediction_record_monocyte
        )  # should be replaced with detected monocytes

        counter += 1

    return output_dict, output_dict_monocytes, output_dict_inflammatory_cells


def write_json_file(*, location, content):
    # Writes a json file
    print(f"Writing to {os.path.abspath(location)}")
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def run():
    print("starting up")
    # device = "cpu"
    CLASSIFICATION_DEVICE = "cuda"
    np.random.seed(0)  # for reproducibility

    # load the models
    instanseg_script = torch.jit.load(INSTANSEG_MODEL_PATH).to(CLASSIFICATION_DEVICE)
    brightfield_nuclei = InstanSeg(instanseg_script, verbosity=0)

    model = PatchClassifier_pl.load_from_checkpoint(
        checkpoint_path=TRAINED_CLF_IHC_PATH, strict=True
    )
    model = model.to(CLASSIFICATION_DEVICE).eval()

    classifier = tta.ClassificationTTAWrapper(
        model, transforms, merge_mode="mean"
    ).eval()

    # TODO: list and cycle the patient ids with the corresponding pas and ihc paths

    gt_dir = os.path.join(INPUT_PATH, "annotations/json_mm")

    patients_ids = get_patients(gt_dir=gt_dir)

    for patient_id in tqdm(patients_ids):
        # extract the split from the annotations dictionary for the given split

        # extract the pas and ihc wsi paths for the given patient id
        img_pascpg_path = Path(INPUT_PATH) / (
            "images/pas-cpg/" + patient_id + "_PAS_CPG.tif"
        )
        ihc_path = Path(INPUT_PATH) / ("images/ihc/" + patient_id + "_IHC_CPG.tif")

        # create dirctory for the patient
        patient_output_dir = os.path.join(OUTPUT_PATH, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        # load the pas and ihc slides
        slidepascpg = TiffSlide(img_pascpg_path)
        slideihc = TiffSlide(ihc_path)

        # define the list of coords, classes and confidences
        all_coords = []
        all_classes = []
        all_confidences = []

        # get random non-empty tiles from the pas and ihc slides
        tiles_he, tiles_ihc, positions_tiles = get_random_non_empty_tiles_with_pos(
            slidepascpg, slideihc, num_images=1000, tile_size=EXTRACTED_TILE_SIZE
        )  # 400

        # iterate over the extracted tiles (pas and ihc) and process them
        for tile_he, tile_ihc, tile_pos in zip(tiles_he, tiles_ihc, positions_tiles):
            # show_images(tile_he,tile_ihc,labels) # for debugging

            # run the instanseg model on the pas tile, specifying the pixel size at 40x (level 0) and rescale the output at 0.25 micrometer per pixel
            labels, input_tensor = brightfield_nuclei.eval_small_image(
                tile_he,
                pixel_size=ORIGINAL_MPP,
                rescale_output=False,
                seed_threshold=0.05,
            )

            # Given that tile_pos is the top-left coordinate (x, y) and the tile size is EXTRACTED_TILE_SIZE,
            # we compute the bounding box as follows:
            tile_bbox = (
                tile_pos,
                (tile_pos[0] + EXTRACTED_TILE_SIZE, tile_pos[1] + EXTRACTED_TILE_SIZE),
            )
            # Now, tile_bbox[0] is the top-left coordinate in (x, y) order.

            # conver the ihc tile to a tensor and rescale it to the destination pixel size
            ihc_tensor = (
                _rescale_to_pixel_size(
                    _to_tensor_float32(tile_ihc), ORIGINAL_MPP, DEST_MPP
                )
                .byte()
                .to(CLASSIFICATION_DEVICE)
            )

            # convert the pas tile to a tensor and rescale it to the destination pixel size
            he_tensor = (
                _rescale_to_pixel_size(
                    _to_tensor_float32(tile_he), ORIGINAL_MPP, DEST_MPP
                )
                .byte()
                .to(CLASSIFICATION_DEVICE)
            )
            # skip the tile if the labels are empty
            if labels.sum() == 0:
                continue

            # check if the ihc and pas tensors have the same shape as the labels
            assert ihc_tensor.shape[-2:] == he_tensor.shape[-2:]
            assert ihc_tensor.shape[-2:] == labels.shape[-2:]

            labels = torch_fastremap(labels)

            # ??? -> from the EXTRACTED_TILE_SIZE pixels squared tile, take the labels of the pas tile, output the patches and masks of the corresponding ihc tile
            # labels are from the instanseg model preds of the pas tile
            crops, masks = get_masked_patches(
                labels.to(CLASSIFICATION_DEVICE), ihc_tensor, patch_size=PATCH_SIZE
            )
            crops = (crops) / 255  # normalise crops values to [0,1]?
            masks = masks  # mask is already in [0,1]
            x_ihc = torch.cat(
                (crops, masks), dim=1
            )  # concatenating the predicted nuclei masks and crops from the pas to the ihc tensor

            # extract the patches (crops) and masks from the pas tile of EXTRACTED_TILE_SIZE pixels squared
            crops, masks = get_masked_patches(
                labels.to(CLASSIFICATION_DEVICE), he_tensor, patch_size=PATCH_SIZE
            )
            crops = (crops).to(torch.uint8)  # convert crops to uint8
            masks = (masks).to(torch.uint8)  # convert masks to uint8

            # concatenate the crops and masks from the pas tile
            # x = (torch.cat((crops, masks), dim=1)).cpu().numpy().astype(np.uint8)

            with torch.no_grad():
                # y_hat_he = torch.cat([classifier_he.forward(x[i:i+batch_size].float().to(CLASSIFICATION_DEVICE)) for i in range(0,len(x_ihc),batch_size)],dim = 0)
                # y_hat_he = y_hat_he.argmax(dim = 1).cpu()

                # predict the nuclei from the random non-empty IHC tile from the ihc slide
                y_hat = torch.cat(
                    [
                        classifier.forward(
                            x_ihc[i : i + BATCH_SIZE].float().to(CLASSIFICATION_DEVICE)
                        )
                        for i in range(0, len(x_ihc), BATCH_SIZE)
                    ],
                    dim=0,
                )
                y_hat = y_hat[:, -3:]  # because of dual training
                y_hat = y_hat.cpu()

            assert y_hat.isnan().sum() == 0

            conf = y_hat.softmax(1)
            y_hat = y_hat.argmax(1)
            y_lymphocytes = y_hat.cpu().numpy()

            # Get centroids (local coordinates) from the predicted label mask.
            coords = centroids_from_lab(labels)[0]
            # Convert local centroids from (row, col) to (col, row) using [:, ::-1],
            # scale them to the destination pixel size, then add the tile's global top-left offset.
            coords_lymphocytes = coords.cpu().numpy()[:, ::-1] * (
                DEST_MPP / ORIGINAL_MPP
            ) + np.array(tile_bbox[0])
            confidence_lymphocytes = conf.cpu().numpy()
            all_coords.extend(coords_lymphocytes)
            all_classes.extend(y_lymphocytes)
            all_confidences.extend(confidence_lymphocytes)

        all_coords = np.array(all_coords)
        all_classes = np.array(all_classes)
        all_confidences = np.array(all_confidences)

        output_dict, output_dict_monocytes, output_dict_inflammatory_cells = get_dicts(
            all_coords, all_classes, all_confidences
        )

        # saving json file
        json_filename_lymphocytes = "detected-lymphocytes.json"
        output_path_json = os.path.join(patient_output_dir, json_filename_lymphocytes)
        write_json_file(location=output_path_json, content=output_dict)

        json_filename_monocytes = "detected-monocytes.json"
        # it should be replaced with correct json files
        output_path_json = os.path.join(patient_output_dir, json_filename_monocytes)
        write_json_file(location=output_path_json, content=output_dict_monocytes)

        json_filename_inflammatory_cells = "detected-inflammatory-cells.json"
        # it should be replaced with correct json files
        output_path_json = os.path.join(
            patient_output_dir, json_filename_inflammatory_cells
        )
        write_json_file(
            location=output_path_json, content=output_dict_inflammatory_cells
        )
        print(f"Saved json files for {patient_id}")

    print("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
