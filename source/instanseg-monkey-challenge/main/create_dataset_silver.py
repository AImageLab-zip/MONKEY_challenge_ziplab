import json
import os
import pdb
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import ttach as tta
from instanseg import InstanSeg
from instanseg.inference_class import (  # fixed import as instanseg.inference_class and not instanseg.instanseg
    _rescale_to_pixel_size,
    _to_tensor_float32,
)
from instanseg.utils.data_download import (
    create_processed_datasets_dir,
    create_raw_datasets_dir,
    download_and_extract,
)
from instanseg.utils.pytorch_utils import get_masked_patches
from instanseg.utils.utils import _move_channel_axis, show_images
from PIL import Image, ImageDraw
from tiffslide import TiffSlide
from tiling import get_random_non_empty_tiles
from tqdm import tqdm
from train import PatchClassifier_pl
from utils import get_classifier

# updated this to my dataset abs path
monkey_dir = Path("/work/grana_urologia/MONKEY_challenge/data/monkey-data")

os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = (
    "/work/grana_urologia/MONKEY_challenge/outputs/instanseg"
)
os.environ["INSTANSEG_DATASET_PATH"] = (
    "/work/grana_urologia/MONKEY_challenge/data/instanseg_dataset"
)
os.environ["INSTANSEG_OUTPUT_PATH"] = (
    "/work/grana_urologia/MONKEY_challenge/outputs/instanseg"
)

INSTANSEG_MODEL_BRIGHTFIELD = "/work/grana_urologia/MONKEY_challenge/checkpoints/instanseg_monkey/instanseg_brightfield_monkey.pt"
TRAINED_CLF_IHC_PATH = "/work/grana_urologia/MONKEY_challenge/checkpoints/instanseg_monkey/1922985/checkpoints/epoch=249-step=195500.ckpt"


LEVEL_0_MPP = 0.2420
TARGET_MPP = 0.5


def create_dictionaries(monkey_dir):
    files = sorted(os.listdir(os.path.join(monkey_dir, "annotations", "xml")))

    label_ids = []
    means_list = []
    annotations_dict = {}

    for file in tqdm(files):
        split = np.random.choice(["train", "val"], p=[0.8, 0.2])

        img_pascpg_path = Path(monkey_dir) / (
            "images/pas-cpg/" + file.split(".")[0] + "_PAS_CPG.tif"
        )
        img_pasdiagnostic_path = Path(monkey_dir) / (
            "images/pas-diagnostic/" + file.split(".")[0] + "_PAS_Diagnostic.tif"
        )
        # img_pasoriginal_path = Path(monkey_dir) / ("images/pas-original/" + file.split(".")[0] + "_PAS_Original.tif")
        ihc_path = Path(monkey_dir) / (
            "images/ihc/" + file.split(".")[0] + "_IHC_CPG.tif"
        )

        slidepascpg = TiffSlide(img_pascpg_path)
        slideihc = TiffSlide(ihc_path)

        tree = ET.parse(monkey_dir / ("annotations/xml/" + file))
        root = tree.getroot()  # Get the root of the XML

        # if split == "val":
        #     destination_img = "/home/cdt/Documents/Projects/monkey-challenge-instanseg/evaluation/validation_set/images/kidney-transplant-biopsy-wsi-pas/"
        #     destination_mask = "/home/cdt/Documents/Projects/monkey-challenge-instanseg/evaluation/validation_set/images/tissue-mask/"

        #     #move images to inference folder
        #
        #     shutil.copy(monkey_dir / ("images/pas-cpg/" + file.split(".")[0] + "_PAS_CPG.tif"), destination_img)
        #     shutil.copy(monkey_dir / ("images/tissue-masks/" + file.split(".")[0] + "_mask.tif"), destination_mask)

        #     shutil.copy(monkey_dir / ("annotations/json/" + file.split(".")[0] + "_inflammatory-cells.json"),
        #     '/home/cdt/Documents/Projects/monkey-challenge-instanseg/evaluation/ground_truth')

        #     shutil.copy(monkey_dir / ("annotations/json/" + file.split(".")[0] + "_lymphocytes.json"),
        #     '/home/cdt/Documents/Projects/monkey-challenge-instanseg/evaluation/ground_truth')

        #     shutil.copy(monkey_dir / ("annotations/json/" + file.split(".")[0] + "_monocytes.json"),
        #     '/home/cdt/Documents/Projects/monkey-challenge-instanseg/evaluation/ground_truth')

        coords = []
        annotations_dict[file] = []

        # Iterate over each annotation and extract relevant information
        for annotation in root.findall(".//Annotation"):
            name = annotation.get("Name")
            part_of_group = annotation.get("PartOfGroup")
            _type = annotation.get("Type")

            if _type == "Polygon":
                coords_ROI = []
                for coordinate in annotation.findall(".//Coordinate"):
                    x = float(coordinate.get("X"))
                    y = float(coordinate.get("Y"))
                    coords_ROI.append([x, y])

                coords_ROI = np.array(coords_ROI)

                x_min, y_min = coords_ROI.min(axis=0)
                x_max, y_max = coords_ROI.max(axis=0)
                bbox_width = int(x_max - x_min)
                bbox_height = int(y_max - y_min)

                # Read the bounding box from the slide
                rgb_data = slidepascpg.read_region(
                    (int(x_min), int(y_min)),
                    0,
                    (bbox_width, bbox_height),
                    as_array=True,
                )

                ihc_data = slideihc.read_region(
                    (int(x_min), int(y_min)),
                    0,
                    (bbox_width, bbox_height),
                    as_array=True,
                )

                mask = Image.new("L", (bbox_width, bbox_height), 0)
                polygon = coords_ROI - [
                    x_min,
                    y_min,
                ]  # Translate polygon to local bbox coordinates
                ImageDraw.Draw(mask).polygon(
                    polygon.flatten().tolist(), outline=1, fill=1
                )
                # Convert the mask to a NumPy array
                binary_mask = np.array(mask)

                annotations_dict[file].append(
                    {
                        "split": split,
                        "pas-cpg": rgb_data,
                        "ihc": ihc_data,
                        "polygon": coords_ROI,
                        "mask": binary_mask,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "dots": [],
                    }
                )

                # show_images(rgb_data)

        for annotation in root.findall(".//Annotation"):
            name = annotation.get("Name")
            part_of_group = annotation.get("PartOfGroup")
            _type = annotation.get("Type")

            if _type == "Dot":
                # Find the coordinates
                coordinates = annotation.find(".//Coordinate")
                x = int(float(coordinates.get("X")))
                y = int(float(coordinates.get("Y")))
                c = 0 if part_of_group == "lymphocytes" else 1

                for i, annotation in enumerate(annotations_dict[file]):
                    if (
                        annotation["bbox"][0] < x < annotation["bbox"][2]
                        and annotation["bbox"][1] < y < annotation["bbox"][3]
                    ):
                        annotations_dict[file][i]["dots"].append(
                            [y - annotation["bbox"][1], x - annotation["bbox"][0], c]
                        )

    # SAVE THE ANNOTATIONS in hdf5 format
    # with open(Path(os.environ["INSTANSEG_OUTPUT_PATH"]) / f"annotations_dict.pth", "wb") as f:
    #     torch.save(annotations_dict, f)
    return annotations_dict


def main():
    np.random.seed(0)  # for reproducibility
    annotations_dict = create_dictionaries(monkey_dir)

    # define patch dimensions and destination pixel size (micrometers per pixel)
    patch_size = 128
    normalise = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define test time augmentations
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
        ]
    )

    # load the models
    instanseg_script = torch.jit.load(INSTANSEG_MODEL_BRIGHTFIELD)
    brightfield_nuclei = InstanSeg(instanseg_script, verbosity=0)

    model = PatchClassifier_pl.load_from_checkpoint(
        checkpoint_path=TRAINED_CLF_IHC_PATH, strict=True
    )

    classifier = model.to("cuda").eval()

    tta_classifier = tta.ClassificationTTAWrapper(
        classifier, transforms, merge_mode="mean"
    ).eval()

    # load the list of xml annotations
    files = os.listdir(os.path.join(monkey_dir, "annotations", "xml"))

    # create the HDF5 file for the silver standard dataset
    with h5py.File(
        Path(os.environ["INSTANSEG_DATASET_PATH"]) / "monkey_cpg_silver.h5", "w"
    ) as f:
        # add attributes to the HDF5 file
        f.attrs["class_names"] = str(
            {"0": "lymphocytes", "1": "monocytes", "2": "other"}
        )  # Convert to string since HDF5 attributes must be simple types

        f.attrs["pixel_size"] = TARGET_MPP

        # create datasets for training and validation splits in the HDF5 file
        for split in ["train", "val"]:
            # create datasets for data and labels
            f.create_dataset(
                f"{split}/data",
                shape=(0, 4, patch_size, patch_size),  # 4 channel patches
                dtype=np.uint8,
                maxshape=(None, 4, patch_size, patch_size),
                chunks=(1, 4, patch_size, patch_size),
                #  compression = "lzf",
            )
            f.create_dataset(
                f"{split}/labels", shape=(0, 1), dtype=np.uint8, maxshape=(None, 1)
            )

        for file in tqdm(files):
            # extract the split from the annotations dictionary for the given split
            split = annotations_dict[file][0]["split"]

            # extract the pas and ihc wsi paths for the given patient id
            img_pascpg_path = Path(monkey_dir) / (
                "images/pas-cpg/" + file.split(".")[0] + "_PAS_CPG.tif"
            )
            ihc_path = Path(monkey_dir) / (
                "images/ihc/" + file.split(".")[0] + "_IHC_CPG.tif"
            )

            # load the pas and ihc slides
            slidepascpg = TiffSlide(img_pascpg_path)
            slideihc = TiffSlide(ihc_path)

            # get random non-empty tiles from the pas and ihc slides
            tiles_he, tiles_ihc = get_random_non_empty_tiles(
                slidepascpg, slideihc, num_images=1000, tile_size=1024
            )  # 400

            # iterate over the extracted tiles (pas and ihc) and process them
            for tile_he, tile_ihc in zip(tiles_he, tiles_ihc):
                # show_images(tile_he,tile_ihc,labels) # for debugging

                # run the instanseg model on the pas tile, specifying the pixel size at 40x (level 0) and rescale the output at 0.25 micrometer per pixel
                labels, input_tensor = brightfield_nuclei.eval_small_image(
                    tile_he,
                    pixel_size=LEVEL_0_MPP,
                    rescale_output=False,
                    seed_threshold=0.05,
                )

                # conver the ihc tile to a tensor and rescale it to the destination pixel size
                ihc_tensor = (
                    _rescale_to_pixel_size(
                        _to_tensor_float32(tile_ihc), LEVEL_0_MPP, TARGET_MPP
                    )
                    .byte()
                    .to(device)
                )

                # convert the pas tile to a tensor and rescale it to the destination pixel size
                he_tensor = (
                    _rescale_to_pixel_size(
                        _to_tensor_float32(tile_he), LEVEL_0_MPP, TARGET_MPP
                    )
                    .byte()
                    .to(device)
                )
                # skip the tile if the labels are empty
                if labels.sum() == 0:
                    continue

                # check if the ihc and pas tensors have the same shape as the labels
                assert ihc_tensor.shape[-2:] == he_tensor.shape[-2:]
                assert ihc_tensor.shape[-2:] == labels.shape[-2:]

                # ??? -> from the 1024 pixels squared tile, take the labels of the pas tile, output the patches and masks of the corresponding ihc tile
                # labels are from the instanseg model preds of the pas tile
                crops, masks = get_masked_patches(
                    labels.to(device), ihc_tensor, patch_size=patch_size
                )
                crops = (crops) / 255  # normalise crops values to [0,1]?
                masks = masks  # mask is already in [0,1]
                x_ihc = torch.cat(
                    (crops, masks), dim=1
                )  # concatenating the predicted nuclei masks and crops from the pas to the ihc tensor

                # extract the patches (crops) and masks from the pas tile of 1024 pixels squared
                crops, masks = get_masked_patches(
                    labels.to(device), he_tensor, patch_size=patch_size
                )
                crops = (crops).to(torch.uint8)  # convert crops to uint8
                masks = (masks).to(torch.uint8)  # convert masks to uint8

                # concatenate the crops and masks from the pas tile
                x = (torch.cat((crops, masks), dim=1)).cpu().numpy().astype(np.uint8)

                with torch.no_grad():
                    batch_size = 128
                    # y_hat_he = torch.cat([classifier_he.forward(x[i:i+batch_size].float().to("cuda")) for i in range(0,len(x_ihc),batch_size)],dim = 0)
                    # y_hat_he = y_hat_he.argmax(dim = 1).cpu()

                    # predict the nuclei from the random non-empty IHC tile from the ihc slide
                    y_hat = torch.cat(
                        [
                            tta_classifier.forward(
                                x_ihc[i : i + batch_size].float().to("cuda")
                            )
                            for i in range(0, len(x_ihc), batch_size)
                        ],
                        dim=0,
                    )
                    y_hat = y_hat.argmax(
                        dim=1
                    ).cpu()  # take the argmax of the predictions and move to cpu

                # for debugging
                # show_images(*x_ihc[y_hat == 1][:8,:3],n_cols = 8)
                # show_images(*x_ihc[y_hat == 0][:8,:3],n_cols = 8)

                y = y_hat.numpy()[:, None]  # convert to numpy and add a new axis

                # ???
                unique, counts = np.unique(y, return_counts=True)
                min_count = counts.min()
                y_subset = np.concatenate(
                    [y[y == i][: min_count + 10] for i in range(3)]
                )
                x_subset = np.concatenate(
                    [x[(y == i).squeeze()][: min_count + 10] for i in range(3)]
                )

                if x_subset.ndim == 5:
                    x_subset = x_subset[0]
                x = x_subset
                y = y_subset[:, None]

                data_ds = f[f"{split}/data"]
                labels_ds = f[f"{split}/labels"]

                data_ds.resize((data_ds.shape[0] + x.shape[0],) + x.shape[1:])
                data_ds[-x.shape[0] :, ...] = x
                labels_ds.resize((labels_ds.shape[0] + y.shape[0],) + y.shape[1:])
                labels_ds[-y.shape[0] :, ...] = y.astype(np.uint8)


if __name__ == "__main__":
    main()
