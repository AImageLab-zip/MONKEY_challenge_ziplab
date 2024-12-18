import glob
import os

import yaml
from tqdm import tqdm

from .config_parser import get_args_and_config
from .dot2polygon import dot2polygon


def create_bboxes_annots(config):
    """
    Create bounding boxes annotations from the XML annotations.

    param config: Configuration dictionary.

    return: 0 if successful, -1 if failed.

    """

    # try to get the dataset configurations from the configuration file then load them, with a default fallback
    dataset_configs = config.get("dataset", {})
    if dataset_configs is None:
        print("Dataset configurations not found in the configuration file.")
        return -1

    dataset_dir = dataset_configs.get("path", "../data/monkey-data")
    lymphocyte_half_box_size = dataset_configs.get("lymphocyte_half_box_size", 4.5)
    monocyte_half_box_size = dataset_configs.get("monocyte_half_box_size", 5.0)
    min_spacing = dataset_configs.get("min_spacing", 0.25)

    annotation_polygon_dirname = config.get(
        "annotation_polygon_dir", "annotations_polygon"
    )

    annotation_dir = os.path.join(dataset_dir, "annotations", "xml")

    annotation_list = glob.glob(os.path.join(annotation_dir, "*.xml"))

    annotation_polygon_dir = os.path.join(dataset_dir, annotation_polygon_dirname)

    if not (os.path.isdir(annotation_polygon_dir)):
        os.mkdir(annotation_polygon_dir)

    loading_bar = tqdm(annotation_list, desc="Creating bounding boxes annotations")

    for xml_path in loading_bar:
        output_path = os.path.join(
            annotation_polygon_dir,
            os.path.splitext(os.path.basename(xml_path))[0]
            + "_polygon"
            + os.path.splitext(os.path.basename(xml_path))[1],
        )

        dot2polygon(
            xml_path,
            lymphocyte_half_box_size,
            monocyte_half_box_size,
            min_spacing,
            output_path,
        )

        loading_bar.set_postfix_str(output_path)

    print("Bounding boxes annotations created successfully.")
    return 0


def folders_to_yml(wsi_dir: str, wsa_dir: str, output_dir: str, output_name: str):
    """
    Generate a yaml file to be used as WSD dataconfig from a folder of slides and a folder of annotation or mask files.
    Assumes files use the same name for both the slides and masks.
    """

    wsa_list = glob.glob(wsa_dir)

    yaml_dict = {"training": []}
    # yaml_dict = {'training': [], 'validation': []}
    for wsa in wsa_list:
        patient_name = os.path.basename(wsa).split(
            os.path.basename(wsa_dir).split("*")[1]
        )[0]  # monocytes
        #     print(patient_name)
        if os.path.isfile(os.path.join(wsi_dir, patient_name + "_PAS_CPG.tif")):
            wsi = os.path.join(wsi_dir, patient_name + "_PAS_CPG.tif")
            print("match found:    ", patient_name)
            yaml_dict["training"].append(
                {"wsa": {"path": str(wsa)}, "wsi": {"path": str(wsi)}}
            )

            # # validation if needed
            # yaml_dict['validation'].append(
            #         {"wsa": {"path": str(wsa)}, "wsi": {"path": str(wsi)}})

        else:
            print("no match found:    ", patient_name)

    # make a folder for output
    if not (os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, output_name), "w") as file:
        yaml.safe_dump(yaml_dict, file)


if __name__ == "__main__":
    args, config = get_args_and_config()

    print(config)
    create_bboxes_annots(config=config)
