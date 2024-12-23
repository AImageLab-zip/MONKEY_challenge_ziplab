import glob
import os

import pandas as pd
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


def create_dataset_df(dataset_path: str, annotations_path: str):
    """
    Creates and updates a metadata DataFrame by mapping WSI image files and
    annotation files to corresponding patient IDs.

    Args:
        dataset_path (str): Path to the dataset directory containing images and metadata.
        annotations_path (str): Path to the directory containing annotation polygon files.

    Returns:
        pd.DataFrame: Updated metadata DataFrame with paths to the relevant WSI images
                      and annotation files for each patient.

    Workflow:
        1. Load and preprocess the metadata DataFrame.
        2. Identify annotation files and corresponding WSI image files.
        3. Update the metadata DataFrame with paths to the matched WSI and annotation files.
        4. Log progress using a progress bar.
    """
    # Load metadata file
    metadata_file = os.path.join(dataset_path, "metadata", "context-information.xlsx")
    metadata_df = pd.read_excel(metadata_file, index_col=0)

    # Preprocess the metadata DataFrame
    metadata_df.reset_index(drop=True, inplace=True)  # Remove the index column
    metadata_df.replace("x", "Not available", inplace=True)  # Replace placeholders

    # Define directories for annotation and image files
    wsa_dir = f"{annotations_path}" + r"/*_polygon.xml"  # Annotation files
    wsi_pas_cpg_dir = os.path.join(dataset_path, "images", "pas-cpg")  # PAS_CPG images
    wsi_ihc_dir = os.path.join(dataset_path, "images", "ihc")  # IHC images
    wsi_diagnostic_dir = os.path.join(
        dataset_path, "images", "pas-diagnostic"
    )  # PAS Diagnostic
    wsi_original_dir = os.path.join(
        dataset_path, "images", "pas-original"
    )  # PAS Original

    # List of annotation files
    wsa_list = glob.glob(wsa_dir)

    # Initialize a progress bar for processing annotation files
    progress_bar = tqdm(wsa_list, desc="Processing WSAs")

    # Iterate through each annotation file
    for wsa in progress_bar:
        # Extract patient name from the annotation file name
        patient_name = os.path.basename(wsa).split(
            os.path.basename(wsa_dir).split("*")[1]
        )[0]

        # Check if the PAS_CPG image exists for the patient
        pas_cpg_path = os.path.join(wsi_pas_cpg_dir, patient_name + "_PAS_CPG.tif")
        if os.path.isfile(pas_cpg_path):
            progress_bar.set_description(
                f"Processing {patient_name}"
            )  # Update progress bar

            # Update metadata DataFrame with paths for PAS_CPG image and annotation
            metadata_df.loc[
                metadata_df["Slide ID"] == patient_name, "WSI PAS_CPG Path"
            ] = pas_cpg_path
            metadata_df.loc[
                metadata_df["Slide ID"] == patient_name, "Annotation Path"
            ] = wsa

            # Check and update for IHC image
            ihc_path = os.path.join(wsi_ihc_dir, patient_name + "_IHC_CPG.tif")
            if os.path.isfile(ihc_path):
                metadata_df.loc[
                    metadata_df["Slide ID"] == patient_name, "WSI IHC_CPG Path"
                ] = ihc_path

            # Check and update for PAS-Diagnostic image
            pas_diagnostic_path = os.path.join(
                wsi_diagnostic_dir, patient_name + "_PAS_Diagnostic.tif"
            )
            if os.path.isfile(pas_diagnostic_path):
                metadata_df.loc[
                    metadata_df["Slide ID"] == patient_name, "WSI PAS_Diagnostic Path"
                ] = pas_diagnostic_path

            # Check and update for PAS-Original image
            pas_original_path = os.path.join(
                wsi_original_dir, patient_name + "_PAS_Original.tif"
            )
            if os.path.isfile(pas_original_path):
                metadata_df.loc[
                    metadata_df["Slide ID"] == patient_name, "WSI PAS_Original Path"
                ] = pas_original_path
        else:
            # Log if no match is found for the patient
            print("No match found:    ", patient_name)

    return metadata_df


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
