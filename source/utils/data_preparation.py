import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from wholeslidedata.iterators import PatchConfiguration, create_patch_iterator

from .asap_parser import parse_asap_dot_annotations
from .config_parser import get_args_and_config
from .dot2polygon import dot2polygon
from .logger import get_logger


# object to aid in data preparation of the Monkey Dataset
class DataPreparator:
    def __init__(self, config):
        # config loading
        self.config = config
        # project configurations
        self.project_config = self.config.get("project", {})
        if self.project_config is None:
            print("Project configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling
        # seed
        self.seed = self.project_config.get("seed", 42)
        # custom logger named for the class
        self.logger = get_logger(name="DataPreparator")

        ## DATA PATHS CONFIGS
        # try to get the dataset configurations from the configuration file then load them, with a default fallback
        self.dataset_configs = self.config.get("dataset", {})
        if self.dataset_configs is None:
            print("Dataset configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling

        self.dataset_dir = self.dataset_configs.get("path", "../data/monkey-data")
        self.annotation_dir = os.path.join(self.dataset_dir, "annotations", "xml")
        # self.json_px_dir = os.path.join(self.dataset_dir, "annotations", "json_pixel")

        self.annotation_polygon_dirname = self.config.get(
            "annotation_polygon_dir", "annotations_polygon"
        )
        self.annotation_polygon_dir = os.path.join(
            self.dataset_dir, self.annotation_polygon_dirname
        )
        self.metadata_file = os.path.join(
            self.dataset_dir, "metadata", "context-information.xlsx"
        )
        self.yaml_wsi_wsa_dir = self.config.get("yaml_wsi_wsa_dir", "./configs/splits")

        self.wsi_col = self.dataset_configs.get("wsi_col", "WSI PAS_CPG Path")

        self.wsa_col = self.dataset_configs.get("wsa_col", "Annotation Path")

        ## BOUNDING BOXES AND SPACING CONFIGS
        self.lymphocyte_half_box_size = self.dataset_configs.get(
            "lymphocyte_half_box_size", 4.5
        )
        self.monocyte_half_box_size = self.dataset_configs.get(
            "monocyte_half_box_size", 5.0
        )
        self.min_spacing = self.dataset_configs.get("min_spacing", 0.25)

        ## DATASET BALANCING CONFIGS

        # number of folds for the dataset
        self.n_folds = self.dataset_configs.get("n_folds", 5)

        # column in the dataset dataframe to balance the folds by
        self.balance_by = self.dataset_configs.get("balance_by", None)

        # bins number for total cells count to balance the dataset in the folds (optionally)
        self.num_bins_total_cells_count = self.dataset_configs.get(
            "num_bins_total_cells_count", 5
        )

        # dataset dataframe with wsi metadata
        self.dataset_df = None
        self.fold_yaml_paths_dict = {}

    def create_bboxes_annots(self):
        """
        Create bounding boxes annotations from the XML annotations.

        return: 0 if successful, -1 if failed.

        """

        annotation_list = glob.glob(os.path.join(self.annotation_dir, "*.xml"))

        if not (os.path.isdir(self.annotation_polygon_dir)):
            os.mkdir(self.annotation_polygon_dir)

        loading_bar = tqdm(annotation_list, desc="Creating bounding boxes annotations")

        for xml_path in loading_bar:
            output_path = os.path.join(
                self.annotation_polygon_dir,
                os.path.splitext(os.path.basename(xml_path))[0]
                + "_polygon"
                + os.path.splitext(os.path.basename(xml_path))[1],
            )

            dot2polygon(
                xml_path,
                self.lymphocyte_half_box_size,
                self.monocyte_half_box_size,
                self.min_spacing,
                output_path,
            )

            loading_bar.set_postfix_str(output_path)

        print("Bounding boxes annotations created successfully.")
        return 0

    def create_dataset_df(self, bboxes=True):
        """
        Creates and updates a metadata DataFrame by mapping WSI image files and
        annotation files to corresponding patient IDs.

        Workflow:
            1. Load and preprocess the metadata DataFrame.
            2. Identify annotation files and corresponding WSI image files.
            3. Update the metadata DataFrame with paths to the matched WSI and annotation files.
            4. Log progress using a progress bar.
        """
        # Load metadata file
        self.dataset_df = pd.read_excel(self.metadata_file, index_col=0)

        # Preprocess the metadata DataFrame
        self.dataset_df.reset_index(drop=True, inplace=True)  # Remove the index column
        self.dataset_df.replace(
            "x", "Not available", inplace=True
        )  # Replace placeholders

        # Define directories for annotation and image files
        if bboxes:
            wsa_dir = (
                f"{self.annotation_polygon_dir}" + r"/*_polygon.xml"
            )  # Annotation files for bboxes in pixel coordinates
        else:
            wsa_dir = (
                f"{self.annotation_dir}" + r"/*.xml"
            )  # point annotations in pixel coordinates
        wsi_pas_cpg_dir = os.path.join(
            self.dataset_dir, "images", "pas-cpg"
        )  # PAS_CPG images
        wsi_ihc_dir = os.path.join(self.dataset_dir, "images", "ihc")  # IHC images
        wsi_diagnostic_dir = os.path.join(
            self.dataset_dir, "images", "pas-diagnostic"
        )  # PAS Diagnostic
        wsi_original_dir = os.path.join(
            self.dataset_dir, "images", "pas-original"
        )  # PAS Original
        wsi_tissue_mask_dir = os.path.join(self.dataset_dir, "images", "tissue-masks")

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
            # save the associated ROI mask path
            mask_path = os.path.join(wsi_tissue_mask_dir, patient_name + "_mask.tif")

            if os.path.isfile(pas_cpg_path):
                progress_bar.set_description(
                    f"Processing {patient_name}"
                )  # Update progress bar

                # Update metadata DataFrame with paths for PAS_CPG image, mask and annotation
                self.dataset_df.loc[
                    self.dataset_df["Slide ID"] == patient_name, "WSI PAS_CPG Path"
                ] = pas_cpg_path
                self.dataset_df.loc[
                    self.dataset_df["Slide ID"] == patient_name, "WSI Mask Path"
                ] = mask_path
                self.dataset_df.loc[
                    self.dataset_df["Slide ID"] == patient_name, "Annotation Path"
                ] = wsa

                # Check and update for IHC image
                ihc_path = os.path.join(wsi_ihc_dir, patient_name + "_IHC_CPG.tif")
                if os.path.isfile(ihc_path):
                    self.dataset_df.loc[
                        self.dataset_df["Slide ID"] == patient_name, "WSI IHC_CPG Path"
                    ] = ihc_path

                # Check and update for PAS-Diagnostic image
                pas_diagnostic_path = os.path.join(
                    wsi_diagnostic_dir, patient_name + "_PAS_Diagnostic.tif"
                )
                if os.path.isfile(pas_diagnostic_path):
                    self.dataset_df.loc[
                        self.dataset_df["Slide ID"] == patient_name,
                        "WSI PAS_Diagnostic Path",
                    ] = pas_diagnostic_path

                # Check and update for PAS-Original image
                pas_original_path = os.path.join(
                    wsi_original_dir, patient_name + "_PAS_Original.tif"
                )
                if os.path.isfile(pas_original_path):
                    self.dataset_df.loc[
                        self.dataset_df["Slide ID"] == patient_name,
                        "WSI PAS_Original Path",
                    ] = pas_original_path
            else:
                # Log if no match is found for the patient
                print("No match found:    ", patient_name)

        # add a total cell count column and quantile bins for the immune cells
        self.dataset_df = self._create_quantile_bins_cells(
            self.dataset_df, n_bins=self.num_bins_total_cells_count
        )

        self.logger.debug(f"Dataset firs rows:\n{self.dataset_df.head()}\n")

        # save the updated metadata file
        self.dataset_df.to_csv(
            os.path.join(self.yaml_wsi_wsa_dir, "dataset_metadata_df.csv"), index=False
        )

        return self.dataset_df

    @staticmethod
    def _create_quantile_bins_cells(dataset_df, n_bins=5):
        """
        Create bins based on quantiles of the total cell count.

        Returns:
            pd.DataFrame: The dataframe with a new 'immune_cell_bin' column.
        """
        dataset_df = dataset_df.copy()
        dataset_df["Total_cells"] = (
            dataset_df["Nb_lymphocytes"] + dataset_df["Nb_monocytes"]
        )

        dataset_df["immune_cell_bin"] = pd.qcut(
            dataset_df["Total_cells"], q=n_bins, labels=range(n_bins)
        )
        return dataset_df

    def split_and_save_kfold(self, save=True):
        """
        Split the data into n folds, save to .yml files with training and validation keys,
        and add fold ID to the dataframe.

        Returns:
            pd.DataFrame: The input dataframe with an additional 'fold_id' column.
            dict: A dictionary containing the fold IDs and their corresponding YAML file paths.
        """
        # Make the output yaml splits directory if not present
        os.makedirs(self.yaml_wsi_wsa_dir, exist_ok=True)

        self.fold_yaml_paths_dict = {}

        # Add a fold_id column initialized to -1 (or reset if already present)
        # self.dataset_df = self.dataset_df.copy()
        self.dataset_df["fold_id"] = -1

        # Choose the splitter
        if self.balance_by and self.balance_by in self.dataset_df.columns:
            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )
            stratify_col = self.dataset_df[self.balance_by]
        else:
            # Use random non-stratified splitter if no column is specified
            skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            stratify_col = None

        # Split the data
        for fold, (train_idx, test_idx) in enumerate(
            skf.split(self.dataset_df, stratify_col)
        ):
            train_data = self.dataset_df.iloc[train_idx]
            validation_data = self.dataset_df.iloc[test_idx]

            # Assign fold IDs to the dataframe
            self.dataset_df.loc[test_idx, "fold_id"] = fold

            # Format data for YAML
            train_yaml = [
                {"wsa": {"path": row[self.wsa_col]}, "wsi": {"path": row[self.wsi_col]}}
                for _, row in train_data.iterrows()
            ]
            validation_yaml = [
                {"wsa": {"path": row[self.wsa_col]}, "wsi": {"path": row[self.wsi_col]}}
                for _, row in validation_data.iterrows()
            ]

            fold_type = f"balanced_{self.balance_by}_" if self.balance_by else ""
            # Save YAML files
            fold_file_path = os.path.join(
                self.yaml_wsi_wsa_dir, f"fold_{fold_type}{fold}.yml"
            )

            # Save the fold yaml file
            if save:
                # Save the fold yaml path to the dictionary
                self.fold_yaml_paths_dict[fold] = fold_file_path
                with open(fold_file_path, "w") as f:
                    yaml.dump(
                        {"training": train_yaml, "validation": validation_yaml}, f
                    )

                print(f"Fold {fold + 1} saved -> {fold_file_path}")

            else:
                self.fold_yaml_paths_dict[fold] = {
                    "training": train_yaml,
                    "validation": validation_yaml,
                }
                print(f"Fold {fold + 1} dict created")

        return self.dataset_df, self.fold_yaml_paths_dict

    def prepare_data(self):
        """
        Prepare the data for the WSD task.
        """
        # 1. Create bounding boxes annotations
        self.create_bboxes_annots()

        # 2. Create and update the dataset dataframe
        self.create_dataset_df(
            bboxes=True
        )  # create the df using bounding boxes annotations

        # 3. Split the data into n folds and save to n .yml files in the specified directory
        dataset_df, folds_paths_dict = self.split_and_save_kfold()

        return dataset_df, folds_paths_dict

    def prepare_data_points_annotations(self):
        # 1. Create the dataset dataframe using points annotations
        self.create_dataset_df(bboxes=False)

        # 2. Split the data into n folds and save to n .yml files in the specified directory
        dataset_df, folds_paths_dict = self.split_and_save_kfold(save=False)

        return dataset_df, folds_paths_dict

    def create_cellvit_dataset_singlerow(
        self,
        output_dir: str,
        group_to_label={"monocytes": 0, "lymphocytes": 1},
        ignore_groups={"ROI"},
        patch_shape=(1024, 1024, 3),
        spacings=(0.24199951445730394,),
        overlap=(0, 0),
        offset=(0, 0),
        center=False,
        cpus=4,
    ):
        """
        Creates a CellViT-compatible dataset from the slides in `self.dataset_df`,
        assuming each Slide ID is unique in one row. The patch configuration is
        instantiated once, and we reuse it for all slides.

        Folder layout (under output_dir):
            ├── splits/
            │   ├── fold_0/
            │   │   ├── train.csv
            │   │   └── val.csv
            │   ├── fold_1/
            │   │   ├── train.csv
            │   │   └── val.csv
            │   ...
            ├── train/
            │   ├── images/
            │   │   ├── <slideID>_0.png
            │   │   ├── <slideID>_1.png
            │   └── labels/
            │       ├── <slideID>_0.csv
            │       ├── <slideID>_1.csv
            ├── test/
            │   ├── images/   (empty)
            │   └── labels/   (empty)
            └── label_map.yaml    (maps label_id -> class_name)

        `train.csv` and `val.csv` for each fold contain one column with
        the patch basenames (no .png extension).

        Args:
            output_dir (str): Root directory for the CellViT dataset structure.
            group_to_label (dict): label dictionary.Default: {"monocytes": 0, "lymphocytes": 1}.
            ignore_groups (set): Key only dictionary with labels to ignore in the xml annotatiosn. Default: {"ROI"} to skip ROI annotations.
            cpus (int): Number of CPUs for patch generation. Default: 4.
        """
        # 0) Prepare the dataset_df with points annotations
        self.prepare_data_points_annotations()

        # 1) Check presence of fold_id
        if "fold_id" not in self.dataset_df.columns:
            raise ValueError(
                "self.dataset_df must have a 'fold_id' column from split_and_save_kfold()."
            )

        # 2) Create folder structure
        os.makedirs(output_dir, exist_ok=True)

        splits_dir = os.path.join(output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)

        train_dir = os.path.join(output_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        train_images_dir = os.path.join(train_dir, "images")
        os.makedirs(train_images_dir, exist_ok=True)

        train_labels_dir = os.path.join(train_dir, "labels")
        os.makedirs(train_labels_dir, exist_ok=True)

        test_dir = os.path.join(output_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)
        # (test remains empty)

        # 3) Identify unique folds
        unique_folds = sorted(self.dataset_df["fold_id"].unique().astype(int))
        for f in unique_folds:
            fold_dir = os.path.join(splits_dir, f"fold_{f}")
            os.makedirs(fold_dir, exist_ok=True)

        # 4) Set up a dictionary to keep track of patch basenames for each fold
        #    patch_lists[fold_id]["train"] => [patch_1, patch_2, ...]
        #    patch_lists[fold_id]["val"]   => [...]
        patch_lists = defaultdict(lambda: {"train": [], "val": []})

        # 5) Instantiate PatchConfiguration once
        patch_config = PatchConfiguration(
            patch_shape=patch_shape,
            spacings=spacings,
            overlap=overlap,
            offset=offset,
            center=center,
        )
        self.logger.info(
            f"Created PatchConfiguration with the params: patch shape: {patch_shape}, spacings: {spacings}, overlap: {overlap}, offset: {offset}, center: {center}"
        )

        # 6) Convert dataset_df to list of dict rows for iteration
        rows = self.dataset_df.to_dict("records")

        # We'll show progress with tqdm
        pbar = tqdm(rows, desc="Creating CellViT Dataset")

        for row in pbar:
            slide_id = row.get("Slide ID", None)
            fold_id = int(
                row["fold_id"]
            )  # This slide is 'val' in fold=fold_id, 'train' in others

            wsi_path = row.get(self.wsi_col)
            xml_path = row.get(self.wsa_col)
            mask_path = row.get("WSI Mask Path", None)

            if not wsi_path or not os.path.isfile(wsi_path):
                self.logger.warning(f"[{slide_id}] Missing WSI file: {wsi_path}")
                continue
            if not xml_path or not os.path.isfile(xml_path):
                self.logger.warning(f"[{slide_id}] Missing annotation XML: {xml_path}")
                continue

            # 6.1 Parse Dot Annotations ignoring ROI groups
            annotations = parse_asap_dot_annotations(
                xml_path=xml_path,
                group_to_label=group_to_label,
                ignore_groups=ignore_groups,
            )

            pbar.set_postfix_str(f"Slide={slide_id}, Annotations={len(annotations)}")

            # 6.2 Create the patch iterator using the pre-made patch_config.
            #     We'll pass it 'file_path' and 'mask_path' dynamically.
            #     The rest is in patch_config.
            patch_iterator = create_patch_iterator(
                image_path=wsi_path,
                mask_path=mask_path
                if mask_path and os.path.isfile(mask_path)
                else None,
                patch_configuration=patch_config,
                cpus=cpus,
                backend="asap",
            )

            # We'll store all generated patch basenames for this slide
            patch_basenames_for_slide = []

            # 6.3 Go through each patch
            for idx_patch, (patch_data, mask_data, info) in enumerate(patch_iterator):
                patch_np = patch_data.squeeze().astype(np.uint8)  # (H, W, 3)
                H, W, _ = info["tile_shape"]
                patch_x = info["x"]
                patch_y = info["y"]

                # patch_spacing = info["spacing"][0]

                # scale_factor = spacings[0] / patch_spacing

                # self.logger.debug(
                #     f"Patch {idx_patch} - Shape: {patch_np.shape}, Spacing: {patch_spacing}, Scale Factor: {scale_factor}"
                # )

                # Build patch basename: <slideID>_<patchIndex>
                patch_basename = f"{slide_id}_{idx_patch}"

                # Save the patch image
                img_path = os.path.join(train_images_dir, patch_basename + ".png")
                plt.imsave(img_path, patch_np)

                # Convert global coords to local patch coords
                patch_anns = []
                for x_g, y_g, label_id in annotations:
                    x_local = x_g - patch_x  # * scale_factor
                    y_local = y_g - patch_y  # * scale_factor

                    # Check if it’s inside this patch
                    if 0 <= x_local < W and 0 <= y_local < H:
                        patch_anns.append((x_local, y_local, label_id))

                # Save CSV for annotation
                csv_path = os.path.join(train_labels_dir, patch_basename + ".csv")
                with open(csv_path, mode="w", newline="") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["x", "y", "label_id"])
                    for xl, yl, lid in patch_anns:
                        writer.writerow([xl, yl, lid])

                patch_basenames_for_slide.append(patch_basename)

            # 6.4 For each fold: if fold==fold_id => val, else => train
            for f in unique_folds:
                if f == fold_id:
                    patch_lists[f]["val"].extend(patch_basenames_for_slide)
                else:
                    patch_lists[f]["train"].extend(patch_basenames_for_slide)

        # 7) Write out train.csv & val.csv for each fold
        for f in unique_folds:
            fold_dir = os.path.join(splits_dir, f"fold_{f}")
            train_csv_path = os.path.join(fold_dir, "train.csv")
            val_csv_path = os.path.join(fold_dir, "val.csv")

            with open(train_csv_path, mode="w", newline="") as tf:
                writer = csv.writer(tf)
                for patch_base in patch_lists[f]["train"]:
                    writer.writerow([patch_base])

            with open(val_csv_path, mode="w", newline="") as vf:
                writer = csv.writer(vf)
                for patch_base in patch_lists[f]["val"]:
                    writer.writerow([patch_base])

        # 8) Save label_map.yaml in the root output directory
        if group_to_label:
            # Invert the dictionary so that label_id -> class_name
            label_map_inverted = {v: k for k, v in group_to_label.items()}
        else:
            label_map_inverted = {}

        label_map_path = os.path.join(output_dir, "label_map.yaml")
        with open(label_map_path, "w") as f:
            # We'll write them in ascending order of label_id
            for label_id in sorted(label_map_inverted.keys()):
                class_name = label_map_inverted[label_id]
                f.write(f'{label_id}: "{class_name}"\n')

        self.logger.info("CellViT-compatible dataset creation complete!")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(
            f"label_map.yaml saved with {len(label_map_inverted)} entries."
        )


if __name__ == "__main__":
    args, config = get_args_and_config()

    print(config)
    DataPrep = DataPreparator(config=config)
    dataset_df, folds_paths_dict = DataPrep.prepare_data()
    # create_bboxes_annots(config=config)
