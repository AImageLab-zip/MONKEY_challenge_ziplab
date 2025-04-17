import csv
import glob

# At module level, add the following helper function
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from asap_parser import parse_asap_dot_annotations
from logger import get_logger
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from wholeslidedata.iterators import PatchConfiguration, create_patch_iterator

# from .dot2polygon import dot2polygon


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
        self.annotation_dir = self.dataset_configs.get(
            "annotation_dir", "annotations/xml"
        )
        self.annotation_dir = os.path.join(self.dataset_dir, self.annotation_dir)
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

        self.ihc_col = self.dataset_configs.get("ihc_col", "WSI IHC_CPG Path")

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

    # def create_bboxes_annots(self):
    #     """
    #     Create bounding boxes annotations from the XML annotations.

    #     return: 0 if successful, -1 if failed.

    #     """

    #     annotation_list = glob.glob(os.path.join(self.annotation_dir, "*.xml"))

    #     if not (os.path.isdir(self.annotation_polygon_dir)):
    #         os.mkdir(self.annotation_polygon_dir)

    #     loading_bar = tqdm(annotation_list, desc="Creating bounding boxes annotations")

    #     for xml_path in loading_bar:
    #         output_path = os.path.join(
    #             self.annotation_polygon_dir,
    #             os.path.splitext(os.path.basename(xml_path))[0]
    #             + "_polygon"
    #             + os.path.splitext(os.path.basename(xml_path))[1],
    #         )

    #         dot2polygon(
    #             xml_path,
    #             self.lymphocyte_half_box_size,
    #             self.monocyte_half_box_size,
    #             self.min_spacing,
    #             output_path,
    #         )

    #         loading_bar.set_postfix_str(output_path)

    #     print("Bounding boxes annotations created successfully.")
    #     return 0

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

        self.logger.info(f"Found {len(wsa_list)} annotation files.")

        # Initialize a progress bar for processing annotation files
        progress_bar = tqdm(wsa_list, desc="Processing WSAs")

        # patient_names = [
        #     os.path.basename(wsa)
        #     .split(os.path.basename(wsa_dir).split("*")[1])[0]
        #     .split("_3class")[0]
        #     for wsa in wsa_list
        # ]

        # # remove the unmatched rows with the patient names
        # self.dataset_df = self.dataset_df[
        #     self.dataset_df["Slide ID"].isin(patient_names)
        # ]

        # Iterate through each annotation file
        for wsa in progress_bar:
            # Extract patient name from the annotation file name
            patient_name = os.path.basename(wsa).split(
                os.path.basename(wsa_dir).split("*")[1]
            )[0]

            if "_3class" in patient_name:
                # remove the _3class suffix
                patient_name = patient_name.split("_3class")[0]

            self.logger.debug(patient_name)

            # Check if the PAS_CPG image exists for the patient
            pas_cpg_path = os.path.join(wsi_pas_cpg_dir, patient_name + "_PAS_CPG.tif")
            # save the associated ROI mask path
            mask_path = os.path.join(wsi_tissue_mask_dir, patient_name + "_mask.tif")

            self.logger.debug(pas_cpg_path)
            self.logger.debug(mask_path)
            self.logger.debug(wsa)

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

        self.logger.info(
            f"Making {self.n_folds} splits for a daset of n° {len(self.dataset_df)} istances"
        )
        self.dataset_df = self.dataset_df.reset_index(
            drop=True
        )  # Reset before splitting

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

    # def prepare_data(self):
    #     """
    #     Prepare the data for the WSD task.
    #     """
    #     # 1. Create bounding boxes annotations
    #     self.create_bboxes_annots()

    #     # 2. Create and update the dataset dataframe
    #     self.create_dataset_df(
    #         bboxes=True
    #     )  # create the df using bounding boxes annotations

    #     # 3. Split the data into n folds and save to n .yml files in the specified directory
    #     dataset_df, folds_paths_dict = self.split_and_save_kfold()

    #     return dataset_df, folds_paths_dict

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
        shift_x=1,
        shift_y=1,
        use_ihc=False,
        use_mask=False,
    ):
        """
        Serial version: calls process_slide_cellvit on each slide in turn.
        """
        # 1) Prepare folds & dataframe
        self.prepare_data_points_annotations()
        if "fold_id" not in self.dataset_df.columns:
            raise ValueError(
                "No 'fold_id' in dataset_df. Did you run split_and_save_kfold()?"
            )

        # 2) Build folder structure
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

        # If using IHC for patches
        if use_ihc:
            self.wsi_col = self.ihc_col

        # 3) Prepare per-fold CSV writers
        unique_folds = sorted(self.dataset_df["fold_id"].unique().astype(int))
        fold_csv_files = {}
        for f in unique_folds:
            fold_dir = os.path.join(splits_dir, f"fold_{f}")
            os.makedirs(fold_dir, exist_ok=True)
            train_csv = open(os.path.join(fold_dir, "train.csv"), "w", newline="")
            val_csv = open(os.path.join(fold_dir, "val.csv"), "w", newline="")
            fold_csv_files[f] = {
                "train_writer": csv.writer(train_csv),
                "val_writer": csv.writer(val_csv),
                "files": (train_csv, val_csv),
            }

        # 4) Pack patch parameters
        patch_params = {
            "patch_shape": patch_shape,
            "spacings": spacings,
            "overlap": overlap,
            "offset": offset,
            "center": center,
        }

        # 5) Loop slides serially
        rows = self.dataset_df.to_dict("records")
        for row in tqdm(rows, desc="Creating CellViT Dataset"):
            # process one slide
            records = self.process_slide_cellvit(
                row,
                patch_params,
                cpus,
                shift_x,
                shift_y,
                train_images_dir,
                train_labels_dir,
                unique_folds,
                self.wsi_col,
                self.wsa_col,
                group_to_label,
                ignore_groups,
                use_mask=use_mask,
            )
            # write each patch registration
            for fold, basename, role in records:
                if role == "val":
                    fold_csv_files[fold]["val_writer"].writerow([basename])
                else:
                    fold_csv_files[fold]["train_writer"].writerow([basename])

        # 6) Close CSV files
        for f in unique_folds:
            f_train, f_val = fold_csv_files[f]["files"]
            f_train.close()
            f_val.close()

        # 7) Write label_map.yaml
        inv_map = {v: k for k, v in group_to_label.items()}
        with open(os.path.join(output_dir, "label_map.yaml"), "w") as f:
            for lid in sorted(inv_map):
                f.write(f'{lid}: "{inv_map[lid]}"\n')

        self.logger.info("CellViT dataset creation complete (serial).")
        return

    def create_cellvit_dataset_singlerow_parallel(
        self,
        output_dir: str,
        group_to_label={"monocytes": 0, "lymphocytes": 1},
        ignore_groups={"ROI"},
        patch_shape=(1024, 1024, 3),
        spacings=(0.24199951445730394,),
        overlap=(0, 0),
        offset=(0, 0),
        center=False,
        n_cpus_global=8,  # global CPU count available
        shift_x=1,
        shift_y=1,
        use_ihc=False,
        use_mask=False,
    ):
        # 1) Ensure folds exist
        self.prepare_data_points_annotations()
        if "fold_id" not in self.dataset_df.columns:
            raise ValueError(
                "No 'fold_id' in dataset_df. Did you run split_and_save_kfold()?"
            )

        # 2) Create folder structure (same as before)
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

        if use_ihc:
            self.wsi_col = self.ihc_col

        # 3) Prepare CSV writers per fold (same as before)
        unique_folds = sorted(self.dataset_df["fold_id"].unique().astype(int))
        fold_csv_files = {}
        for f in unique_folds:
            fold_dir = os.path.join(splits_dir, f"fold_{f}")
            os.makedirs(fold_dir, exist_ok=True)
            train_csv_path = os.path.join(fold_dir, "train.csv")
            val_csv_path = os.path.join(fold_dir, "val.csv")
            f_train = open(train_csv_path, mode="w", newline="")
            f_val = open(val_csv_path, mode="w", newline="")
            fold_csv_files[f] = {
                "train_writer": csv.writer(f_train),
                "val_writer": csv.writer(f_val),
                "files": (f_train, f_val),
            }

        # 4) Prepare patch configuration parameters
        patch_params = {
            "patch_shape": patch_shape,
            "spacings": spacings,
            "overlap": overlap,
            "offset": offset,
            "center": center,
        }
        self.logger.info(
            f"PatchConfiguration: patch_shape={patch_shape}, spacings={spacings}, "
            f"overlap={overlap}, offset={offset}, center={center}"
        )

        # 5) Compute worker allocation:
        rows = self.dataset_df.to_dict("records")
        n_slides = len(rows)
        n_workers = min(n_slides, n_cpus_global)  # avoid oversubscription
        cpus_per_worker = max(1, n_cpus_global // n_workers)
        self.logger.info(
            f"Using {n_workers} worker processes with {cpus_per_worker} cpus each."
        )

        # 6) Process each slide in parallel using ProcessPoolExecutor
        all_patch_records = []  # holds tuples: (fold, patch_basename, role)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    self.process_slide_cellvit,
                    row,
                    patch_params,
                    cpus_per_worker,  # assign per-worker cpus here
                    shift_x,
                    shift_y,
                    train_images_dir,
                    train_labels_dir,
                    unique_folds,
                    self.wsi_col,
                    self.wsa_col,
                    group_to_label,
                    ignore_groups,
                    use_mask=use_mask,
                )
                for row in rows
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing slides"
            ):
                # try:
                patch_records = future.result()
                all_patch_records.extend(patch_records)
                # except Exception as e:
                #     self.logger.error(f"Error processing a slide: {e}")

        # 7) Write patch registrations to the fold CSV files
        for fold, patch_basename, role in all_patch_records:
            if role == "val":
                fold_csv_files[fold]["val_writer"].writerow([patch_basename])
            else:
                fold_csv_files[fold]["train_writer"].writerow([patch_basename])

        # 8) Close fold CSV files
        for f in unique_folds:
            f_train, f_val = fold_csv_files[f]["files"]
            f_train.close()
            f_val.close()

        # 9) Write label_map.yaml
        if group_to_label:
            label_map_inverted = {v: k for k, v in group_to_label.items()}
        else:
            label_map_inverted = {}
        label_map_path = os.path.join(output_dir, "label_map.yaml")
        with open(label_map_path, "w") as f:
            for label_id in sorted(label_map_inverted.keys()):
                class_name = label_map_inverted[label_id]
                f.write(f'{label_id}: "{class_name}"\n')

        self.logger.info(
            "CellViT dataset creation complete (with boundary clamp + dynamic updates)."
        )
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(
            f"label_map.yaml saved with {len(label_map_inverted)} entries."
        )


    def process_slide_cellvit(
        self,
        row,
        patch_params,
        patch_cpus,
        shift_x,
        shift_y,
        train_images_dir,
        train_labels_dir,
        unique_folds,
        wsi_col,
        wsa_col,
        group_to_label,
        ignore_groups,
        use_mask=False,
    ):
        """
        Process one slide:
        - If the WSI is missing, return empty.
        - Otherwise, create the patch iterator, save patch images (if not already saved),
        create CSV annotation files (if needed), and for each patch return a list of
        tuples (fold, patch_basename, role) where role is "train" or "val".
        Skips empty patches, reporting via tqdm.
        """
        import csv
        import os

        import matplotlib.pyplot as plt
        from asap_parser import parse_asap_dot_annotations
        from tqdm import tqdm
        from wholeslidedata.iterators import PatchConfiguration, create_patch_iterator

        def clamp(x, y, width, height, sx, sy):
            x_shifted = max(0, min(x + sx, width - 1))
            y_shifted = max(0, min(y + sy, height - 1))
            return x_shifted, y_shifted

        slide_id = row.get("Slide ID")
        try:
            val_fold_id = int(row.get("fold_id"))
        except (ValueError, TypeError):
            print(f"[{slide_id}] Invalid fold_id, skipping.")
            return []

        wsi_path = row.get(wsi_col)
        xml_path = row.get(wsa_col)
        mask_path = row.get("WSI Mask Path") if use_mask else None
        results = []

        if not (isinstance(wsi_path, str) and os.path.isfile(wsi_path)):
            print(f"[{slide_id}] Missing WSI => 0 patches generated.")
            return results

        if xml_path and os.path.isfile(xml_path):
            print(f"[{slide_id}] Found XML: {xml_path}\nProcessing it...")
            annotations = parse_asap_dot_annotations(
                xml_path=xml_path,
                group_to_label=group_to_label,
                ignore_groups=ignore_groups,
            )
            print(f"[{slide_id}] Found {len(annotations)} annotations.")
        else:
            annotations = []
            print(f"[{slide_id}] No XML => empty annotation list.")

        patch_config = PatchConfiguration(
            patch_shape=patch_params["patch_shape"],
            spacings=patch_params["spacings"],
            overlap=patch_params["overlap"],
            offset=patch_params["offset"],
            center=patch_params["center"],
        )

        print("Creating patch iterator...")
        patch_iterator = create_patch_iterator(
            image_path=wsi_path,
            mask_path=(mask_path if (mask_path and os.path.isfile(mask_path)) else None),
            patch_configuration=patch_config,
            cpus=patch_cpus,
            backend="asap" if use_mask else "openslide",
        )

        # Get total for tqdm
        try:
            total_patches = len(patch_iterator)
        except TypeError:
            total_patches = None

        print(
            f"[{slide_id}] Processing {len(annotations)} annotations in "
            f"{total_patches or '?'} patches..."
        )

        for idx_patch, patch_datainfo in enumerate(
            tqdm(patch_iterator, total=total_patches, desc=f"{slide_id} patches")
        ):
            # unpack depending on use_mask
            if use_mask:
                patch_data, _, info = patch_datainfo
            else:
                patch_data, info = patch_datainfo
            H, W, _ = info["tile_shape"]
            patch_x, patch_y = info["x"], info["y"]

            # collect annotations inside this patch
            patch_anns = []
            for x_g, y_g, label_id in annotations:
                x_local = x_g - patch_x
                y_local = y_g - patch_y
                if 0 <= x_local < W and 0 <= y_local < H:
                    x_l = round(x_local)
                    y_l = round(y_local)
                    x_c, y_c = clamp(x_l, y_l, W, H, shift_x, shift_y)
                    patch_anns.append((x_c, y_c, label_id))

            # skip empty patches
            if not patch_anns:
                #tqdm.write(f"[{slide_id}] Skipped patch {idx_patch} (no annotations)")
                continue

            patch_basename = f"{slide_id}_{idx_patch}"
            img_path = os.path.join(train_images_dir, patch_basename + ".png")
            csv_path = os.path.join(train_labels_dir, patch_basename + ".csv")

            # (A) Save patch image if not exists
            if not os.path.isfile(img_path):
                #print(f"[{slide_id}] Saving patch image: {img_path}")
                plt.imsave(img_path, patch_data.squeeze().astype("uint8"))

            # (B) Create patch annotation CSV
            with open(csv_path, mode="w", newline="") as cf:
                writer = csv.writer(cf)
                for x_c, y_c, lid in patch_anns:
                    writer.writerow([x_c, y_c, lid])

            # (C) Register patch in split CSVs
            for f in unique_folds:
                role = "val" if f == val_fold_id else "train"
                results.append((f, patch_basename, role))

        del patch_iterator
        return results


if __name__ == "__main__":
    USE_IHC = False
    USE_MASK = False

    # specify the output directory and the mapping of the groups to the labels
    output_dir = "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_full_3_cls_instanseg_annotated"
    group_to_label = {"monocytes": 0, "lymphocytes": 1, "other": 2}

    config = {
        "project": {"seed": 42},
        "dataset": {
            "path": "/work/grana_urologia/MONKEY_challenge/data/monkey-data",
            "annotation_dir": "annotations/xml_all_instanseg_3_classes",
            "wsi_col": "WSI PAS_CPG Path",
            "ihc_col": "WSI IHC_CPG Path",
            "wsa_col": "Annotation Path",
            "lymphocyte_half_box_size": 4.5,
            "monocyte_half_box_size": 5.0,
            "min_spacing": 0.25,
            "n_folds": 5,
            "balance_by": None,
            "num_bins_total_cells_count": 5,
        },
        "annotation_polygon_dir": "annotations_polygon",
        "yaml_wsi_wsa_dir": "/work/grana_urologia/MONKEY_challenge/source/configs/splits",
    }

    data_prep = DataPreparator(config)

    # create a CellVit plus plus finetune compatible dataset with the specified parameters

    # #NON MULTIPROCESSING VERSION
    # data_prep.create_cellvit_dataset_singlerow(
    #     output_dir=output_dir,
    #     group_to_label=group_to_label,
    #     ignore_groups={"ROI"},
    #     patch_shape=(256, 256, 3),
    #     spacings=(0.24199951445730394,),
    #     overlap=(0, 0),
    #     offset=(0, 0),
    #     center=False,
    #     cpus=2,
    #     use_ihc=USE_IHC,
    #     use_mask=USE_MASK,
    # )

    # MULTIPROCESSING VERSION
    data_prep.create_cellvit_dataset_singlerow_parallel(
        output_dir=output_dir,
        group_to_label=group_to_label,
        ignore_groups={"ROI"},
        patch_shape=(256, 256, 3),
        spacings=(0.24199951445730394,),
        overlap=(0, 0),
        offset=(0, 0),
        center=False,
        n_cpus_global=int(os.environ.get("SLURM_CPUS_PER_TASK", 2)),
        use_ihc=USE_IHC,
        use_mask=USE_MASK,
    )
