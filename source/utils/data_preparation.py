import glob
import os

import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

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

    def create_dataset_df(self):
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
        wsa_dir = (
            f"{self.annotation_polygon_dir}" + r"/*_polygon.xml"
        )  # Annotation files
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
                self.dataset_df.loc[
                    self.dataset_df["Slide ID"] == patient_name, "WSI PAS_CPG Path"
                ] = pas_cpg_path
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

    def split_and_save_kfold(self):
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

            # Save the fold yaml path to the dictionary
            self.fold_yaml_paths_dict[fold] = fold_file_path

            # Save the fold yaml file
            with open(fold_file_path, "w") as f:
                yaml.dump({"training": train_yaml, "validation": validation_yaml}, f)

            print(f"Fold {fold + 1} saved -> {fold_file_path}")

        return self.dataset_df, self.fold_yaml_paths_dict

    def prepare_data(self):
        """
        Prepare the data for the WSD task.
        """
        # 1. Create bounding boxes annotations
        self.create_bboxes_annots()

        # 2. Create and update the dataset dataframe
        self.create_dataset_df()

        # 3. Split the data into n folds and save to n .yml files in the specified directory
        dataset_df, folds_paths_dict = self.split_and_save_kfold()

        return dataset_df, folds_paths_dict


# def folders_to_yml(wsi_dir: str, wsa_dir: str, output_dir: str, output_name: str):
#     """
#     Generate a yaml file to be used as WSD dataconfig from a folder of slides and a folder of annotation or mask files.
#     Assumes files use the same name for both the slides and masks.
#     """

#     wsa_list = glob.glob(wsa_dir)

#     yaml_dict = {"training": []}
#     # yaml_dict = {'training': [], 'validation': []}
#     for wsa in wsa_list:
#         patient_name = os.path.basename(wsa).split(
#             os.path.basename(wsa_dir).split("*")[1]
#         )[0]  # monocytes
#         #     print(patient_name)
#         if os.path.isfile(os.path.join(wsi_dir, patient_name + "_PAS_CPG.tif")):
#             wsi = os.path.join(wsi_dir, patient_name + "_PAS_CPG.tif")
#             print("match found:    ", patient_name)
#             yaml_dict["training"].append(
#                 {"wsa": {"path": str(wsa)}, "wsi": {"path": str(wsi)}}
#             )

#             # # validation if needed
#             # yaml_dict['validation'].append(
#             #         {"wsa": {"path": str(wsa)}, "wsi": {"path": str(wsi)}})

#         else:
#             print("no match found:    ", patient_name)

#     # make a folder for output
#     if not (os.path.isdir(output_dir)):
#         os.mkdir(output_dir)

#     with open(os.path.join(output_dir, output_name), "w") as file:
#         yaml.safe_dump(yaml_dict, file)

if __name__ == "__main__":
    args, config = get_args_and_config()

    print(config)
    DataPrep = DataPreparator(config=config)
    dataset_df, folds_paths_dict = DataPrep.prepare_data()
    # create_bboxes_annots(config=config)
