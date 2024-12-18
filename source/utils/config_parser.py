import argparse
import os
import pprint
import time

import yaml
from logger import get_logger

# Constants for default values

# PROJECT
BASE_PATH = "/work/grana_urologia"  # base path used for other paths
TIMESTAMP = time.strftime("%m-%d__%H-%M-%S")
DEFAULT_CONFIG_PATH = "./configs/baseline/base.yaml"
DEFAULT_PROJECT_NAME = "wsi_uro_patch_diagnosis"
DEFAULT_LOG_PATH = "./logs/scripts"
DEFAULT_OUTPUT_DIR = "./model_results"
DEFAULT_IMG_SAVE_DIR_PATH = "./imgs"
DEFAULT_MODEL_SAVE_DIR_PATH = None
DEFAULT_SEED = 42
DEFAULT_USE_WANDB = True
DEFAULT_WAND_MODEL_WATCH = True
DEFAULT_FILE_LOG = False


# Dataset defaults
DEFAULT_DATASET_NAME = "wsi_uro_v2_512px_only_annotations"
DEFAULT_DATASET_PATH = f"{BASE_PATH}/WSI_patches"
DEFAULT_EVAL_ON_TEST_DURING_TRAINING = False
DEFAULT_VALID_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.2
DEFAULT_RESIZE_DIMS = 512
DEFAULT_STRATIFY_BY = "tumor"
DEFAULT_DROP_NONE_CLASS = True
DEFAULT_SPLIT_BY_WSI = False
DEFAULT_AUGS = "none"
DEFAULT_NORMALIZE = True

# Model defaults
DEFAULT_MODEL_NAME = "resnet50"
DEFAULT_PRETRAINED = True
DEFAULT_FREEZE_ALL_LAYERS = False
DEFAULT_AMOUNT_FROZEN_LAYERS = 0

# Optimizer defaults
DEFAULT_OPTIMIZER = "Adam"
DEFAULT_WEIGHTED_LOSS = False

# Training defaults
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_WEIGHTED_CLASSES = False
DEFAULT_LABEL_THRESHOLD = 0.5
DEFAULT_EVALUATE_EVERY = 5
DEFAULT_USE_BINARY_CLASSES = True
DEFAULT_CONTINUE_TRAINING = False

# # Patchifier defaults
# DEFAULT_WSI_DATA_PATH = f"{BASE_PATH}/WSI-data"
# DEFAULT_WSI_GEOJSON_PATH = f"{BASE_PATH}/wsi_geojson_data"
# DEFAULT_OUTPUT_PATCHES_PATH = f"{BASE_PATH}/WSI_patches"
# DEFAULT_PATCHIFIER_PATCH_SIZE = 512
# DEFAULT_PATCHIFIER_PATCH_ZOOM_MULTIPLIER = 1
# DEFAULT_MAX_PATCH_BACKGROUND = 0.8
# DEFAULT_LABEL_MIN_AREA_INTERSECTION = 0.05
# DEFAULT_USE_OTSU = False
# DEFAULT_USE_CONTOUR_APPROX = True
# DEFAULT_SAVE_MASK_PATCHES = True
# DEFAULT_SAVE_ANNOT_PATCHES = True


def str2bool(v):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.

    Args:
        v (str): The string representation of the boolean value.

    Returns:
        bool: The corresponding boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    # Start argparse
    parser = argparse.ArgumentParser(description="MONKEY Challenge ZipLab Project")

    # Arguments to be specified by args
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_false",
        default=True,
        dest="use_wandb",
        help="Disables wandb logging",
    )
    parser.add_argument(
        "--test",
        action="store_false",
        default=True,
        dest="train",
        help="If True, only test the model, skipping training. Default is False.",
    )

    # Add argument for config file location
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to the configuration file"
    )

    # Parse config argument to get the config file path
    args, _ = parser.parse_known_args()
    config_path = args.config
    logger = get_logger(name="argparser", args=args)

    # Load configuration from YAML file if it exists
    logger.info(
        f"\n{10*'='}\nTrying to load configuration file from specified or default '--config' args..."
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}:")
            config_formatted = pprint.pformat(config, indent=4)
            logger.debug(config_formatted)
    else:
        logger.info(
            f"Configuration file {config_path} does not exist.\nLoading default config, please refer to the utils/config_parser.py script for the default args."
        )
        config = {}

    # Extract configuration sections
    project_config = config.get("project", {})
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    optimizer_config = config.get("optimizer", {})
    training_config = config.get("training", {})
    patchifier_config = config.get("patchifier", {})
    other_config = config.get("other", {})

    # Set default timestamp
    timestamp = (
        time.strftime("%m-%d__%H-%M-%S")
        if project_config.get("timestamp", "auto") == "auto"
        else project_config.get("timestamp")
    )

    # Project args
    parser.add_argument(
        "--project_name",
        default=project_config.get("name", DEFAULT_PROJECT_NAME),
        help="Project name for experiment tracking in wandb",
    )
    parser.add_argument(
        "--log_dir",
        default=project_config.get("log_dir", DEFAULT_LOG_PATH),
        help="Set log directory",
    )
    parser.add_argument(
        "--output_dir",
        default=project_config.get("output_dir", DEFAULT_OUTPUT_DIR),
        help="Path for saving model info and weights",
    )
    parser.add_argument(
        "--timestamp",
        default=timestamp,
        help="Set timestamp for folder naming and wandb tracking",
    )
    parser.add_argument(
        "--seed",
        default=project_config.get("seed", DEFAULT_SEED),
        type=int,
        help="Seed to use for reproducing results",
    )
    parser.add_argument(
        "--continue_training",
        type=str2bool,
        default=project_config.get("continue_training", DEFAULT_CONTINUE_TRAINING),
        help="Continue training from the last checkpoint",
    )
    parser.add_argument(
        "--file_log",
        type=str2bool,
        default=project_config.get("file_log", DEFAULT_FILE_LOG),
        help="Set file logging directory",
    )
    parser.add_argument(
        "--img_save_dir_path",
        default=project_config.get("img_save_dir_path", DEFAULT_IMG_SAVE_DIR_PATH),
        help="Path of images used for testing",
    )
    parser.add_argument(
        "--model_save_dir_path",
        default=project_config.get("model_save_dir_path", DEFAULT_MODEL_SAVE_DIR_PATH),
        help="Path of model dir weights used for continuing training and/or testing. Default is None.",
    )
    parser.add_argument(
        "--wandb_model_watch",
        type=str2bool,
        default=project_config.get("wandb_model_watch", DEFAULT_WAND_MODEL_WATCH),
        help="Enable wandb model watch",
    )

    # Dataset args
    parser.add_argument(
        "--dataset_name",
        default=dataset_config.get("name", DEFAULT_DATASET_NAME),
        help="Dataset name for wandb",
    )
    parser.add_argument(
        "--dataset_path",
        default=dataset_config.get("path", DEFAULT_DATASET_PATH),
        help="Path of the wsi dataset patches folder",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=dataset_config.get("valid_ratio", DEFAULT_VALID_RATIO),
        help="Ratio of the validation set among the train set",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=dataset_config.get("test_ratio", DEFAULT_TEST_RATIO),
        help="Ratio of the test set among the whole dataset",
    )
    # parser.add_argument(
    #     "--eval_on_test_during_training",
    #     type=str2bool,
    #     default=dataset_config.get(
    #         "eval_on_test_during_training", DEFAULT_EVAL_ON_TEST_DURING_TRAINING
    #     ),
    #     help="If True, use the test set during training for evaluation instead of the validation set.\nThe --valid_ratio must be 0 to use this flag. Default is False.",
    # )

    parser.add_argument(
        "--resize_dims",
        default=dataset_config.get("resize_dims", DEFAULT_RESIZE_DIMS),
        type=int,
        help="Size of the input images",
    )
    parser.add_argument(
        "--stratify_by",
        default=dataset_config.get("stratify_by", DEFAULT_STRATIFY_BY),
        type=str,
        help="Selects the label to stratify by if split_by_patch is used",
    )
    # parser.add_argument(
    #     "--drop_none_class",
    #     type=str2bool,
    #     default=dataset_config.get("drop_none_class", DEFAULT_DROP_NONE_CLASS),
    #     help="If True, use the none class in the dataset",
    # )
    # parser.add_argument(
    #     "--split_by_wsi",
    #     type=str2bool,
    #     default=dataset_config.get("split_by_wsi", DEFAULT_SPLIT_BY_WSI),
    #     help="If True, split the dataset by WSI",
    # )
    parser.add_argument(
        "--augs",
        default=dataset_config.get("augs", DEFAULT_AUGS),
        type=str,
        choices=["full", "essentials", "essentials-no-resize", "resize", "none"],
        help="Select augmentations",
    )
    parser.add_argument(
        "--normalize",
        type=str2bool,
        default=dataset_config.get("normalize", DEFAULT_NORMALIZE),
        help="Enable dataset normalization",
    )

    # Model args
    parser.add_argument(
        "--model",
        default=model_config.get("name", DEFAULT_MODEL_NAME),
        type=str,
        help="Which model to choose for training",
    )
    parser.add_argument(
        "--pretrained",
        type=str2bool,
        default=model_config.get("pretrained", DEFAULT_PRETRAINED),
        help="Load a pretrained model (if possible based on the '--model=model_name' specified. Default is True.",
    )
    parser.add_argument(
        "--freeze_all_layers",
        type=str2bool,
        default=model_config.get("freeze_all_layers", DEFAULT_FREEZE_ALL_LAYERS),
        help="Freeze all layers in the model",
    )
    parser.add_argument(
        "--amount_frozen_layers",
        type=int,
        default=model_config.get("amount_frozen_layers", DEFAULT_AMOUNT_FROZEN_LAYERS),
        help="Number of layers to freeze in the model",
    )

    # Optimizer args
    parser.add_argument(
        "--optimizer",
        default=optimizer_config.get("name", DEFAULT_OPTIMIZER),
        type=str,
        choices=["Adam", "SGD"],
        help="Select optimizer",
    )
    parser.add_argument(
        "--weighted_loss",
        type=str2bool,
        default=optimizer_config.get("weighted_loss", DEFAULT_WEIGHTED_LOSS),
        help="Use weighted Cross Entropy Loss",
    )

    # Training args
    parser.add_argument(
        "--lr",
        default=training_config.get("learning_rate", DEFAULT_LEARNING_RATE),
        type=float,
        help="Learning rate to use",
    )
    parser.add_argument(
        "--epochs",
        default=training_config.get("epochs", DEFAULT_EPOCHS),
        type=int,
        help="Max number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=training_config.get("batch_size", DEFAULT_BATCH_SIZE),
        type=int,
        help="Minibatch size",
    )
    parser.add_argument(
        "--weighted_classes",
        type=str2bool,
        default=training_config.get("weighted_classes", DEFAULT_WEIGHTED_CLASSES),
        help="Use weighted sampler for unbalanced classes",
    )
    # parser.add_argument(
    #     "--label_threshold",
    #     type=float,
    #     default=training_config.get("label_threshold", DEFAULT_LABEL_THRESHOLD),
    #     help="Multi label classification threshold",
    # )
    parser.add_argument(
        "--evaluate_every",
        default=training_config.get("evaluate_every", DEFAULT_EVALUATE_EVERY),
        type=int,
        help="Number of epochs interval to wait to evaluating the metrics on the train/val sets",
    )
    # parser.add_argument(
    #     "--binary_classes",
    #     type=str2bool,
    #     default=training_config.get("binary_classes", DEFAULT_USE_BINARY_CLASSES),
    #     help="Enable binary classification. Default is False.",
    # )

    # Patchifier args
    # parser.add_argument(
    #     "--wsi_data_path",
    #     type=str,
    #     default=patchifier_config.get("wsi_data_path", DEFAULT_WSI_DATA_PATH),
    #     help="Path to original WSIs to patchify/infer from",
    # )

    # parser.add_argument(
    #     "--wsi_geojson_path",
    #     type=str,
    #     default=patchifier_config.get("wsi_geojson_path", DEFAULT_WSI_GEOJSON_PATH),
    #     help="Path to the WSIs associated geojson annotations",
    # )

    # parser.add_argument(
    #     "--output_patches_path",
    #     type=str,
    #     default=patchifier_config.get(
    #         "output_patches_path", DEFAULT_OUTPUT_PATCHES_PATH
    #     ),
    #     help="Path where the patchified WSIs will be saved.",
    # )

    # parser.add_argument(
    #     "--patchifier_patch_size",
    #     default=patchifier_config.get(
    #         "patchifier_patch_size", DEFAULT_PATCHIFIER_PATCH_SIZE
    #     ),
    #     type=int,
    #     help="Patch size to use for patchifying the WSIs original data.",
    # )

    # parser.add_argument(
    #     "--patch-multiplier",
    #     default=patchifier_config.get(
    #         "patch-multiplier", DEFAULT_PATCHIFIER_PATCH_ZOOM_MULTIPLIER
    #     ),
    #     type=int,
    #     help="Multiplier for the patch size to simulate zoom levels.",
    # )

    # parser.add_argument(
    #     "--max_patch_background",
    #     default=patchifier_config.get(
    #         "max_patch_background", DEFAULT_MAX_PATCH_BACKGROUND
    #     ),
    #     type=float,
    #     help="Max background threshold (0 to 1 (percentage)) to select the patches from the patchification of the WSIs. If a patch has less than the max background in it, it will be selected and saved, else it will be discared.",
    # )

    # parser.add_argument(
    #     "--label_min_area_intersection",
    #     default=patchifier_config.get(
    #         "label_min_area_intersection", DEFAULT_LABEL_MIN_AREA_INTERSECTION
    #     ),
    #     type=float,
    #     help="Min value from 0 (0%) to 1 (100%) of area intersection between the annotationst in the WSI and the patch are to have for labeling the patch with the same annotations.",
    # )

    # parser.add_argument(
    #     "--use_otsu",
    #     type=str2bool,
    #     default=patchifier_config.get("use_otsu", DEFAULT_USE_OTSU),
    #     help="Enable Otsu thresholding for segmenting the WSI tissue and patchifying it. Default is False (disabled).",
    # )

    # parser.add_argument(
    #     "--use_contour_approx",
    #     type=str2bool,
    #     default=patchifier_config.get("use_contour_approx", DEFAULT_USE_CONTOUR_APPROX),
    #     help="Enable contour approximation to mask the tissue of the WSI, smoothing borders and filling segmentation holes in the masked tissue. Default is True (enabled).",
    # )
    # parser.add_argument(
    #     "--save_mask_patches",
    #     type=str2bool,
    #     default=patchifier_config.get("save_mask_patches", DEFAULT_SAVE_MASK_PATCHES),
    #     help="Enable saving of tissue mask patches along the original wsi patches. Default is True",
    # )
    # parser.add_argument(
    #     "--save_annot_patches",
    #     type=str2bool,
    #     default=patchifier_config.get("save_annot_patches", DEFAULT_SAVE_ANNOT_PATCHES),
    #     help="Enable saving of annotation mask patches along the original wsi patches, if annotations are passed during the patchification. Default is True",
    # )

    # Parse and return the arguments
    args, _ = parser.parse_known_args()

    # # assert if some config are not contradicting
    # if args.eval_on_test_during_training and args.valid_ratio != 0:
    #     raise ValueError(
    #         "If using 'eval_on_test_during_training', the 'valid_ratio' must be 0."
    #     )

    formatted_loaded_args = pprint.pformat(args, indent=4)
    logger.info(
        f"Load of args completed!\nLoaded the following configuration:\n{formatted_loaded_args}\n{'='*10}"
    )

    return args


# debug/test usage with formatted print
if __name__ == "__main__":
    args = get_args()
    pprint.pp(args)
