import argparse
import os
import pprint

import yaml

from .logger import get_logger

# Constants for default values
DEFAULT_CONFIG_PATH = "./configs/baseline/detectron2_baseline.yml"


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


def load_yaml(yaml_path, logger):
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {yaml_path}:")
            config_formatted = pprint.pformat(config, indent=4)
            # logger.debug(config_formatted)
    else:
        logger.info(
            f"Configuration file {yaml_path} does not exist.\nLoading empty config dictionary."
        )
        config = {}

    return config


def get_args_and_config():
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
        "--fold",
        type=int,
        default=None,
        help="Fold number to train on",
        required=False,
    )

    parser.add_argument(
        "--test",
        action="store_false",
        default=True,
        dest="train",
        help="If True, only test the model, skipping training. Default is False.",
    )

    parser.add_argument(
        "--continue_training",
        action="store_true",
        default=False,
        help="If True, continue training from existing model. Default is False.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to the model directory to load the model from",
        required=False,
    )

    # Add argument for config file location
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to the configuration file"
    )

    # Parse config argument to get the config file path
    args, _ = parser.parse_known_args()
    config_path = args.config
    logger = get_logger(name="argparser", args=args)

    # check if model path is passed if continue_training is True
    if args.continue_training and args.model_dir is None:
        raise ValueError(
            "Model directory path must be provided if continue_training is True."
        )

    # Load configuration from YAML file if it exists
    logger.info(
        f"\n{10*'='}\nTrying to load configuration file from specified or default '--config' args..."
    )

    config = load_yaml(config_path, logger)

    formatted_loaded_args = pprint.pformat(vars(args), indent=4)
    formatted_config = pprint.pformat(config, indent=4)
    logger.info("Load of args and config completed!")
    logger.debug(
        f"Load of args and config completed!\nLoaded the following args:\n{formatted_loaded_args}\nLoaded the following configs:\n{formatted_config}\n{'='*10}"
    )

    return args, config


# debug/test usage with formatted print
if __name__ == "__main__":
    args, config = get_args_and_config()
    pprint.pp(args)
    pprint.pp(config)
