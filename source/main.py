import os

from utils.config_parser import get_args_and_config
from utils.data_preparation import DataPreparator


def test_torch():
    """
    Test if torch is installed and working with GPU
    """
    import torch

    print(torch.__version__)
    print(torch.cuda.is_available())
    print("It's working!")


if __name__ == "__main__":
    # test_torch()

    print(os.getcwd())
    args, config = get_args_and_config()
    print(config)
    data_preparator = DataPreparator(config)
    data_preparator.prepare_data()
