import os

from experiments.BaselineDetectronExperiment import BaselineDetectronExperiment
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
    # data_preparator = DataPreparator(config)
    # dataset_df, folds_paths_dict = data_preparator.prepare_data()
    # print(dataset_df.head())
    # print(folds_paths_dict)
    experiment = BaselineDetectronExperiment(args, config)
    experiment.train()
    # experiment.test()
