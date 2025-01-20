from experiments.BaselineDetectronExperiment import BaselineDetectronExperiment
from utils.config_parser import get_args_and_config


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
    args, config = get_args_and_config()
    experiment = BaselineDetectronExperiment(args, config)
    experiment.train()
    # experiment.test()
