# from experiments.BaselineDetectronExperiment import BaselineDetectronExperiment
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
    args, config = get_args_and_config()

    output_dir = "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_classes"
    group_to_label = {"monocytes": 0, "lymphocytes": 1, "other": 2}

    data_prep = DataPreparator(config)
    data_prep.create_cellvit_dataset_singlerow(
        output_dir=output_dir,
        group_to_label=group_to_label,
        ignore_groups={"ROI"},
        patch_shape=(256, 256, 3),
        spacings=(0.24199951445730394,),
        overlap=(0, 0),
        offset=(0, 0),
        center=False,
        cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
    )
    # experiment = BaselineDetectronExperiment(args, config)
    # experiment.train()
    # experiment.test()
