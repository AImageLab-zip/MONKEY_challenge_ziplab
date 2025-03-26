import os

from utils.config_parser import get_args_and_config
from utils.data_preparation import DataPreparator


if __name__ == "__main__":
    # Parse arguments and config
    args, config = get_args_and_config()

    #specify the output directory and the mapping of the groups to the labels
    output_dir = (
        "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel"
    )
    group_to_label = {"monocytes": 0, "lymphocytes": 1, "other": 2}

    data_prep = DataPreparator(config)
    
    #create a CellVit plus plus finetune compatible dataset with the specified parameters
    
    data_prep.create_cellvit_dataset_singlerow_parallel(
        output_dir=output_dir,
        group_to_label=group_to_label,
        ignore_groups={"ROI"},
        patch_shape=(256, 256, 3),
        spacings=(0.24199951445730394,),
        overlap=(0, 0),
        offset=(0, 0),
        center=False,
        n_cpus_global=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)),
    )
