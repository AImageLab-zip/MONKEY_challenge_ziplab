import os

from utils.data_preparation import DataPreparator

if __name__ == "__main__":
    USE_IHC = True

    # specify the output directory and the mapping of the groups to the labels
    output_dir = "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_ihc"
    group_to_label = {"monocytes": 0, "lymphocytes": 1, "other": 2}

    config = {
        "project": {"seed": 42},
        "dataset": {
            "path": "/work/grana_urologia/MONKEY_challenge/data/monkey-data",
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
        "yaml_wsi_wsa_dir": "./configs/splits",
    }

    data_prep = DataPreparator(config)

    # create a CellVit plus plus finetune compatible dataset with the specified parameters

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
        use_ihc=USE_IHC,
    )
