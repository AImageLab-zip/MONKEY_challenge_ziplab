import os
from pprint import pprint

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def split_and_save_kfold(
    n_folds: int = 5,
    dataset_df: pd.DataFrame = pd.DataFrame(),
    balance_by=None,
    seed: int = 42,
):
    """
    Split the data into n folds, from a metadata dataframe, then save the folds with the respective patient ids
    """

    print(f"Making {n_folds} splits for a daset of n° {len(dataset_df)} istances")
    dataset_df = dataset_df.reset_index(drop=True)  # Reset before splitting

    # Add a fold_id column initialized to -1 (or reset if already present)
    # dataset_df = dataset_df.copy()
    dataset_df["fold_id"] = -1

    fold_patient_ids = {}

    # Choose the splitter
    if balance_by and balance_by in dataset_df.columns:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        stratify_col = dataset_df[balance_by]
        print(
            f"Stratifying by {balance_by} with {len(stratify_col.unique())} unique values"
        )
    else:
        # Use random non-stratified splitter if no column is specified
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        stratify_col = None
        print("No stratification column provided, using random split")

    # Split the data
    for fold, (train_idx, patients_idx_list) in enumerate(
        skf.split(dataset_df, stratify_col)
    ):
        validation_data = dataset_df.iloc[patients_idx_list]

        # Assign fold IDs to the dataframe
        dataset_df.loc[patients_idx_list, "fold_id"] = fold

        # Save the patient IDs for the current fold
        fold_patient_ids[fold] = validation_data["Slide ID"].tolist()

    return fold_patient_ids


if __name__ == "__main__":
    N_FOLDS = 5
    SEED = 42
    BALANCE_SPLIT_BY = None

    # Load the original CSV
    filelist_csv_path = "/work/grana_urologia/MONKEY_challenge/data/filelist_wsi/filelist_monkey_pas_cpg.csv"
    dataset_df_path = (
        "/work/grana_urologia/MONKEY_challenge/data/dataset_metadata_df.csv"
    )
    dataset_filelist_df = pd.read_csv(filelist_csv_path)

    # Extract patient IDs from the filename
    dataset_filelist_df["patient_id"] = dataset_filelist_df["path"].apply(
        lambda x: os.path.basename(x).split("_PAS")[0]
    )

    # Output directory
    output_dir = "/work/grana_urologia/MONKEY_challenge/data/filelist_wsi/fold_splits"
    os.makedirs(output_dir, exist_ok=True)

    dataset_df = pd.read_csv(dataset_df_path)

    patients_fold_split_dict = split_and_save_kfold(
        n_folds=N_FOLDS,
        dataset_df=dataset_df,
        balance_by=BALANCE_SPLIT_BY,
        seed=SEED,
    )

    pprint(patients_fold_split_dict, indent=4)

    # Generate one CSV per fold
    for fold_id, patient_ids in patients_fold_split_dict.items():
        fold_df = (
            dataset_filelist_df[dataset_filelist_df["patient_id"].isin(patient_ids)]
            .sort_values("patient_id")
            .drop(columns=["patient_id"])
        )  # sort before dropping
        fold_df.to_csv(os.path.join(output_dir, f"fold_{fold_id}.csv"), index=False)
