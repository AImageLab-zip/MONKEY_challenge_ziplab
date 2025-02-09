import os

import pandas as pd

# def create_fold_all(dataset_root, val_fold=0):
#     """
#     Creates a fold_all directory in the splits folder and generates a train.csv
#     containing validation samples from all folds except the chosen validation fold,
#     and a val.csv containing only the validation samples from the specified fold.

#     :param dataset_root: Path to the root dataset directory
#     :param val_fold: The fold to be used as validation (default is 0)
#     """
#     splits_dir = os.path.join(dataset_root, "splits")
#     fold_all_dir = os.path.join(splits_dir, "fold_all")
#     os.makedirs(fold_all_dir, exist_ok=True)

#     train_csv_path = os.path.join(fold_all_dir, "train.csv")
#     val_csv_path = os.path.join(fold_all_dir, "val.csv")

#     val_df = None  # Store only the specified validation fold dataframe
#     train_dfs = []  # Store validation dataframes from all other folds

#     for fold in range(5):  # Assuming folds 0 to 4
#         fold_dir = os.path.join(splits_dir, f"fold_{fold}")
#         val_file = os.path.join(fold_dir, "val.csv")

#         if os.path.exists(val_file):
#             df = pd.read_csv(val_file, header=None)
#             if fold == val_fold:
#                 val_df = df
#             else:
#                 train_dfs.append(df)

#     # Create train.csv (all val.csv combined except validation fold)
#     if train_dfs:
#         train_df = pd.concat(train_dfs, ignore_index=True)
#         train_df.to_csv(train_csv_path, index=False, header=False)
#         print(f"Created {train_csv_path} with {len(train_df)} entries.")
#     else:
#         print("No training data found from folds.")

#     # Create val.csv (only the specified validation fold)
#     if val_df is not None:
#         val_df.to_csv(val_csv_path, index=False, header=False)
#         print(f"Created {val_csv_path} with {len(val_df)} entries.")
#     else:
#         print(f"No validation data found in fold_{val_fold}.")


# if __name__ == "__main__":
#     dataset_root = (
#         "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel"
#     )
#     val_fold = 1
#     create_fold_all(dataset_root, val_fold=val_fold)


def create_patient_based_split(dataset_root, patient_id="A_P000038"):
    """
    Creates a directory in the splits folder and generates:
    - train.csv: containing all samples except those from the specified patient.
    - val.csv: containing only samples from the specified patient.

    :param dataset_root: Path to the root dataset directory
    :param patient_id: The patient ID prefix to exclude from training and use for validation
    """
    splits_dir = os.path.join(dataset_root, "splits")
    patient_split_dir = os.path.join(splits_dir, "patient_split_5")
    os.makedirs(patient_split_dir, exist_ok=True)

    train_csv_path = os.path.join(patient_split_dir, "train.csv")
    val_csv_path = os.path.join(patient_split_dir, "val.csv")

    all_dfs = []  # Store all validation dataframes
    val_dfs = []  # Store only patient-specific data

    for fold in range(5):  # Assuming folds 0 to 4
        fold_dir = os.path.join(splits_dir, f"fold_{fold}")
        val_file = os.path.join(fold_dir, "val.csv")

        if os.path.exists(val_file):
            df = pd.read_csv(val_file, header=None)
            all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)

        # Separate validation and training data based on patient ID prefix
        val_df = full_df[full_df[0].str.startswith(patient_id)]
        train_df = full_df[~full_df[0].str.startswith(patient_id)]

        # Save the new splits
        train_df.to_csv(train_csv_path, index=False, header=False)
        val_df.to_csv(val_csv_path, index=False, header=False)

        print(f"Created {train_csv_path} with {len(train_df)} entries.")
        print(f"Created {val_csv_path} with {len(val_df)} entries.")
    else:
        print("No data found in validation splits.")


if __name__ == "__main__":
    dataset_root = (
        "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel"
    )
    create_patient_based_split(dataset_root)
