from pathlib import Path

# import numpy as np
# import torch


# def inspect_labels(fold_dir: Path):
#     val_dir = fold_dir / "val_results"
#     gt = torch.load(val_dir / "gt.pt")
#     preds_file = val_dir / "predictions.pt"
#     # If predictions file is missing, compute from probabilities:
#     if (val_dir / "predictions.pt").is_file():
#         preds = torch.load(preds_file)
#     else:
#         probs = torch.load(val_dir / "probabilities.pt")
#         preds = torch.argmax(probs, dim=1)

#     gt_np = gt.cpu().numpy()
#     preds_np = preds.cpu().numpy()

#     unique_gt = np.unique(gt_np)
#     unique_preds = np.unique(preds_np)

#     print(
#         f"{fold_dir.name} - GT unique labels: {unique_gt} | Pred unique labels: {unique_preds}"
#     )

#     print("Ground Truth counts:")
#     for label in unique_gt:
#         count = (gt_np == label).sum()
#         print(f"  Label {label}: {count}")

#     print("Prediction counts:")
#     for label in unique_preds:
#         count = (preds_np == label).sum()
#         print(f"  Label {label}: {count}")
#     print("-" * 50)

#     return gt_np, preds_np


# # Assuming results_dir is your parent folder with fold subdirectories:
# results_dir = Path(
#     "/work/grana_urologia/MONKEY_challenge/experiments/cellvit-finetune/PAS_sam-h_baseline_finetune_3_class_dataset/finetune_eval/results"
# )
# fold_dirs = sorted(
#     [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]
# )

# all_gt_list = []
# all_preds_list = []

# for fold in fold_dirs:
#     gt_np, preds_np = inspect_labels(fold)
#     all_gt_list.append(gt_np)
#     all_preds_list.append(preds_np)

# # Aggregate data across all folds
# all_gt = np.concatenate(all_gt_list, axis=0)
# all_preds = np.concatenate(all_preds_list, axis=0)

# # Compute cumulative counts for each ground truth and prediction label
# unique_gt_all, counts_gt_all = np.unique(all_gt, return_counts=True)
# unique_preds_all, counts_preds_all = np.unique(all_preds, return_counts=True)

# print("Cumulative Ground Truth counts across all folds:")
# for label, count in zip(unique_gt_all, counts_gt_all):
#     print(f"  Label {label}: {count}")

# print("Cumulative Prediction counts across all folds:")
# for label, count in zip(unique_preds_all, counts_preds_all):
#     print(f"  Label {label}: {count}")


# Set your dataset root paths
pas_dir = Path(
    "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train/images"
)
pas_labels = Path(
    "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train/labels"
)
ihc_dir = Path(
    "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_ihc/train/images"
)
ihc_labels = Path(
    "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_ihc/train/labels"
)

# Recursively count all files (change to .glob("*") for non-recursive)
pas_files = list(pas_dir.rglob("*.*"))  # or rglob("*.pt") for specific extensions
pas_labels_files = list(pas_labels.rglob("*.*"))
ihc_labels_files = list(ihc_labels.rglob("*.*"))
ihc_files = list(ihc_dir.rglob("*.*"))

print(f"Total files in PAS directory: {len(pas_files)}")
print(f"Total files in PAS labels directory: {len(pas_labels_files)}")
print(f"Total files in IHC directory: {len(ihc_files)}")
print(f"Total files in IHC labels directory: {len(ihc_labels_files)}")
# Check if the number of files in images between PAS and IHC are the same
assert len(pas_files) == len(
    ihc_files
), "Mismatch in number of image files between PAS and IHC"
# Check if the number of files in labels between PAS and IHC are the same
assert len(pas_labels_files) == len(
    ihc_labels_files
), "Mismatch in number of label files between PAS and IHC"
# Check if the number of files in images between PAS and labels are the same
assert len(pas_files) == len(
    pas_labels_files
), "Mismatch in number of image files between PAS and labels"
# Check if the number of files in images between IHC and labels are the same
assert len(ihc_files) == len(
    ihc_labels_files
), "Mismatch in number of image files between IHC and labels"
