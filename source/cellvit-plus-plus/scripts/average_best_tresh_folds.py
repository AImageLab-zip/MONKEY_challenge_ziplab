from pathlib import Path
import torch
import numpy as np
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score, PrecisionRecallCurve

# Path to the master folder that contains the validation folds (subfolders)
master_folder = Path(
    "/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/logs_local"
)

# Initialize lists to store metrics from each fold
fold_results = []

# Iterate over each subfolder in the master folder
for fold_dir in master_folder.iterdir():
    if not fold_dir.is_dir():
        continue
    # Look for the validation results folder inside this fold directory
    val_result_dir = fold_dir / "val_results"
    if not val_result_dir.exists():
        print(f"Skipping {fold_dir.name}: no val_results folder found.")
        continue

    # Load ground truth and predicted probabilities for this fold
    try:
        gt = torch.load(val_result_dir / "gt.pt")
        probabilities = torch.load(val_result_dir / "probabilities.pt")
    except Exception as e:
        print(f"Error loading files in {val_result_dir}: {e}")
        continue

    # Initialize metrics functions (binary classification assumed)
    pr_curve_func = PrecisionRecallCurve(task="binary")
    f1_score_func = F1Score(task="binary")
    auroc_func = AUROC(task="binary")
    accuracy_func = Accuracy(task="binary")
    conf_matrix_func = ConfusionMatrix(task="binary")

    # Compute the precision-recall curve using probabilities for the positive class
    precision, recall, thresholds = pr_curve_func(probabilities[:, 1], gt)

    # --- Find Best Recall Threshold ---
    precision_constraint = 0.5  # Adjust to control false positives
    valid_indices = torch.where(precision > precision_constraint)[0]
    if len(valid_indices) > 0:
        best_recall_idx = valid_indices[torch.argmax(recall[valid_indices])]
    else:
        best_recall_idx = torch.argmax(recall)
    best_recall_thresh = thresholds[best_recall_idx]

    # --- Find Best F1-Score Threshold ---
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_idx = torch.argmax(f1_scores)
    best_f1_thresh = thresholds[best_f1_idx]

    # Make predictions using:
    # 1) Default 0.5 threshold
    # 2) Best Recall threshold
    # 3) Best F1 threshold
    pred_argmax = probabilities[:, 1] > 0.5
    pred_recall = probabilities[:, 1] > best_recall_thresh
    pred_f1 = probabilities[:, 1] > best_f1_thresh

    # Compute confusion matrices (for informational purposes)
    conf_matrix_argmax = conf_matrix_func(pred_argmax, gt)
    conf_matrix_recall = conf_matrix_func(pred_recall, gt)
    conf_matrix_f1 = conf_matrix_func(pred_f1, gt)

    # Compute scalar metrics
    f1_argmax = f1_score_func(pred_argmax, gt)
    f1_recall = f1_score_func(pred_recall, gt)
    f1_f1 = f1_score_func(pred_f1, gt)
    acc_argmax = accuracy_func(pred_argmax, gt)
    acc_recall = accuracy_func(pred_recall, gt)
    acc_f1 = accuracy_func(pred_f1, gt)
    auroc_val = auroc_func(probabilities[:, 1], gt)

    # Store metrics in a dictionary for this fold
    fold_results.append(
        {
            "fold": fold_dir.name,
            "auroc": auroc_val.item(),
            "f1_argmax": f1_argmax.item(),
            "f1_recall": f1_recall.item(),
            "f1_f1": f1_f1.item(),
            "acc_argmax": acc_argmax.item(),
            "acc_recall": acc_recall.item(),
            "acc_f1": acc_f1.item(),
            "best_recall_thresh": best_recall_thresh.item(),
            "best_f1_thresh": best_f1_thresh.item(),
            "recall_best_recall": recall[best_recall_idx].item(),
            "precision_best_recall": precision[best_recall_idx].item(),
            "recall_best_f1": recall[best_f1_idx].item(),
            "precision_best_f1": precision[best_f1_idx].item(),
        }
    )

# Print individual fold metrics
print("\nPer-Fold Metrics:")
for res in fold_results:
    print(res)

# Compute mean metrics across folds (for all numeric keys except 'fold')
if fold_results:
    mean_metrics = {}
    numeric_keys = [key for key in fold_results[0].keys() if key != "fold"]
    for key in numeric_keys:
        mean_metrics[key] = np.mean([res[key] for res in fold_results])
    print("\nMean Metrics Across Folds:")
    print(mean_metrics)
else:
    print("No valid validation results found.")
