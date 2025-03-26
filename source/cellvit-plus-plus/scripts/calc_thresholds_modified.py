# -*- coding: utf-8 -*-
# Find best classifier threshold using PR curve (optimized for recall & F1) & Compute Confusion Matrix
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Modified for recall optimization - MONKEY Challenge
# Added Confusion Matrix Calculation & Best F1 Threshold
# Institute for Artificial Intelligence in Medicine, University Medicine Essen

from pathlib import Path

import torch
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score, PrecisionRecallCurve

logdir = Path(
    "/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/logs_local/2025-01-31T113018_cellvit++ sam-h finetuning"
)
val_result_dir = logdir / "val_results"

# Load ground truth and predicted probabilities
gt = torch.load(val_result_dir / "gt.pt")  # Ground truth labels
probabilities = torch.load(val_result_dir / "probabilities.pt")  # Probabilities

# Initialize evaluation metrics
pr_curve_func = PrecisionRecallCurve(task="binary")
f1_score_func = F1Score(task="binary")
auroc_func = AUROC(task="binary")
accuracy_func = Accuracy(task="binary")
conf_matrix_func = ConfusionMatrix(task="binary")  # Binary confusion matrix

# Compute precision-recall curve
precision, recall, thresholds = pr_curve_func(probabilities[:, 1], gt)

# --- Find Best Recall Threshold ---
precision_constraint = 0.5  # Adjust if needed to prevent too many false positives

valid_indices = torch.where(precision > precision_constraint)[
    0
]  # Only consider reasonable precision
if len(valid_indices) > 0:
    best_recall_idx = valid_indices[torch.argmax(recall[valid_indices])]
else:
    best_recall_idx = torch.argmax(recall)  # Fallback: pure highest recall

best_recall_thresh = thresholds[best_recall_idx]

# --- Find Best F1-Score Threshold ---
f1_scores = 2 * (precision * recall) / (precision + recall)
best_f1_idx = torch.argmax(f1_scores)  # Index of threshold that maximizes F1-score
best_f1_thresh = thresholds[best_f1_idx]

# Compare performance at:
# 1) Default argmax threshold (0.5)
# 2) Best Recall threshold
# 3) Best F1-score threshold
pred_argmax = probabilities[:, 1] > 0.5
pred_recall = probabilities[:, 1] > best_recall_thresh
pred_f1 = probabilities[:, 1] > best_f1_thresh

# Compute Confusion Matrices
conf_matrix_argmax = conf_matrix_func(pred_argmax, gt)
conf_matrix_recall = conf_matrix_func(pred_recall, gt)
conf_matrix_f1 = conf_matrix_func(pred_f1, gt)

# Compute Metrics
f1_argmax = f1_score_func(pred_argmax, gt)
f1_recall = f1_score_func(pred_recall, gt)
f1_f1 = f1_score_func(pred_f1, gt)
acc_argmax = accuracy_func(pred_argmax, gt)
acc_recall = accuracy_func(pred_recall, gt)
acc_f1 = accuracy_func(pred_f1, gt)
auroc = auroc_func(probabilities[:, 1], gt)

# Print Results
print(f"AUROC: {auroc:.4f}")
print("\n--- Performance Comparison ---")
print(f"F1 @ 0.5 threshold: {f1_argmax:.4f}")
print(f"F1 @ best recall threshold: {f1_recall:.4f}")
print(f"F1 @ best F1 threshold: {f1_f1:.4f}")
print(f"Accuracy @ 0.5 threshold: {acc_argmax:.4f}")
print(f"Accuracy @ best recall threshold: {acc_recall:.4f}")
print(f"Accuracy @ best F1 threshold: {acc_f1:.4f}")

print("\n--- Threshold Selection ---")
print(f"Best Recall Threshold: {best_recall_thresh:.4f}")
print(f"Recall at Best Threshold: {recall[best_recall_idx]:.4f}")
print(f"Precision at Best Threshold: {precision[best_recall_idx]:.4f}")
print(f"Best F1 Threshold: {best_f1_thresh:.4f}")
print(f"Recall at Best F1 Threshold: {recall[best_f1_idx]:.4f}")
print(f"Precision at Best F1 Threshold: {precision[best_f1_idx]:.4f}")

# Print Confusion Matrices
print("\nConfusion Matrix @ 0.5 threshold (default):")
print(conf_matrix_argmax.int())  # Convert to integer for better readability

print("\nConfusion Matrix @ best recall threshold:")
print(conf_matrix_recall.int())  # Convert to integer for better readability

print("\nConfusion Matrix @ best F1 threshold:")
print(conf_matrix_f1.int())  # Convert to integer for better readability
