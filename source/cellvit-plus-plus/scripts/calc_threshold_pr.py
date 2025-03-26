# -*- coding: utf-8 -*-
# Find best classifier threshold using PR curve
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from pathlib import Path

import torch
from torchmetrics import AUROC, Accuracy, F1Score, PrecisionRecallCurve

logdir = Path(
    "/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/logs_local/2025-01-31T113018_cellvit++ sam-h finetuning"
)
val_result_dir = logdir / "val_results"

gt = torch.load(val_result_dir / "gt.pt")
probabilities = torch.load(val_result_dir / "probabilities.pt")

pr_curve_func = PrecisionRecallCurve(task="binary")
f1_score_func = F1Score(task="binary")
auroc_func = AUROC(task="binary")
accuracy_func = Accuracy(task="binary")

precision, recall, thresholds = pr_curve_func(probabilities[:, 1], gt)


# Find the threshold that maximizes F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)
thresh = thresholds[torch.argmax(f1_scores)]


# argmax
pred_argmax = probabilities[:, 1] > 0.5
pred_thresh = probabilities[:, 1] > thresh

f1_argmax = f1_score_func(pred_argmax, gt)
f1_thresh = f1_score_func(pred_thresh, gt)
acc_argmax = accuracy_func(pred_argmax, gt)
acc_thresh = accuracy_func(pred_thresh, gt)
auroc = auroc_func(probabilities[:, 1], gt)

print(f"ROC PR: {auroc}")
print(f"F1 argmax: {f1_argmax}")
print(f"F1 thresh: {f1_thresh}")
print(f"Acc argmax: {acc_argmax}")
print(f"Acc thresh: {acc_thresh}")
print(f"Thresholds: {thresh}")
