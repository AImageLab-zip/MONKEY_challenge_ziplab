#!/usr/bin/env python
import argparse
import json
import os
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-fold metrics and compute overall metrics."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root folder containing fold subdirectories (e.g. fold_0, fold_1, ...)",
    )
    parser.add_argument(
        "--label_mapping",
        type=str,
        default='{"monocytes": 0, "lymphocytes": 1, "other": 2}',
        help='JSON string for label mapping, e.g., \'{"monocytes": 0, "lymphocytes": 1, "other": 2}\'',
    )
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Annotate each cell with count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # If label_mapping provided, parse JSON string, otherwise use default ordering for 3 classes.
    if args.label_mapping:
        try:
            label_mapping = json.loads(args.label_mapping)
            # Sort mapping by numeric value.
            sorted_mapping = sorted(label_mapping.items(), key=lambda x: x[1])
            sorted_class_ids = [v for k, v in sorted_mapping]
            sorted_class_names = [k for k, v in sorted_mapping]
        except Exception as e:
            print(f"Error parsing label_mapping: {e}")
            return
    else:
        label_mapping = None
        sorted_class_ids = [0, 1, 2]
        sorted_class_names = [str(i) for i in sorted_class_ids]

    # Gather all fold subdirectories (e.g., fold_0, fold_1, etc.)
    fold_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")
    )

    all_fold_scores = []  # from scores.json
    per_fold_class_reports = []  # classification reports from sklearn
    per_fold_metrics = {}  # storing per-fold metrics for final JSON
    all_gt = []
    all_preds = []
    all_probs = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        val_results_dir = fold_dir / "val_results"
        scores_file = val_results_dir / "scores.json"
        gt_file = val_results_dir / "gt.pt"
        preds_file = val_results_dir / "predictions.pt"
        probs_file = val_results_dir / "probabilities.pt"

        fold_dict = {}

        # Read scores.json for the fold
        if scores_file.is_file():
            with open(scores_file, "r") as f:
                fold_scores = json.load(f)
            fold_dict["scores"] = fold_scores
            all_fold_scores.append(fold_scores)
        else:
            print(f"[WARNING] Missing scores.json in {fold_dir}")
            continue

        # Load ground truth and probabilities
        if not gt_file.is_file() or not probs_file.is_file():
            print(
                f"[WARNING] Missing gt.pt or probabilities.pt in {fold_dir}, skipping fold."
            )
            continue

        gt_fold = torch.load(gt_file)  # shape: [N]
        probs_fold = torch.load(probs_file)  # shape: [N, C] with C=3

        # If predictions file exists, load it; otherwise compute predictions from probabilities.
        if preds_file.is_file():
            preds_fold = torch.load(preds_file)
        else:
            preds_fold = torch.argmax(probs_fold, dim=1)

        # Compute per-fold confusion matrix and classification report
        # If label_mapping is provided, use sorted_class_ids; otherwise default [0,1,2]
        cm_fold = confusion_matrix(
            gt_fold, preds_fold, labels=sorted_class_ids
        ).tolist()
        report_fold = classification_report(
            gt_fold, preds_fold, labels=sorted_class_ids, output_dict=True
        )
        fold_dict["confusion_matrix"] = cm_fold
        fold_dict["classification_report"] = report_fold

        per_fold_metrics[fold_name] = fold_dict
        per_fold_class_reports.append(report_fold)

        # Accumulate for aggregated metrics
        all_gt.append(gt_fold)
        all_preds.append(preds_fold)
        all_probs.append(probs_fold)

    # Compute mean & std of scores from scores.json across folds
    mean_std_dict = {}
    if all_fold_scores:
        metric_keys = list(all_fold_scores[0].keys())
        for k in metric_keys:
            vals = [s[k] for s in all_fold_scores if k in s]
            mean_std_dict[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    else:
        print("No fold scores found. Exiting.")
        return

    # Compute macro and per-class precision/recall over folds
    macro_precision_list = []
    macro_recall_list = []
    per_class_precision = {str(i): [] for i in sorted_class_ids}
    per_class_recall = {str(i): [] for i in sorted_class_ids}

    for report in per_fold_class_reports:
        # Extract macro avg precision and recall
        macro_precision_list.append(report["macro avg"]["precision"])
        macro_recall_list.append(report["macro avg"]["recall"])
        # For each class in sorted order
        for i in sorted_class_ids:
            per_class_precision[str(i)].append(report[str(i)]["precision"])
            per_class_recall[str(i)].append(report[str(i)]["recall"])

    # Compute means and stds for macro precision/recall
    mean_std_dict["Precision/Macro/Validation"] = {
        "mean": float(np.mean(macro_precision_list)),
        "std": float(np.std(macro_precision_list)),
    }
    mean_std_dict["Recall/Macro/Validation"] = {
        "mean": float(np.mean(macro_recall_list)),
        "std": float(np.std(macro_recall_list)),
    }
    # Compute means and stds for each class precision and recall
    for i in sorted_class_ids:
        key_prec = f"Precision/Class{i}/Validation"
        key_recall = f"Recall/Class{i}/Validation"
        mean_std_dict[key_prec] = {
            "mean": float(np.mean(per_class_precision[str(i)])),
            "std": float(np.std(per_class_precision[str(i)])),
        }
        mean_std_dict[key_recall] = {
            "mean": float(np.mean(per_class_recall[str(i)])),
            "std": float(np.std(per_class_recall[str(i)])),
        }

    # Aggregate predictions across folds
    y_true_agg = torch.cat(all_gt).cpu().numpy()
    y_pred_agg = torch.cat(all_preds).cpu().numpy()
    y_prob_agg = torch.cat(all_probs).cpu().numpy()

    # Compute aggregated confusion matrix using sorted_class_ids order
    cm_agg = confusion_matrix(y_true_agg, y_pred_agg, labels=sorted_class_ids).tolist()
    report_agg = classification_report(
        y_true_agg, y_pred_agg, labels=sorted_class_ids, output_dict=True
    )

    # Count instances per class
    unique_labels, counts = np.unique(y_true_agg, return_counts=True)
    class_counts = {
        str(label): int(count) for label, count in zip(unique_labels, counts)
    }

    # Compute ROC curve for each class (one-vs-rest)
    n_classes = len(sorted_class_ids)
    y_true_onehot = np.zeros((y_true_agg.size, n_classes))
    for idx, label in enumerate(y_true_agg):
        y_true_onehot[idx, sorted_class_ids.index(label)] = 1

    roc_per_class = {}
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    for i, class_id in enumerate(sorted_class_ids):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob_agg[:, i])
        roc_per_class[str(class_id)] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auc(fpr, tpr),
        }
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        auc_dict[i] = auc(fpr, tpr)

    # Compute micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_prob_agg.ravel())
    roc_micro = {
        "fpr": fpr_micro.tolist(),
        "tpr": tpr_micro.tolist(),
        "auc": auc(fpr_micro, tpr_micro),
    }

    # (Optional) Plot and save the ROC curve
    plt.figure(figsize=(7, 6))
    colors = cycle(["red", "green", "blue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr_dict[i],
            tpr_dict[i],
            color=color,
            lw=2,
            label=f"Class {sorted_class_names[i]} (AUC = {auc_dict[i]:.2f})",
        )
    plt.plot(
        fpr_micro,
        tpr_micro,
        color="deeppink",
        lw=2,
        linestyle=":",
        label=f"Micro-average (AUC = {roc_micro['auc']:.2f})",
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (One-vs-Rest)")
    plt.legend(loc="lower right")
    roc_plot_file = results_dir / "final_roc_curve.png"
    plt.savefig(roc_plot_file, dpi=150)
    plt.close()

    # Plot and save cumulative (aggregated) confusion matrix
    cum_cm_plot_file = results_dir / "cumulative_confusion_matrix.png"
    plot_confusion_matrix(
        np.array(cm_agg),
        class_names=sorted_class_names,
        title="Cumulative Confusion Matrix",
        save_path=cum_cm_plot_file,
    )

    # Compile all metrics into a single dictionary
    final_results = {
        "folds": per_fold_metrics,
        "mean_std": mean_std_dict,
        "aggregated": {
            "confusion_matrix": cm_agg,
            "classification_report": report_agg,
        },
        "roc": {"per_class": roc_per_class, "micro": roc_micro},
        "class_counts": class_counts,
    }

    # Save all aggregated metrics to fold_results.json in the root folder
    out_json_file = results_dir / "fold_results.json"
    with open(out_json_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Saved all aggregated metrics to {out_json_file}")
    print(f"Cumulative confusion matrix plot saved to {cum_cm_plot_file}")


if __name__ == "__main__":
    main()
