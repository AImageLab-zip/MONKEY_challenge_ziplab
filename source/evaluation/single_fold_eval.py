#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute confusion matrix and classification report for a single fold."
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="Directory of a specific fold (e.g., fold_0)",
    )
    parser.add_argument(
        "--label_mapping",
        type=str,
        default='{"monocytes": 0, "lymphocytes": 1, "other": 2}',
        help='JSON string for label mapping, e.g., \'{"monocytes": 0, "lymphocytes": 1, "other": 2}\'',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fold_dir = Path(args.fold_dir)
    val_results_dir = fold_dir / "val_results"

    gt_file = val_results_dir / "gt.pt"
    preds_file = val_results_dir / "predictions.pt"
    probs_file = val_results_dir / "probabilities.pt"

    if not gt_file.is_file():
        print(f"Ground truth file not found: {gt_file}")
        return

    # Load ground truth
    gt = torch.load(gt_file)

    # Load predictions if available; otherwise compute from probabilities.
    if preds_file.is_file():
        preds = torch.load(preds_file)
    elif probs_file.is_file():
        probs = torch.load(probs_file)
        preds = torch.argmax(probs, dim=1)
    else:
        print(f"Neither predictions.pt nor probabilities.pt found in {val_results_dir}")
        return

    # Parse label mapping
    try:
        label_mapping = json.loads(args.label_mapping)
        # Sort mapping by numeric value.
        sorted_mapping = sorted(label_mapping.items(), key=lambda x: x[1])
        sorted_class_ids = [v for k, v in sorted_mapping]
        sorted_class_names = [k for k, v in sorted_mapping]
    except Exception as e:
        print(f"Error parsing label mapping: {e}")
        return

    # Compute confusion matrix and classification report.
    cm = confusion_matrix(
        gt.cpu().numpy(), preds.cpu().numpy(), labels=sorted_class_ids
    )
    report = classification_report(
        gt.cpu().numpy(),
        preds.cpu().numpy(),
        labels=sorted_class_ids,
        target_names=sorted_class_names,
        output_dict=True,
    )

    # Print results.
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
