import argparse
import os
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# --- Label Map ---
label_map = {"monocytes": 0, "lymphocytes": 1, "other": 2}
label_colors = cm.get_cmap("tab10")


def main(base_path, grid_size, output_file, seed):
    random.seed(seed)

    # Derived paths
    train_path = os.path.join(base_path, "train")
    labels_path = os.path.join(train_path, "labels")
    images_path = os.path.join(train_path, "images")

    # Get list of valid label/image pairs
    valid_files = []
    for fname in os.listdir(labels_path):
        if fname.endswith(".csv"):
            csv_path = os.path.join(labels_path, fname)
            img_path = os.path.join(images_path, fname.replace(".csv", ".png"))
            if os.path.exists(img_path) and os.path.getsize(csv_path) > 0:
                valid_files.append(fname)

    if len(valid_files) < grid_size * grid_size:
        raise ValueError("Not enough valid samples for the requested grid size.")

    selected_files = random.sample(valid_files, grid_size * grid_size)

    # Create grid plot
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3)
    )
    for idx, fname in enumerate(selected_files):
        row, col = divmod(idx, grid_size)
        ax = axes[row, col]

        # Load image
        img_path = os.path.join(images_path, fname.replace(".csv", ".png"))
        image = Image.open(img_path)
        ax.imshow(image)

        # Load annotations
        csv_path = os.path.join(labels_path, fname)
        df = pd.read_csv(csv_path, header=None, names=["x", "y", "label"])

        for label_name, label_id in label_map.items():
            points = df[df["label"] == label_id]
            ax.scatter(
                points["x"],
                points["y"],
                color=label_colors(label_id % 10),
                s=10,
                label=label_name,
            )

        ax.set_title(fname.replace(".csv", ""), fontsize=8)
        ax.axis("off")

    # Add a single shared legend
    handles = [
        plt.Line2D(
            [],
            [],
            color=label_colors(label_id % 10),
            marker="o",
            linestyle="",
            label=label_name,
        )
        for label_name, label_id in label_map.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(label_map), fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Grid image saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot N x N random annotated images from a dataset"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_ihc",
        help="Dataset root path (excluding /train)",
    )
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size (NxN)")
    parser.add_argument(
        "--output_file",
        type=str,
        default="grid_output.png",
        help="Output image file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    main(args.base_path, args.grid_size, args.output_file, args.seed)
