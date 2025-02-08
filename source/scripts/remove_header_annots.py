import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the root folder containing the CSV files
root_folder = "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train/labels_old"

# Define the output folder
output_folder = "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train/labels"
os.makedirs(output_folder, exist_ok=True)

# Define image size (change this if your images are different)
IMAGE_WIDTH = 256  # Adjust based on your dataset
IMAGE_HEIGHT = 256  # Adjust based on your dataset

# Optional: Shift annotations by 1 pixel (set to 0 to disable)
PIXEL_SHIFT_X = 1  # Shift X by 1 pixel (set to 0 for no shift)
PIXEL_SHIFT_Y = 1  # Shift Y by 1 pixel (set to 0 for no shift)


def clamp(x, y, width, height, shift_x, shift_y):
    """
    First rounds x and y, applies an optional shift, and clamps them using:
      x = max(0, min(x + shift_x, width - 1))
      y = max(0, min(y + shift_y, height - 1))
    Returns integer values.
    """
    # Round and convert to int
    x = np.round(x, 0).astype(int)
    y = np.round(y, 0).astype(int)

    # Shift and clamp
    x = max(0, min(x + shift_x, width - 1))
    y = max(0, min(y + shift_y, height - 1))

    return x, y


for filename in tqdm(os.listdir(root_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(root_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # If the file is empty, create an empty file in the output folder
        if os.stat(file_path).st_size == 0:
            open(output_path, "w").close()
            print(f"Empty file preserved: {filename}")
            continue

        try:
            # Load CSV manually (avoiding pandas) - expecting x, y, label_id format
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Process only non-empty lines
            if not lines:
                open(output_path, "w").close()
                continue

            processed_lines = []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue  # Skip malformed rows

                x, y, label_id = map(float, parts[:3])  # Convert x, y, label to float
                x, y = clamp(
                    x, y, IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL_SHIFT_X, PIXEL_SHIFT_Y
                )

                processed_lines.append(
                    f"{x},{y},{int(label_id)}\n"
                )  # Ensure all are int

            # Save the processed file
            with open(output_path, "w") as f:
                f.writelines(processed_lines)

            print(f"Processed and saved: {filename} → {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            open(output_path, "w").close()
            continue

print("✅ All CSV files processed!")
print("✅ All CSV files processed!")
