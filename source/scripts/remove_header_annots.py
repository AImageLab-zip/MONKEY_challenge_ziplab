import os

import pandas as pd
from tqdm import tqdm

# Define the root folder containing the CSV files
root_folder = "/work/grana_urologia/MONKEY_challenge/data/labels_old"

# Define the output folder
output_folder = (
    "/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit/train/labels_fixed"
)
os.makedirs(output_folder, exist_ok=True)

# Define image size (change this if your images are different)
IMAGE_WIDTH = 256  # Adjust based on your dataset
IMAGE_HEIGHT = 256  # Adjust based on your dataset

# Optional: Shift annotations by 1 pixel (set to 0 to disable)
PIXEL_SHIFT_X = 1  # Shift X by 1 pixel (set to 0 for no shift)
PIXEL_SHIFT_Y = 1  # Shift Y by 1 pixel (set to 0 for no shift)


def clamp(x, y, width, height, shift_x, shift_y):
    """
    Ensures x, y remain inside valid image bounds.
    Applies small shift (if enabled) for better alignment.
    """
    x = max(0, min(x + shift_x, width - 1))  # Shift X and clamp
    y = max(0, min(y + shift_y, height - 1))  # Shift Y and clamp
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
            # Read CSV, skipping header
            df = pd.read_csv(file_path, skiprows=1, header=None)

            # Ensure at least x, y, label_id columns
            if df.shape[1] < 3:
                print(f"Skipping {filename}: Not enough columns")
                open(output_path, "w").close()
                continue

            # Apply clamping + shifting to x and y
            df.iloc[:, 0], df.iloc[:, 1] = zip(
                *df.iloc[:, [0, 1]].apply(
                    lambda row: clamp(
                        row[0],
                        row[1],
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT,
                        PIXEL_SHIFT_X,
                        PIXEL_SHIFT_Y,
                    ),
                    axis=1,
                )
            )

            # Save the fixed CSV (without header)
            df.to_csv(output_path, index=False, header=False)
            print(f"Processed and saved: {filename} → {output_path}")

        except pd.errors.EmptyDataError:
            open(output_path, "w").close()
            print(f"Only header found; output is empty: {filename}")
            continue

print("✅ All CSV files processed!")
