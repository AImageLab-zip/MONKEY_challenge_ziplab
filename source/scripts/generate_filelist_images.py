import argparse
import csv
import os


def generate_csv(folder_path, output_csv="output.csv"):
    """Generates a CSV file with absolute paths to files inside the given folder and fixed parameters 0.5 and 40."""
    file_entries = []

    # Iterate over files in the given folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_entries.append([file_path, 0.25, 40])

    # Write to CSV
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "slide_mpp", "magnification"])
        writer.writerows(file_entries)

    print(f"CSV file '{output_csv}' generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV from folder contents.")
    parser.add_argument(
        "folder_path", type=str, help="Path to the folder containing files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Output CSV file name (default: output.csv)",
    )
    args = parser.parse_args()

    generate_csv(args.folder_path, args.output)
