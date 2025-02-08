import argparse
import json


def px_to_mm(px: float, spacing: float) -> float:
    """
    Convert a length in pixels to millimeters.

    Parameters:
    - px: float
        The length in pixels.
    - spacing: float
        The pixel spacing in micrometers per pixel.
    """
    return px * spacing / 1000


def convert_json_px_to_mm(input_json: str, spacing: float) -> str:
    """
    Convert all pixel coordinates in the JSON data to millimeters.

    Parameters:
    - input_json: str
        The input JSON string with pixel coordinates.
    - spacing: float
        The pixel spacing in micrometers per pixel.

    Returns:
    - str: The output JSON string with millimeter coordinates.
    """
    data = json.loads(input_json)

    # Process points
    for point in data.get("points", []):
        px_coords = point.get("point", [])
        if len(px_coords) == 2:
            # Convert pixel coordinates to mm
            point["point"] = [px_to_mm(coord, spacing) for coord in px_coords]

    return json.dumps(data, indent=4)


# Example usage
if __name__ == "__main__":
    SPACING_CONST = 0.24199951445730394  # maximum micro-meter per pixel spacing (resolution) of the whole slide images

    # argparse for json input path and output path
    # parser = argparse.ArgumentParser(
    #     description="Convert JSON pixel coordinates to millimeters."
    # )
    # parser.add_argument(
    #     "input_json", type=str, help="Input JSON file with pixel coordinates."
    # )
    # parser.add_argument(
    #     "output_json", type=str, help="Output JSON file with millimeter coordinates."
    # )
    # args = parser.parse_args()

    # input_json_path = args.input_json
    # output_json_path = args.output_json

    input_json_path = "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/json_pixel/A_P000003_monocytes.json"
    output_json_path = "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/json_mm/A_P000003_monocytes.json"

    # Load JSON file
    with open(input_json_path, "r") as f:
        input_json = f.read()

    # Convert JSON
    output_json = convert_json_px_to_mm(input_json, SPACING_CONST)

    # save the output json with a custom name using the output path
    with open(output_json_path, "w") as f:
        f.write(output_json)

    print(
        f"Converted JSON px coords from file: {input_json_path} in JSON mm coords file saved to: {output_json_path}"
    )
