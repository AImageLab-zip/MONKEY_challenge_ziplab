import json
import xml.etree.ElementTree as ET

SPACING_LEVEL0 = 0.24199951445730394

"""
# JSON file (as used by the Grand Challenge platform) to XML Format for ASAP (Automated Slide Analysis Platform) conversion.

This script reads a JSON file containing point-based annotation data, filters points by a specified probability cutoff
(optional), and converts the data into XML format compatible with ASAP.
Each point in the JSON is transformed into an XML structure, with coordinates adjusted by a predefined spatial scaling factor (`SPACING_LEVEL0`).
The XML file contains an `Annotations`  section and an `AnnotationGroups` section.

### Constants
- `SPACING_LEVEL0`: A scaling factor to adjust JSON coordinate points to the XML format requirements (um/pixel).

### Functions
- `json_to_xml(json_file, xml_file, prob_cutoff=0.0)`: Reads JSON data, filters points by `probability` (optional),
  and creates an XML structure saved to a specified file.

### Parameters
- `json_file` (str): Path to the input JSON file with point annotations.
- `xml_file` (str): Path where the output XML file should be saved.
- `prob_cutoff` (float): Probability threshold for filtering points; only points with a `probability` value less than or equal
  to this cutoff will be included in the XML. Default is `0.0`.

### JSON Structure Example
The input JSON file is expected to contain a dictionary with the following structure:
```json
{
    "name": "example_name",
    "points": [
        {
            "name": "point_name",
            "point": [x_coordinate, y_coordinate],
            "probability": probability_value
        },
        ...
    ]
}
"""


def indent(elem, level=0, space="\t"):
    """
    Adds indentation to an XML element tree.
    """
    i = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + space
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent(child, level + 1, space)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def json_to_xml(json_file, xml_file, prob_cutoff=0.0):
    # Load JSON data from file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Create the root XML element
    root = ET.Element("ASAP_Annotations")

    # Create elements based on JSON structure
    main_element = ET.SubElement(root, "Annotations")
    for point in data["points"]:
        # Debugging output
        print(f"Processing point: {point}")

        if point["probability"] >= prob_cutoff:
            print(
                f"Adding point: {point['name']} with probability {point['probability']}"
            )  # Debugging
            point_element = ET.SubElement(
                main_element,
                "Annotation",
                {
                    "Name": point["name"],
                    "PartOfGroup": f"detected-{data['name']}",
                    "Type": "Dot",
                    "Color": "#00F900",
                },
            )
            coords_element = ET.SubElement(point_element, "Coordinates")
            coord_element = ET.SubElement(
                coords_element,
                "Coordinate",
                {
                    "Order": "0",
                    "X": str(point["point"][0] * 1000 / SPACING_LEVEL0),
                    "Y": str(point["point"][1] * 1000 / SPACING_LEVEL0),
                },
            )
        else:
            print(
                f"Skipping point: {point['name']} with probability {point['probability']} > {prob_cutoff}"
            )  # Debugging

    annotation_groups = ET.SubElement(root, "AnnotationGroups")
    group = ET.SubElement(
        annotation_groups,
        "Group",
        {
            "Name": f"detected-{data['name']}",
            "PartOfGroup": "None",
            "Color": "#00F900",
        },
    )

    # Create an XML tree and save it to a file
    tree = ET.ElementTree(root)
    indent(root, space="\t")
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    json_to_xml(
        "/work/grana_urologia/MONKEY_challenge/docker_inference_grand_challenge/test_output/detected-inflammatory-cells.json",
        "./A_P000001_detected-inflammatory-cells.xml",
    )
