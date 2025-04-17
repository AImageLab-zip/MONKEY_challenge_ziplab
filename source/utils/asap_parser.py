import xml.etree.ElementTree as ET

from tqdm import tqdm


def parse_asap_dot_annotations(xml_path, group_to_label=None, ignore_groups=None):
    """
    Parse ASAP XML annotations that use Type="Dot", ignoring certain groups (e.g. "ROI").
    Returns a list of (x, y, label_id) in GLOBAL coordinates.

    Args:
        xml_path (str): Path to the ASAP XML file.
        group_to_label (dict): Maps PartOfGroup strings (e.g. "lymphocytes") to integer label IDs.
                               If None, all recognized groups default to label 0.
        ignore_groups (set): A set of group names to ignore (e.g., {"ROI"}).
                             If an annotation's PartOfGroup is in this set, it is skipped.

    Returns:
        List[Tuple[float, float, int]]
    """
    if group_to_label is None:
        group_to_label = {}

    if ignore_groups is None:
        ignore_groups = set()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations_root = root.find("Annotations")
    if annotations_root is None:
        raise ValueError("XML does not contain <Annotations> section")

    dots = []
    annotations = annotations_root.findall("Annotation")
    for annotation in tqdm(annotations, desc="Parsing dot annotations"):
        anno_type = annotation.get("Type", "")
        group_name = annotation.get("PartOfGroup", "unknown")

        if anno_type.lower() != "dot" or group_name in ignore_groups:
            continue

        label_id = group_to_label.get(group_name, 0)
        coords_root = annotation.find("Coordinates")
        if coords_root is None:
            continue

        for coordinate in coords_root.findall("Coordinate"):
            x_str = coordinate.get("X")
            y_str = coordinate.get("Y")
            if x_str is None or y_str is None:
                continue

            x_val = float(x_str)
            y_val = float(y_str)
            dots.append((x_val, y_val, label_id))

    return dots
