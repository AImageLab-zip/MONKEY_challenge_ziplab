import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from tqdm import tqdm

# ---------------------------------------------------------------------
MPP = 0.24199951445730394  # µm per pixel
ROUND_NDIGITS = 4  # exactly 4 decimal places
SKIP_GROUP = "ROI"  # ROI polygons stay un‑scaled
# ---------------------------------------------------------------------


# ---------- formatting helpers --------------------------------------
def _fmt(val: float) -> str:
    """Always return a string with exactly 4 dp (xxxxx.0000)."""
    return f"{round(val, ROUND_NDIGITS):.{ROUND_NDIGITS}f}"


def _ordered_attrs(anno):
    """Return Name, Type, PartOfGroup, Color (ASAP order)."""
    return OrderedDict(
        (k, anno.get(k))
        for k in ("Name", "Type", "PartOfGroup", "Color")
        if anno.get(k) is not None
    )


def _indent(elem, level=0, space="\t"):
    """Tab + newline indentation identical to the JSON helper."""
    i = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + space
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1, space)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ---------------------------------------------------------------------


def _rebuild_groups(root, group2color):
    """Remove any existing <AnnotationGroups>; rebuild with groups seen."""
    for old in root.findall("AnnotationGroups"):
        root.remove(old)

    ag = ET.SubElement(root, "AnnotationGroups")
    for name, color in sorted(group2color.items()):
        grp = ET.SubElement(
            ag,
            "Group",
            OrderedDict([("Name", name), ("PartOfGroup", "None"), ("Color", color)]),
        )
        # add an <Attributes /> child only if it existed in source
        if name.lower() != "other":  # mimic the example you posted
            grp.append(ET.Element("Attributes"))


def convert_folder(input_dir, output_dir, mpp=MPP):
    os.makedirs(output_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]

    for fname in tqdm(xml_files, desc="Converting XML annotations"):
        in_path, out_path = map(
            lambda p: os.path.join(p, fname), (input_dir, output_dir)
        )

        tree = ET.parse(in_path)
        root = tree.getroot()
        ann_root = root.find("Annotations")

        group2color = {}  # collect true groups

        for anno in ann_root:
            grp = anno.get("PartOfGroup") or ""
            group2color.setdefault(grp, anno.get("Color", "#000000"))
            do_scale = grp.lower() != SKIP_GROUP.lower()

            # update coordinates
            for c in anno.iter("Coordinate"):
                x = float(c.get("X"))
                y = float(c.get("Y"))
                if do_scale:
                    x *= mpp / 1000
                    y *= mpp / 1000
                c.set("X", _fmt(x))
                c.set("Y", _fmt(y))

            # enforce attribute order
            anno.attrib.clear()
            anno.attrib.update(_ordered_attrs(anno))

        _rebuild_groups(root, group2color)  # only groups actually present
        _indent(root, space="\t")  # final pretty‑print

        tree.write(out_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    INPUT_DIR = "/work/grana_urologia/MONKEY_challenge/data/instanseg_3_classes_xml_annotations_all_wsi"
    OUTPUT_DIR = "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/xml_all_instanseg_3_classes"
    convert_folder(INPUT_DIR, OUTPUT_DIR)
