#!/usr/bin/env python3
import argparse
import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def xml_to_geojson(xml_path: Path, geojson_path: Path) -> None:
    """
    Convert ASAP Dot‑annotation XML into a GeoJSON array of MultiPoint Features,
    with:
      - monocytes → yellow
      - lymphocytes → red
      - other → green (under the name “OtherCells” to avoid QuPath’s default)
    """

    def hex_to_rgb(h: str) -> List[int]:
        h = h.lstrip("#")
        return [int(h[i : i + 2], 16) for i in (0, 2, 4)]

    COLOR_MAP = {
        "monocytes": "#FFFF00",
        "lymphocytes": "#FF0000",
        "other": "#00FF00",
    }
    NAME_MAP = {
        "monocytes": "Monocytes",
        "lymphocytes": "Lymphocytes",
        "other": "OtherCells",  # renamed to avoid built‑in “Other”
    }

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    group_coords: Dict[str, List[Tuple[float, float]]] = {}
    for anno in root.findall(".//Annotation"):
        if anno.get("Type") != "Dot":
            continue
        grp = anno.get("PartOfGroup", "").lower()
        coord = anno.find("Coordinates/Coordinate")
        if coord is None:
            continue
        x, y = float(coord.get("X", "0")), float(coord.get("Y", "0"))
        group_coords.setdefault(grp, []).append((x, y))

    features: List[Dict[str, Any]] = []
    for grp, coords in group_coords.items():
        hexcol = COLOR_MAP.get(grp, "#000000")
        rgb = hex_to_rgb(hexcol)
        name = NAME_MAP.get(grp, grp.title())

        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [[x, y] for x, y in coords],
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": name, "color": rgb},
                },
            }
        )

    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with geojson_path.open("w", encoding="utf-8") as f:
        # QuPath will accept a plain array of Feature objects:
        json.dump(features, f, ensure_ascii=False, indent=2)


def batch_convert_xml_folder(input_dir: Path, output_dir: Path) -> None:
    """
    Convert all .xml files in `input_dir` to .geojson in `output_dir`,
    preserving filenames. Uses tqdm for progress reporting.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_paths = list(input_dir.glob("*.xml"))
    for xml_path in tqdm(xml_paths, desc="Converting XML → GeoJSON"):
        geojson_path = output_dir / f"{xml_path.stem}.geojson"
        xml_to_geojson(xml_path, geojson_path)


def main() -> None:
    # parser = argparse.ArgumentParser(
    #     description="Convert a single ASAP Dot-annotation XML to GeoJSON for QPath"
    # )
    # parser.add_argument(
    #     "xml_file",
    #     type=Path,
    #     nargs="?",
    #     default=Path(
    #         "/work/grana_urologia/MONKEY_challenge/data/instanseg_3_classes_xml_annotations_all_wsi/A_P000001.xml"
    #     ),
    #     help="Input ASAP XML file",
    # )
    # parser.add_argument(
    #     "geojson_file",
    #     type=Path,
    #     nargs="?",
    #     default=Path("./A_P000001.geojson"),
    #     help="Output GeoJSON file",
    # )
    # args = parser.parse_args()
    # xml_to_geojson(args.xml_file, args.geojson_file)

    input_dir = Path(
        "/work/grana_urologia/MONKEY_challenge/outputs/instanseg/xml_3_classes"
    )
    output_dir = Path(
        "/work/grana_urologia/MONKEY_challenge/outputs/instanseg/geojson_3_classes"
    )

    batch_convert_xml_folder(input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
