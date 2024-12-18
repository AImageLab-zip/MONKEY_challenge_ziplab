import glob
import os

from tqdm import tqdm

from .config_parser import get_args_and_config
from .dot2polygon import dot2polygon


def create_bboxes_annots(config):
    """
    Create bounding boxes annotations from the XML annotations.

    param config: Configuration dictionary.

    return: 0 if successful, -1 if failed.

    """

    # try to get the dataset configurations from the configuration file then load them, with a default fallback
    dataset_configs = config.get("dataset", {})
    if dataset_configs is None:
        print("Dataset configurations not found in the configuration file.")
        return -1

    dataset_dir = dataset_configs.get("path", "../data/monkey-data")
    lymphocyte_half_box_size = dataset_configs.get("lymphocyte_half_box_size", 4.5)
    monocyte_half_box_size = dataset_configs.get("monocyte_half_box_size", 5.0)
    min_spacing = dataset_configs.get("min_spacing", 0.25)

    annotation_polygon_dirname = config.get(
        "annotation_polygon_dir", "annotations_polygon"
    )
    
    annotation_dir = os.path.join(dataset_dir, "annotations", "xml")

    annotation_list = glob.glob(os.path.join(annotation_dir, "*.xml"))

    annotation_polygon_dir = os.path.join(dataset_dir, annotation_polygon_dirname)
    
    if not (os.path.isdir(annotation_polygon_dir)):
        os.mkdir(annotation_polygon_dir)

    

    loading_bar = tqdm(annotation_list, desc="Creating bounding boxes annotations")

    for xml_path in loading_bar:
        output_path = os.path.join(
            annotation_polygon_dir,
            os.path.splitext(os.path.basename(xml_path))[0]
            + "_polygon"
            + os.path.splitext(os.path.basename(xml_path))[1],
        )

        dot2polygon(
            xml_path,
            lymphocyte_half_box_size,
            monocyte_half_box_size,
            min_spacing,
            output_path,
        )

        loading_bar.set_postfix_str(output_path)

    print("Bounding boxes annotations created successfully.")
    return 0


if __name__ == "__main__":
    args, config = get_args_and_config()

    print(config)
    create_bboxes_annots(config=config)
