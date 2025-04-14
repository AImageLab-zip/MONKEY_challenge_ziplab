"""
It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path
from glob import glob
import os
import json
from tqdm import tqdm


from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
Model_PATH = Path("/opt/ml/model")


from wsdetectron2 import Detectron2DetectionPredictor
from structures import Point





def run():
    # Read the input

    image_paths = glob(os.path.join(INPUT_PATH,"images/kidney-transplant-biopsy-wsi-pas/*.tif"))
    mask_paths = glob(os.path.join(INPUT_PATH,"images/tissue-mask/*.tif"))


    image_path = image_paths[0]
    mask_path = mask_paths[0]

    output_path = OUTPUT_PATH
    json_filename_lymphocytes = "detected-lymphocytes.json"
    weight_root = os.path.join(Model_PATH,"model_final.pth")

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    patch_shape=(128,128,3)
    spacings=(0.5,)
    overlap=(0,0)
    offset=(0,0)
    center=False

    patch_configuration = PatchConfiguration(patch_shape=patch_shape,
                                             spacings=spacings,
                                             overlap=overlap,
                                             offset=offset,
                                             center=center)

    model = Detectron2DetectionPredictor(
    output_dir=output_path,
    threshold= 0.1,
    nms_threshold=0.3,
    weight_root = weight_root
    )


    iterator = create_patch_iterator(image_path=image_path,
                               mask_path=mask_path,
                               patch_configuration=patch_configuration,
                               cpus=4,
                               backend='asap')


    # Save your output
    inference(
        iterator=iterator,
        predictor=model,
        spacing = spacings[0],
        image_path=image_path,
        output_path=output_path
    )

    iterator.stop()

    location_detected_lymphocytes_all = glob(os.path.join(OUTPUT_PATH, "*.json"))
    location_detected_lymphocytes = location_detected_lymphocytes_all[0]
    print(location_detected_lymphocytes_all)
    print(location_detected_lymphocytes)
    # Secondly, read the results
    result_detected_lymphocytes = load_json_file(
        location=location_detected_lymphocytes,
    )


    return 0


def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000

def to_wsd(points):
    """Convert list of coordinates into WSD points"""
    new_points = []
    for i, point in enumerate(points):
        p = Point(
            index=i,
            label=Label("lymphocyte", 1, color="blue"),
            coordinates=[point],
        )
        new_points.append(p)
    return new_points


def inference(iterator, predictor, spacing, image_path, output_path):
    print("predicting...")
    output_dict_lymphocytes = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    annotations_lymphocytes = []
    annotations_monocytes = []
    annotations_inflammatory_cells = []
    counter_lymphocytes = 0
    counter_monocytes = 0
    counter_inflammatory_cells = 0

    spacing_min = 0.25
    ratio = spacing/spacing_min
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)


    for x_batch, y_batch, info in tqdm(iterator):
        x_batch = x_batch.squeeze(0)
        y_batch = y_batch.squeeze(0)

        predictions = predictor.predict_on_batch(x_batch)
        for idx, prediction in enumerate(predictions):

            c = info['x']
            r = info['y']

            for detections in prediction:
                x, y, label, confidence = detections.values()

                if x == 128 or y == 128:
                    continue

                if y_batch[idx][y][x] == 0:
                    continue

                # counter += 1
                x = x*ratio + c # x is in spacing= 0.5 but c is in spacing = 0.25
                y= y*ratio + r
                
                # inflammatory_cells
                prediction_record_inflammatory_cells = {
                    "name" : "Point "+str(counter_inflammatory_cells),
                    "point": [
                        px_to_mm(x, spacing),
                        px_to_mm(y, spacing),
                        0.24199951445730394,
                    ],
                    "probability": confidence,
                }
                
                output_dict_inflammatory_cells["points"].append(prediction_record_inflammatory_cells)

                annotations_inflammatory_cells.append((x, y))
                counter_inflammatory_cells += 1
                
                # lymphocytes
                if label == "lymphocyte":
                    prediction_record_lymphocytes = {
                        "name" : "Point "+str(counter_lymphocytes),
                        "point": [
                            px_to_mm(x, spacing),
                            px_to_mm(y, spacing),
                            0.24199951445730394,
                        ],
                        "probability": confidence,
                    }

                    output_dict_lymphocytes["points"].append(prediction_record_lymphocytes)

                    annotations_lymphocytes.append((x, y))
                    counter_lymphocytes += 1
                    
                # monocytes
                if label == "monocyte":
                    prediction_record_monocytes = {
                        "name" : "Point "+str(counter_monocytes),
                        "point": [
                            px_to_mm(x, spacing),
                            px_to_mm(y, spacing),
                            0.24199951445730394,
                        ],
                        "probability": confidence,
                    }

                    output_dict_monocytes["points"].append(prediction_record_monocytes)

                    annotations_monocytes.append((x, y))
                    counter_monocytes += 1
                    
                    



    print(f"Predicted {len(annotations_inflammatory_cells)} inflammatory_cells points")
    print(f"Predicted {len(annotations_lymphocytes)} lymphocytes points")
    print(f"Predicted {len(annotations_monocytes)} monocytes points")
    print("saving predictions...")
    
    # ###################################
    # # XML:
    # # saving xml file (inflammatory_cells)
    # annotations_inflammatory_cells_wsd = to_wsd(annotations_inflammatory_cells)
    # xml_filename = 'points_results_inflammatory-cells.xml'
    # output_path_xml = os.path.join(output_path,xml_filename)
    # write_point_set(
    #     annotations_inflammatory_cells_wsd,
    #     output_path_xml,
    #     label_color="blue",
    # )
    
    # # saving xml file (lymphocytes)
    # annotations_lymphocytes_wsd = to_wsd(annotations_lymphocytes)
    # xml_filename = 'points_results_lymphocytes.xml'
    # output_path_xml = os.path.join(output_path,xml_filename)
    # write_point_set(
    #     annotations_lymphocytes_wsd,
    #     output_path_xml,
    #     label_color="red",
    # )
    
    # # saving xml file (monocytes)
    # annotations_monocytes_wsd = to_wsd(annotations_monocytes)
    # xml_filename = 'points_results_monocytes.xml'
    # output_path_xml = os.path.join(output_path,xml_filename)
    # write_point_set(
    #     annotations_monocytes_wsd,
    #     output_path_xml,
    #     label_color="yellow",
    # )


    ###################################
    # JSON
    # saving json file (inflammatory-cells)
    json_filename = "detected-inflammatory-cells.json"
    output_path_json = os.path.join(output_path, json_filename)
    # with open(output_path_json, "w") as outfile:
    #     json.dump(output_dict_inflammatory_cells, outfile, indent=4)
    write_json_file(
        location=output_path_json,
        content=output_dict_inflammatory_cells
    )
    
    # saving json file (lymphocytes)
    json_filename = "detected-lymphocytes.json"
    output_path_json = os.path.join(output_path, json_filename)
    # with open(output_path_json, "w") as outfile:
    #     json.dump(output_dict, outfile, indent=4)
    write_json_file(
        location=output_path_json,
        content=output_dict_lymphocytes
    )


    # saving json file (monocytes)
    json_filename = "detected-monocytes.json"
    output_path_json = os.path.join(output_path, json_filename)
    # with open(output_path_json, "w") as outfile:
    #     json.dump(output_dict_monocytes, outfile, indent=4)
    write_json_file(
        location=output_path_json,
        content=output_dict_monocytes
    )



    print("finished!")



def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))




def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())