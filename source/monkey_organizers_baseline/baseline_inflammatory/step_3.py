import os
import json
import torch
import creationism
from tqdm import tqdm
import glob



from wholeslidedata.interoperability.asap.annotationwriter import write_point_set
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label

from utils.wsdetectron2 import Detectron2DetectionPredictor
from utils.structures import Point

def main():


    #########################################################################################
    # paths:

    cwd = os.getcwd()
    print('current directory is:', cwd)

    image_dir = r'./data/images'
    mask_dir = r'./data/ROI_masks_025'
    output_path = r"./outputs/results"
    if not(os.path.isdir(output_path)): os.mkdir (output_path)
    json_filename = "detected-lymphocytes.json"

    print(f"Pytorch GPU available: {torch.cuda.is_available()}")
    print(image_dir, mask_dir)

    #########################################################################################
    # defining patch configuration for each image:

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

    #########################################################################################
    # loading saved model:

    model = Detectron2DetectionPredictor(
        output_dir=output_path,
        threshold=0.1,
        nms_threshold=0.3,
        weight_root = r"./outputs/model_final.pth"
    )

    #########################################################################################
    # useful functions:

    def px_to_mm(px: int, spacing: float):
        return px * spacing / 1000

    def to_wsd(points):
        """Convert list of coordinates into WSD points"""
        new_points = []
        for i, point in enumerate(points):
            p = Point(
                index=i,
                label=Label("infcell", 1, color="blue"),
                coordinates=[point],
            )
            new_points.append(p)
        return new_points

    def write_json_file(*, location, content):
        # Writes a json file
        with open(location, 'w') as f:
            f.write(json.dumps(content, indent=4))

    #########################################################################################
    # inference each image with loaded model:

    def inference(iterator, predictor, spacing, image_path, output_path, json_filename):
        print("predicting...")
        output_dict = {
            "name": "lymphocytes",
            "type": "Multiple points",
            "version": {"major": 1, "minor": 0},
            "points": [],
        }

        annotations = []
        counter = 0

        spacing_min = 0.25
        ratio = spacing / spacing_min
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

                    x = x * ratio + c  # x is in spacing= 0.5 but c is in spacing = 0.25
                    y = y * ratio + r
                    prediction_record = {
                        "name": "Point " + str(counter),
                        "point": [
                            px_to_mm(x, spacing),
                            px_to_mm(y, spacing),
                            0.24199951445730394,
                        ],
                        "probability": confidence,
                    }
                    output_dict["points"].append(prediction_record)
                    annotations.append((x, y))
                    counter += 1

        print(f"Predicted {len(annotations)} points")
        print("saving predictions...")

        patient_name = os.path.splitext(os.path.basename(image_path))[0]

        # saving xml file
        annotations_wsd = to_wsd(annotations)
        #     xml_filename = 'points_results.xml'
        xml_filename = patient_name + '_points_results.xml'
        output_path_xml = os.path.join(output_path, xml_filename)
        write_point_set(
            annotations_wsd,
            output_path_xml,
            label_color="blue",
        )

        # saving json file
        #     output_path_json = os.path.join(output_path, json_filename)
        output_path_json = os.path.join(output_path, patient_name + json_filename)
        write_json_file(
            location=output_path_json,
            content=output_dict
        )

        print("finished!   " , patient_name)

    #########################################################################################
    # inference loop:

    image_list = glob.glob(os.path.join(image_dir,'*.tif'))
    for image_path in image_list:
        mask_path = os.path.join(mask_dir, os.path.splitext(os.path.basename(image_path))[0] + '_mask' + os.path.splitext(os.path.basename(image_path))[1])
        print(image_path,mask_path)

        iterator = create_patch_iterator(image_path=image_path,
                                       mask_path=mask_path,
                                       patch_configuration=patch_configuration,
                                       cpus=4,
                                       backend='asap')

        inference(
            iterator=iterator,
            predictor=model,
            spacing = spacings[0],
            image_path=image_path,
            output_path=output_path,
            json_filename=json_filename
        )

        iterator.stop()

    #########################################################################################

if __name__ == "__main__":
    main()