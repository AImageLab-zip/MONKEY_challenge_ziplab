
from pathlib import Path
from glob import glob
import os
import json
from tqdm import tqdm

from instanseg.utils.utils import show_images, _move_channel_axis
from skimage.measure import label as sklabel
from instanseg import InstanSeg
from instanseg.utils.pytorch_utils import torch_fastremap, centroids_from_lab
from instanseg.utils.pytorch_utils import get_masked_patches
from instanseg.instanseg import _to_tensor_float32, _rescale_to_pixel_size
import numpy as np
import torch
from typing import List
import ttach as tta

for_submission = True
use_tta = True


normalise_HE_bool = False

final_submission_1 = ["1937330.pt","1950672.pt","1949389_2.pt","1949717.pt"] #this is the public leaderboard #2 solution
model_names = ["1952372.pt","1950672.pt","1949389_2.pt"] ##this is the public leaderboard #1 solution

instanseg_name = "instanseg_brightfield_monkey.pt"


destination_pixel_size =  0.5
patch_size = 128
rescale_output = False if destination_pixel_size == 0.5 else True

if for_submission:
    INPUT_PATH = Path("/input") #remove test fro docker
    OUTPUT_PATH = Path("/output")
    MODEL_PATH = Path("models")


def normalise_HE(x):
    import torch
    import torchstain
    from instanseg.utils.utils import _move_channel_axis
    device = x.device
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.maxCRef = normalizer.maxCRef.to(device)
    normalizer.HERef = normalizer.HERef.to(device)
    norm = normalizer.normalize(I=x, stains=False, Io = 240, beta = 0.01)
    norm = torch.clamp(norm[0], 0, 255)
    norm = _move_channel_axis(norm)
    return norm
 

def scatter_plot(coords, y_hat, image):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # plt.figure(figsize=(10,10))
    # plt.imshow(_move_channel_axis(image, to_back=True))
    # plt.show()

    plt.figure(figsize=(10,10))
    plt.imshow(_move_channel_axis(image, to_back=True))
    plt.scatter(coords[y_hat == 0,1], coords[y_hat == 0,0], marker="x", color = "blue")
    plt.scatter(coords[y_hat == 1,1], coords[y_hat == 1,0], marker="x", color = "green")
    plt.scatter(coords[y_hat == 2,1], coords[y_hat == 2,0], marker="x", color = "red")

    # Create custom legend
    legend_labels = {0: "Lymphocytes", 1: "Monocytes", 2: "Other"}
    colors = ["blue", "green", "red"]  # Adjust based on your "jet" colormap

    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in legend_labels]
    plt.legend(handles=patches, loc="upper right")

    plt.show()



class ModelEnsemble(torch.nn.Module):
    def __init__(self, model_paths: List[str], device: str, use_tta: bool = False):
        super().__init__()
        self.models = torch.nn.ModuleList([
            self.load_model(model_path, device, use_tta) for model_path in model_paths
        ])
        self.device = device
    
    def load_model(self, model_path: str, device: str, use_tta: bool):
        model = torch.jit.load(model_path).eval().to(device)
        if use_tta:
            transforms = tta.Compose([
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0,90, 180, 270]),  
            ])
            model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
        return model
    
    def forward(self, x):
        with torch.no_grad():
            predictions = [model(x) for model in self.models]
            ensemble_prediction = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_prediction


def run():
    # Read the input
    print("starting up")
    device = "cpu"
    classification_device = "cuda"
    
    instanseg_script = torch.jit.load(os.path.join(MODEL_PATH,instanseg_name)).to("cuda")
    brightfield_nuclei = InstanSeg(instanseg_script, verbosity = 0)

    classifier = ModelEnsemble(
        model_paths=[os.path.join(MODEL_PATH, name) for name in model_names],
        device=classification_device,
        use_tta=use_tta
    )

    image_paths = sorted(glob(os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif")))

    mask_paths = sorted(glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif")))

    print(image_paths)
    print(mask_paths)
    records = []

    for image_path,mask_path in tqdm(zip(image_paths,mask_paths), total=len(image_paths), desc="Processing images"):
      #  print(image_path,mask_path)

        if not for_submission:
            output_path = OUTPUT_PATH / str(Path(image_path).name.replace("_PAS_CPG.tif", "") + "/output")
            output_path.mkdir(parents=True, exist_ok=True)
           # print(output_path)

            name = str(Path(image_path).name.replace("_PAS_CPG.tif", ""))

            records.append(create_record(
                pk=name,
                image_name=f"{name}_PAS_CPG.tif",
                inputs_relative_path="images/kidney-transplant-biopsy-wsi-pas",
                outputs_relative_paths={
                    "detected-lymphocytes": "detected-lymphocytes.json",
                    "detected-monocytes": "detected-monocytes.json",
                    "detected-inflammatory-cells": "detected-inflammatory-cells.json"
                }))
   
        else:
            output_path = OUTPUT_PATH

        # image_path = image_paths[0]
        # mask_path = mask_paths[0]
       # output_path = OUTPUT_PATH
        
        img_pascpg_path = image_path
        mask_path = mask_path

        from tiffslide import TiffSlide
        slidepascpg = TiffSlide(img_pascpg_path)
        slidemask = TiffSlide(mask_path)

        mask_thumbnail = slidemask.read_region((0, 0), 5, size = (10000,10000), as_array=True, padding=False)
        img_thumbnail = slidepascpg.read_region((0, 0), 5, size = (10000,10000), as_array=True, padding=False)
        factor = slidemask.level_downsamples[5]

        mask_labels = sklabel(mask_thumbnail[:,:,0] > 0)

        all_coords = []
        all_classes = []
        all_confidences = []

        for i in range(1,mask_labels.max() + 1):
            mask = mask_labels == i
            bbox = np.argwhere(mask > 0)
            bbox = bbox.min(0) ,bbox.max(0) 
            mask_thumbnail_ = mask[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1]]
            bbox = bbox[0] * factor, bbox[1] * factor
            mask_full_res = slidemask.read_region((bbox[0][1], bbox[0][0]), 0, (bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]), as_array=True)
            image = slidepascpg.read_region((bbox[0][1], bbox[0][0]), 0, (bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]), as_array=True)

            labels , input_tensor = brightfield_nuclei.eval_medium_image(image,pixel_size = 0.24199951445730394, rescale_output = rescale_output, seed_threshold = 0.1, tile_size= 1024)
            mask = _rescale_to_pixel_size(_to_tensor_float32(mask_full_res), 0.24199951445730394, destination_pixel_size).to(device)

            tensor = _rescale_to_pixel_size(_to_tensor_float32(image), 0.24199951445730394, destination_pixel_size).to(device)
            labels = labels.to(device) * torch.tensor(mask).bool()

            if normalise_HE_bool:
                tensor = normalise_HE(tensor)
        #  show_images(tensor.byte(),labels,mask,labels=[1])

            assert labels.shape[-2:] == tensor.shape[-2:]

            labels = torch_fastremap(labels)
            crops,masks = get_masked_patches(labels,tensor.to(device), patch_size=patch_size)
            x =(torch.cat((crops / 255.0,masks),dim= 1))


            with torch.amp.autocast("cuda"):

                with torch.no_grad():
                    batch_size = 128
                    y_hat = torch.cat([classifier.forward(x[i:i+batch_size].float().to(classification_device)) for i in range(0,len(x),batch_size)],dim = 0)
                    y_hat = y_hat[:,-3:] #because of dual training
                    y_hat = y_hat.cpu()

            assert y_hat.isnan().sum() == 0

          #  scatter_plot(centroids_from_lab(labels)[0].cpu().numpy(), y_hat.argmax(1), tensor.int().cpu().numpy())
            
            old = False
            if old:
                conf = y_hat.softmax(1).max(1)[0]
                y_hat = y_hat.argmax(1)
              #  x_lymphocytes = x[y_hat != 2]
                y_lymphocytes = y_hat[y_hat != 2].cpu().numpy()
                coords = centroids_from_lab(labels)[0]
                coords_lymphocytes = coords[y_hat != 2].cpu().numpy()[:,::-1] * ( destination_pixel_size / 0.24199951445730394 ) + bbox[0][::-1]
                confidence_lymphocytes = conf[y_hat != 2].cpu().numpy() 
                all_coords.extend(coords_lymphocytes)
                all_classes.extend(y_lymphocytes)
                all_confidences.extend(confidence_lymphocytes)
            else:
                conf = y_hat.softmax(1)
                y_hat = y_hat.argmax(1)
                y_lymphocytes = y_hat.cpu().numpy()

                coords = centroids_from_lab(labels)[0]
                coords_lymphocytes = coords.cpu().numpy()[:,::-1] * ( destination_pixel_size / 0.24199951445730394 ) + bbox[0][::-1]
                confidence_lymphocytes = conf.cpu().numpy() 
                all_coords.extend(coords_lymphocytes)
                all_classes.extend(y_lymphocytes)
                all_confidences.extend(confidence_lymphocytes)

        all_coords = np.array(all_coords)
        all_classes = np.array(all_classes)
        all_confidences = np.array(all_confidences)
            
        output_dict, output_dict_monocytes, output_dict_inflammatory_cells = get_dicts(all_coords,all_classes,all_confidences)

        # saving json file
        json_filename_lymphocytes = "detected-lymphocytes.json"
        output_path_json = os.path.join(output_path, json_filename_lymphocytes)
        write_json_file(
            location=output_path_json,
            content=output_dict
        )

        json_filename_monocytes = "detected-monocytes.json"
        # it should be replaced with correct json files
        output_path_json = os.path.join(output_path, json_filename_monocytes)
        write_json_file(
            location=output_path_json,
            content=output_dict_monocytes
        )

        json_filename_inflammatory_cells = "detected-inflammatory-cells.json"
        # it should be replaced with correct json files
        output_path_json = os.path.join(output_path, json_filename_inflammatory_cells)
        write_json_file(
            location=output_path_json,
            content=output_dict_inflammatory_cells
        )

        if for_submission:
            break

    print("Done")

    if not for_submission:
        file_path = "/home/cdt/Documents/Projects/monkey-challenge-instanseg/evaluation/test/predictions.json"
        write_json_file_2(file_path, records)
        print(f"JSON file written to {file_path}")

    return 0


def get_dicts(coords_lymphocytes,y_lymphocytes,conf):
 
    output_dict = {
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
    counter = 0
 
    for cc,class_,confidence in zip(coords_lymphocytes,y_lymphocytes, conf):
        
        x,y = cc
 
        x = x * 0.24199951445730394 / 1000
        y = y * 0.24199951445730394 / 1000
 
        prediction_record_inflammatory = {
            "name": "Point " + str(counter),
            "point": [
                x,
                y,
                0.24199951445730394,
            ],
            "probability": sum(confidence[:2]),
        }
 
        prediction_record_monocyte = {
            "name": "Point " + str(counter),
            "point": [
                x,
                y,
                0.24199951445730394,
            ],
            "probability": confidence[1].item(),
        }
 
        prediction_record_lymphocyte = {
            "name": "Point " + str(counter),
            "point": [
                x,
                y,
                0.24199951445730394,
            ],
            "probability": confidence[0].item(),
        }
 
        output_dict_inflammatory_cells["points"].append(
        prediction_record_inflammatory)  # should be replaced with detected inflammatory_cells
 
        output_dict["points"].append(prediction_record_lymphocyte)
 
        output_dict_monocytes["points"].append(prediction_record_monocyte)  # should be replaced with detected monocytes
 
 
        counter +=1
 
    return output_dict, output_dict_monocytes, output_dict_inflammatory_cells



def write_json_file(*, location, content):
    # Writes a json file
    print(f"Writing to {os.path.abspath(location)}")
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


        import json

def write_json_file_2(file_path, data):
    """
    Writes the provided data to a JSON file.

    Args:
        file_path (str): The path of the JSON file to write to.
        data (list): A list of dictionaries representing the JSON data.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def create_record(pk, image_name, inputs_relative_path, outputs_relative_paths):
    """
    Creates a record with the specified details.

    Args:
        pk (str): Primary key for the record.
        image_name (str): The image name.
        inputs_relative_path (str): Relative path for the inputs.
        outputs_relative_paths (dict): Dictionary with output types and their paths.

    Returns:
        dict: A single record formatted as per the JSON structure.
    """
    return {
        "pk": pk,
        "inputs": [
            {
                "image": {
                    "name": image_name
                },
                "interface": {
                    "slug": "kidney-transplant-biopsy",
                    "kind": "Image",
                    "super_kind": "Image",
                    "relative_path": inputs_relative_path
                }
            }
        ],
        "outputs": [
            {
                "interface": {
                    "slug": output_slug,
                    "kind": "Multiple points",
                    "super_kind": "File",
                    "relative_path": output_path
                }
            }
            for output_slug, output_path in outputs_relative_paths.items()
        ]
    }




if __name__ == "__main__":
    raise SystemExit(run())
