#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.util
import os
import subprocess
import uuid
from glob import glob
from pathlib import Path

from cellvit.training.evaluate.inference_cellvit_wsi_single import (
    CellViTInfExpDetection,
    create_test_dataset,
)

monkey_ascii_art = """####################################################################################################
###########################################++............-+#########################################
####################################-............................-##################################
###############################+...                              .....+#############################
############################-......                             .........-##########################
#########################-.........                             . ..........+#######################
#######################...                                                .....#####################
#####################.....                                                .......###################
###################-.                                                          ...-#################
#################+...                                                          .....+###############
################-....                                                          ......-##############
###############.. ..     ....................         . .......................    ...-#############
##############-..        ......-++++++++++...         .......-+++++++++-.......    ....-############
#############-..  ..........-++++++-+++++++-...       .....+++++++--++++++-....    .....-###########
############+...    ......-+++-++++++++-++++...       ....++++--+++++++--+++...       ...+##########
############....    .....++++++-+--+++-++-++-..       ....++--+-++++-++-+++++-.       ....##########
###########-....    ....++++-+++-+++++.+++++.         ....+++++--++-++-++-++++-.....    ..+#########
###########.....    ...++++++++++++.++-+++++.         ....+++++-++.++-++-++++++-....    ...#########
##########+.....    ..++++--++-++-+++++-+++..         .....+++--++-+-++-+++-++++....    ...+########
##########-        ..-+++++++-+-+-++.+++++-..         . ....+++++--+-+-+++++++++-...    ...-########
##########.        ..+++++++++--++-++-.++-...         .......++-.++-++--+++++++++...    ...-########
##########..   ......+++++.----++-++-++-.--......... .......-..+++-++-++---.-++++...    ....########
##########..   ... ..++-+++++++-+------+++-.++++-......+++++++++-----+-+++++++-++...    ....########
##########..   ... ..++++++++++++++++-.........+++...+++-.......-++++++++++++++++...    ....########
##########.        ..++++--------.-++--+++.....-++..-++......++--+++-.-------++++...    ....########
##########.        ..+++++++--++-+++-++-++-.....++..+++.....+++-+-+++-++-.++++++-...    ...-########
##########-        ...+++.-++++-+++-+++-+++.....+#..++-....+++--++-+++-+++++.-++....    ...-########
##########+.....    ..-++++++-++++--++++-+++....##..+#-...++++++-+---+++--+++++.....    ...+########
###########.....    ...-+++-+++-+++++-+++-++....##-.+#-...++++++-+++++-+++--++-.....    ...#########
###########-....    ....-+++++-+++-++++++++#-...##-.+#-..-#+-++++++-+++-+++++-......    ..+#########
############....    ......++++-++-+++-+#####...-##+.##-...##++++-+++++++-+++...         ..##########
############+...    ........++++-+#+#+#####-...-##+.##-....+####+#+++-++++-....         .+##########
#############-..         .. ...+##########.....-##+.##-......+########+#..         .....-###########
##############-.           ..  ................-##+.##-... ......-+-......         ....-############
###############.         ... ..................+##-.##+..  ............. .         ...-#############
################-....              .. .........+##-.##+.....                   .. ...-##############
#################+...               .. ........+##-.##+.....                   .....################
###################-......         ....+######-###-.##+.....              .... ...-#################
#####################.....         ..#######+#####..##+.....              .......###################
#######################...         .-#-.....-.-###..##+.....              .....#####################
#########################-.........###.-#...-..-.#..##+.....         .......-#######################
############################-......+--+.......####..##+.....         ....-##########################
###############################+.......++-.-######+.###...............+#############################
####################################+..+###########.###..........+##################################
#####################################+.######....-#+.--.-++#########################################
#####################################--#####......+####+.###########################################
#####################################.###+##+.....######+.##########################################
#####################################.###-.-##++###-.####.##########################################
####################################+.###.+-+###-.+#-+##--##########################################
####################################--###.#+-####+...++.+###########################################
####################################--##+.##.+########+.############################################
####################################-+##+-###-.+####+.-#############################################
####################################-+##-+#####++--+################################################
####################################-+##-+##########################################################
####################################--##-+##########################################################
####################################+-##-+##########################################################
####################################+.##-+##########################################################
##################...###+..+######+...###-.-.+#..-###+..####+.-###..-####......###+..####-.-########
##################...+##....#####...+#.-####+.#....##+..####+..#+..+#####..++++####+..##-.-#########
#################+.-..#+.-..####+..#####-..####..+..#+..####+..-.-#######..---+######....+##########
#################-.++.-..#-.+###+..#####-.-####..+#..-..####+..-.-#######..++++#######..+###########
#################..##...+#+.-####...+##...+####..+##....####+.-#+..+#####..++++#######..+###########
################+..##+..##+.-######-....+######..+###-..####+.-###..-####......#######..+###########
####################################################################################################
####################################################################################################"""


def check_module(module_name):
    """Check if a module is installed in the environment."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"ERROR: {module_name} is not installed.")
        return False
    print(f"INFO: {module_name} is installed.")
    return True


def run_ray_test():
    """Run the ray_test.py script."""
    try:
        print("INFO: Executing 'cellvit/inference/ray_test.py'...")
        result = subprocess.run(
            ["python", "cellvit/inference/ray_test.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"SUCCESS: Script output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error while running 'ray_test.py': {e.stderr}")
        raise e
    except Exception as e:
        print(f"ERROR: Unexpected error while running 'ray_test.py': {e}")
        raise e


def check_environment():
    try:
        print("INFO: Checking installed libraries...")
        required_modules = ["torch", "torchaudio", "torchvision"]

        for module in required_modules:
            check_module(module)

        local_modules = [
            "cellvit.training.base_ml.base_cli",
            "cellvit.training.experiments.experiment_cell_classifier",
            "cellvit.inference.cli",
            "cellvit.inference.inference_memory",
        ]

        for module in local_modules:
            check_module(module)

        print("SUCCESS: Environment is correctly set up.")
    except ImportError as e:
        print(f"ERROR: {e}")
        raise e

    # GPU availability check
    try:
        import torch

        print("INFO: Checking GPU availability...")
        use_cuda = torch.cuda.is_available()

        if not use_cuda:
            raise SystemError("No CUDA-capable GPU detected.")

        print("SUCCESS: CUDA-capable GPU detected.")
        print(f"INFO: CUDNN Version: {torch.backends.cudnn.version()}")
        print(f"INFO: Number of CUDA Devices: {torch.cuda.device_count()}")
        print(f"INFO: CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(
            f"INFO: CUDA Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    except SystemError as e:
        print(f"ERROR: GPU Error: {e}")
        raise e
    except Exception as e:
        print(f"ERROR: Unexpected error during GPU check: {e}")
        raise e

    # CuPy availability and GPU access check
    try:
        print("INFO: Checking CuPy availability...")
        import cupy as cp

        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b

        if cp.allclose(c, [5, 7, 9]):
            print("SUCCESS: CuPy is functioning correctly.")
        else:
            raise RuntimeError("CuPy operation validation failed.")

        print("INFO: Checking GPU availability with CuPy...")
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise SystemError("No CUDA-capable GPU detected by CuPy.")

        print(f"SUCCESS: CuPy detected {device_count} CUDA device(s).")

        for device_id in range(device_count):
            cp.cuda.Device(device_id).use()
            props = cp.cuda.runtime.getDeviceProperties(device_id)

            print(f"INFO: Device ID: {device_id}")
            print(f"INFO:   Name: {props['name']}")
            print(f"INFO:   Total Global Memory: {props['totalGlobalMem']} bytes")
            print(f"INFO:   Multiprocessor Count: {props['multiProcessorCount']}")
            print(f"INFO:   Compute Capability: {props['major']}.{props['minor']}")
            print("")

        print("SUCCESS: CuPy is able to access the GPU and perform operations.")
    except ImportError as e:
        print(f"ERROR: CuPy Import Error: {e}")
        raise e
    except RuntimeError as e:
        print(f"ERROR: CuPy Error: {e}")
        raise e
    except SystemError as e:
        print(f"ERROR: CuPy GPU Error: {e}")
        raise e
    except Exception as e:
        print(f"ERROR: Unexpected error during CuPy check: {e}")
        raise e

    run_ray_test()

    separator = "*" * 60
    print(f"\n{separator}\nSUCCESS: Everything checked!\n{separator}")

    return 0


# Set constants for WSI info and patch extraction
MPP_LEVEL0_VALUE = 0.24199951445730394
FILTERING_THRESHOLD = (
    3.0  # threshold im micrometers to filter out eventual overlapping detections
)
PROB_THRESHOLD = 0.5  # threshold for filtering out low probability detections
INPUT_SHAPE_2D = (256, 256)
INPUT_SHAPE_3D = (256, 256, 3)
SPACINGS = (0.25,)
OVERLAP = (0, 0)
OFFSET = (0, 0)
CENTER = False


def run():
    print("Checking environment...")
    check_environment()
    print("Environment check completed.")
    print("Running script...")

    # Set GPU
    GPU = 0

    # Set CPU count
    CPUS = max(1, os.cpu_count() - 1)

    # # NOTE: The following lines are used to simulate the input and output paths in the Docker container
    INPUT_PATH = Path("test")  # Simulated /input
    OUTPUT_PATH = Path("test_output")  # Simulated /output
    MODEL_PATH = Path("example_model")  # Simulated /opt/ml/model

    # #NOTE: Uncomment the following lines and comment the above lines to use the actual paths in the Docker container
    # INPUT_PATH = Path("/input")
    # OUTPUT_PATH = Path("/output")
    # MODEL_PATH = Path("/opt/ml/model")

    # Resources folder path (included in the Docker image: includes backbones and models)
    RESOURCES_PATH = Path("resources")  # Simulated resources (internal folder)

    # Ensure directories exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    RESOURCES_PATH.mkdir(parents=True, exist_ok=True)

    # set fixed model backbones paths
    SAM_H_BACKBONE_PATH = os.path.join(
        RESOURCES_PATH, "backbones", "SAM-H", "CellViT-SAM-H-x40-AMP.pth"
    )

    VIT_256_BACKBONE_PATH = os.path.join(
        RESOURCES_PATH, "backbones", "VIT-256", "CellViT-256-x40-AMP.pth"
    )

    # set fallback model path
    FALLBACK_MODEL_CLF_PATH = os.path.join(
        RESOURCES_PATH, "models", "model_best_vit.pth"
    )

    # The external model (provided via a mounted volume) is found under MODEL_PATH.
    external_models = sorted(MODEL_PATH.glob("*.pth"))

    if not external_models:
        print(
            f"No external model checkpoint found in {MODEL_PATH}!\nDefault model will be used..."
        )
        external_model_file = "Fallback Model"
        external_model_path = FALLBACK_MODEL_CLF_PATH
    else:
        external_model_file = str(external_models[0])
        print(f"Found external model: {external_model_file}")
        external_model_path = external_model_file  # Use the full path directly

    if external_model_file == "Fallback Model":
        print(
            "No valid external model found. Using fallback classifier model with VIT-256 backbone."
        )
        fixed_backbone_path = VIT_256_BACKBONE_PATH
    else:
        if "sam" in str(external_model_file).lower():
            print("Using SAM-H backbone")
            fixed_backbone_path = SAM_H_BACKBONE_PATH
        elif "vit" in str(external_model_file).lower():
            print("Using VIT-256 backbone")
            fixed_backbone_path = VIT_256_BACKBONE_PATH
        else:
            print("Unknown model type. Defaulting to VIT-256 backbone.")
            fixed_backbone_path = VIT_256_BACKBONE_PATH
            external_model_path = FALLBACK_MODEL_CLF_PATH

    # set an id for temp files
    temp_id = str(uuid.uuid4())
    # Set a temp directory path with the unique id
    temp_dir_path = os.path.join(RESOURCES_PATH, "temp", temp_id)
    os.makedirs(temp_dir_path, exist_ok=True)
    # Set a logdir in the temp directory
    logdir = os.path.join(temp_dir_path, "logs")
    # Set a dataset path in the temp directory
    dataset_path = os.path.join(temp_dir_path, "temp_test_dataset")
    os.makedirs(dataset_path, exist_ok=True)

    # Find WSI and mask files
    wsi_files = glob(
        os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif")
    )
    mask_files = glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif"))

    if not wsi_files:
        raise FileNotFoundError(
            f"No WSI files found in {INPUT_PATH / 'images/kidney-transplant-biopsy-wsi-pas'}"
        )
    if not mask_files:
        raise FileNotFoundError(
            f"No mask files found in {INPUT_PATH / 'images/tissue-mask'}"
        )

    # Select the first available image and mask
    wsi_path = wsi_files[0]
    mask_path = mask_files[0]
    print(f"Found WSI: {wsi_path}")
    print(f"Found Mask: {mask_path}")

    print("Creating patchified WSI dataset...")
    # Create test dataset
    create_test_dataset(
        wsi_path=wsi_path,
        mask_path=mask_path,
        output_dir=dataset_path,
        patch_shape=INPUT_SHAPE_3D,
        spacings=SPACINGS,
        overlap=OVERLAP,
        offset=OFFSET,
        center=CENTER,
        cpus=CPUS,
    )

    print("Creating experiment for inference...")
    # Instantiate experiment
    experiment = CellViTInfExpDetection(
        logdir=logdir,
        cellvit_path=fixed_backbone_path,
        model_path=external_model_path,
        dataset_path=dataset_path,
        roi_mask_path=mask_path,
        roi_mask_path=mask_path,
        normalize_stains=False,
        gpu=GPU,
        input_shape=INPUT_SHAPE_2D,
        output_path=OUTPUT_PATH,
        mpp_value=MPP_LEVEL0_VALUE,
        thresh_filtering=FILTERING_THRESHOLD,
        prob_threshold=PROB_THRESHOLD,
    )

    print("Running inference...")
    # Run inference
    experiment.run_inference()

    print("✅ Inference completed. JSON files written to:", OUTPUT_PATH)
    saved_json_files = glob(os.path.join(OUTPUT_PATH, "*.json"))
    print("Saved JSON files:", saved_json_files)

    print("🧹 Cleaning up temporary files...")
    # Clean up temporary files inside the temp directory
    os.system(f"rm -rf {temp_dir_path}")
    print("🧹 Temporary files cleaned up.")

    print("All done!")
    print(monkey_ascii_art)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
