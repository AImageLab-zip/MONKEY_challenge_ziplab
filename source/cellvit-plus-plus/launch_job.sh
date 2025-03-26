#!/bin/bash

#SBATCH --account=grana_urologia
#SBATCH --job-name=monkey_challenge_cellvit++
#SBATCH --partition=all_usr_prod 

#### Create a directory for the logs and log stdout and stderr
#SBATCH --output=./logs/%j/output_%j.txt
#SBATCH --error=./logs/%j/error_%j.txt

#SBATCH --time=00:20:00  # Set a maximum time limit (HH:MM:SS)

#SBATCH --cpus-per-task=8 # Request number of CPU cores
#SBATCH --mem-per-cpu=8G  # memory per CPU core

### total memory will be: cpus-per-task * mem-per-cpu

#SBATCH --ntasks=1 # number of tasks per node

#SBATCH --gres=gpu:1 ## number of gpus 

### if you need a specific gpu type, you can use the constraint flag between OR "|" symbols. Example:

#SBATCH --constraint="gpu_RTX5000_16G|gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"

#####"gpu_RTX5000_16G|gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"

### printing some information about the job
echo "== Starting scheduled run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
echo "========================="

### NOTE!!!!
### For optimal performance, load the env from your home directory (project local env are slower).
### We know is not optimal ... but it is what it is

echo "== Loading modules and activating env... =="

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda deactivate ##deactivate any active conda environment

## Load the necessary modules
module unload cuda
module load cuda/12.1 # Load compatible CUDA version

echo /work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/bin/ > /work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/cellvit_env/lib/python3.10/site-packages/asap.pth

## Activate your virtual environment
conda activate /work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/cellvit_env

# Export ASAP-specific environment variables
echo "== Exporting ASAP-related environment variables... =="
# Export paths for ASAP binaries and libraries
export PATH=/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/bin:$PATH
export LD_LIBRARY_PATH=/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/grana_urologia/MONKEY_challenge/asap/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# # Export ASAP-specific environment variables
# echo "== Exporting ASAP-related environment variables... =="
# # Export paths for ASAP binaries and libraries
# export PATH=/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/bin:$PATH
# export LD_LIBRARY_PATH=/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/work/grana_urologia/MONKEY_challenge/asap/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# # Print the environment variables for debugging (optional)
# echo "PATH: $PATH"
# echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# # Ensure ASAP is properly injected into Python's library path
# echo "== Injecting ASAP path into Python library... =="
# echo "/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/bin" > /work/grana_urologia/MONKEY_challenge/monkey_env/lib/python3.8/site-packages/asap.pth

echo "== Environment activated and variables exported! =="




echo "========================="

echo "== Running script =="

export SLURM_CPUS_PER_TASK=16
#export SLURM_GPUS_PER_TASK

export CUDA_VISIBLE_DEVICES=0

#Run your Python script
#  --classifier_path=/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/checkpoints/classifier/sam-h/monkey.pth
# python ./cellvit/detect_cells.py --model=/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/checkpoints/SAM/CellViT-SAM-H-x40-AMP.pth --classifier_path=/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/checkpoints/classifier/sam-h/monkey.pth --outdir=/work/grana_urologia/MONKEY_challenge/source/sota_architectures/test_data/output_monkey_clf --geojson process_wsi --wsi_path=/work/grana_urologia/MONKEY_challenge/source/sota_architectures/test_data/A_P000001_PAS_CPG.tif --wsi_properties="{\"slide_mpp\": 0.25, \"magnification\": 40}"

# python ./cellvit/detect_cells.py --model=/work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/checkpoints/SAM/CellViT-SAM-H-x40-AMP.pth --outdir=/work/grana_urologia/MONKEY_challenge/data/cell_positions_preds --binary process_dataset --filelist=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit/filelist_2.csv

#python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train_configs/SAM-H/fold_best_params_patient_split_1.yaml

#python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train_configs/SAM-H/fold_best_params_patient_split_2.yaml

#python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train_configs/SAM-H/fold_best_params_patient_split_3.yaml

#python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train_configs/SAM-H/fold_best_params_patient_split_4.yaml

#python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train_configs/SAM-H/fold_best_params_patient_split_5.yaml

# python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit/train_configs/SAM-H/fold_2.yaml

# python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit/train_configs/SAM-H/fold_3.yaml

# python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit/train_configs/SAM-H/fold_4.yaml

#python3 ./cellvit/train_cell_classifier_head.py --config=/work/grana_urologia/MONKEY_challenge/data/monkey_cellvit_3_cls_parallel/train_configs/SAM-H/fold_0.yaml


# cd /work/grana_urologia/MONKEY_challenge/source/sota_architectures/CellViT-plus-plus/cellvit/training/evaluate

# python inference_cellvit_wsi_single.py

cd /work/grana_urologia/MONKEY_challenge/docker_inference_grand_challenge

# python check_environment.py

python inference.py

echo "== Finished running script! =="

echo "== Deactivating environment... =="
conda deactivate ## Deactivate the virtual environment
module unload cuda # Unload CUDA modules

echo "========================="
echo "== Finished at $(date) =="



