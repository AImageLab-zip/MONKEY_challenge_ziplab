#!/bin/bash

#SBATCH --account=grana_urologia
#SBATCH --job-name=monkey_challenge_task
#SBATCH --partition=all_usr_prod 

#### Create a directory for the logs and log stdout and stderr
#SBATCH --output=./logs/%j/output_%j.txt
#SBATCH --error=./logs/%j/error_%j.txt

#SBATCH --time=24:00:00  # Set a maximum time limit (HH:MM:SS)

#SBATCH --cpus-per-task=4 # Request number of CPU cores
#SBATCH --mem-per-cpu=4G  # memory per CPU core

### total memory will be: cpus-per-task * mem-per-cpu

#SBATCH --ntasks=1 # number of tasks per node

### no gpu for now!!!
###SBATCH --gres=gpu:1 ## number of gpus 

### if you need a specific gpu type, you can use the constraint flag between OR "|" symbols. Example:

###SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"

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
module load cuda/11.0 # Load compatible CUDA version

## Activate your virtual environment
conda activate /work/grana_urologia/MONKEY_challenge/monkey_env

# Export ASAP-specific environment variables
echo "== Exporting ASAP-related environment variables... =="
# Export paths for ASAP binaries and libraries
export PATH=/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/bin:$PATH
export LD_LIBRARY_PATH=/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/grana_urologia/MONKEY_challenge/asap/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Print the environment variables for debugging (optional)
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# # Ensure ASAP is properly injected into Python's library path
# echo "== Injecting ASAP path into Python library... =="
# echo "/work/grana_urologia/MONKEY_challenge/asap/opt/ASAP/bin" > /work/grana_urologia/MONKEY_challenge/monkey_env/lib/python3.8/site-packages/asap.pth

echo "== Environment activated and variables exported! =="

echo "========================="

echo "== Running script =="

# Navigate to your source folder
cd /work/grana_urologia/MONKEY_challenge/source

# Run your Python script
#python main.py --config=/work/grana_urologia/MONKEY_challenge/source/configs/baseline/detectron2_baseline.yml --fold=4

python main.py --config=/work/grana_urologia/MONKEY_challenge/source/configs/baseline/cellvit_dataset_creation.yml

echo "== Finished running script! =="

echo "== Deactivating environment... =="
conda deactivate ## Deactivate the virtual environment
module unload cuda # Unload CUDA modules

echo "========================="
echo "== Finished at $(date) =="
