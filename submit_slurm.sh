#!/bin/bash

#SBATCH --account=grana_urologia
#SBATCH --job-name=monkey_challenge_task
#SBATCH --partition=all_usr_prod 

#### Create a directory for the logs and log stdout and stderr
#SBATCH --output=./logs/%j/output_%j.txt
#SBATCH --error=./logs/%j/error_%j.txt

#SBATCH --time=00:10:00  # Set a maximum time limit (HH:MM:SS)

#SBATCH --cpus-per-task=4 # Request number of CPU cores
#SBATCH --mem-per-cpu=2G  # memory per CPU core

### total memory will be: cpus-per-task * mem-per-cpu

#SBATCH --ntasks=1 # number of tasks per node

#SBATCH --gres=gpu:1 ## number of gpus 

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

## load the modules you need
module unload cuda
module load cuda/11.0 #load old cuda version
# ## activate your virtual environment using it's path (example)
source /work/grana_urologia/MONKEY_challenge/env/bin/activate

echo "== Environment activated! =="

echo "== Exporting environment variables... =="
## optional: export some environment variables
##export my_var1="Hello_World!"
echo "== Done exporting environment variables! =="

echo "========================="

echo "== Running scripts =="

cd /work/grana_urologia/MONKEY_challenge/source
## run your python script (example with args)

python main.py 

echo "== Finished running script! =="

echo "== Deactivating environment... =="

deactivate ##deactivate the virtual environment
module unload cuda # unload the modules

echo "========================="
echo "== finished at $(date)"
