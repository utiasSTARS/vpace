#!/bin/bash

#SBATCH --account=def-schuurma
#SBATCH --gres=gpu:v100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=47000M        # memory per node
#SBATCH --time=00-12:00      # time (DD-HH:MM)
#SBATCH --output=/home/chanb/scratch/vpace/run_reports/%N-%j.out  # %N for node name, %j for jobID

# loading mujoco stuff
module load StdEnv/2020
module load python/3.9.6
module load mujoco/2.2.2
module load cuda/11.4

# hardcoded folders
VENV_FOLDER="$HOME/vpace_env"
REPO_PATH="$HOME/src/vpace/vpace"
SAVE_PATH="$HOME/scratch/vpace"
export VPACE_TOP_DIR="$REPO_PATH"

source "$VENV_FOLDER/bin/activate"

echo --expert_top_dir "${REPO_PATH}" --top_save_path "${SAVE_PATH}" "$@"
python $REPO_PATH/run_vpace.py \
  --expert_top_dir "${REPO_PATH}" \
  --top_save_path "${SAVE_PATH}" \
  "$@"