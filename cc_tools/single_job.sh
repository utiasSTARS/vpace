#!/bin/bash

#SBATCH --account=def-schuurma
#SBATCH --gres=gpu:v100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=47000M        # memory per node
#SBATCH --time=00-12:00      # time (DD-HH:MM)
#SBATCH --output=/home/chanb/scratch/vpace/run_reports/%N-%j.out  # %N for node name, %j for jobID

# hardcoded folders
VENV_FOLDER="$HOME/vpace_env"
DATA_FOLDER="$SLURM_TMPDIR/data"  # for copied starting data
REPO_PATH="$HOME/src/vpace/vpace"
SRC_DATA_PATH="$REPO_PATH/expert_data"
SAVE_PATH="$HOME/scratch/vpace"
export VPACE_TOP_DIR="$REPO_PATH"

# untar data to compute node
mkdir -p "$DATA_FOLDER"
cp -R "$SRC_DATA_PATH" "$DATA_FOLDER"

# gen and/or start environment
# source gen_venv.sh cc_reqs.txt custom_reqs.txt  # source ensures that now we'll be running the new python env
source "$VENV_FOLDER/bin/activate"

# python $HOME/my_projects/lfgp-internal/scripts/lfebp/run_lfebp.py \
python $REPO_PATH/run_vpace.py \
  --expert_top_dir "${DATA_FOLDER}" \
  --top_save_path "${SAVE_PATH}" \
  "$@"