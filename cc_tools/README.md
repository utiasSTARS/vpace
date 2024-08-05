# Running on Compute Canada
Assume that `lfgp` and `vpace` are saved under `$HOME/src/vpace`.

## Installation
```
module load StdEnv/2020
module load python/3.9.6
module load mujoco/2.2.2
module load cuda/11.4

# ========================================================================
# First time setup to install mujoco binary
mkdir ~/mujocodwn
mkdir ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/mujocodwn
tar -xvzf ~/mujocodwn/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
rm -rf ~/mujocodwn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# ========================================================================

python -m venv ~/vpace_env
source ~/vpace_env/bin/activate

pip install -e lfgp/rl_sandbox
```

### Verifying Installation
Run `salloc --time=0:20:00 --mem=3500 --gres=gpu:1 --account=def-<account>` and run the following:
```
module load StdEnv/2020
module load python/3.9.6
module load mujoco/2.2.2
module load cuda/11.4
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

python $REPO_PATH/run_vpace.py   --expert_top_dir "${DATA_FOLDER}"   --top_save_path "${SAVE_PATH}"   --exp_name vp   --q_regularizer vp   --q_over_max_penalty 1.0
```
If the code runs without error, we can start starting slurm jobs.

## Experiments
Run `run_experiments.sh` to kick off an experiment with 5 seeds.
The script takes in four arguments:
- `main_task`: The task to run
- `task_mode`: Single task or multitask
- `job_name_post`: The experiment name for logging
- `algo`: The reward model

Optionally, you may include extra arguments for the code.
For example, if we want to use value penalization with $\lambda = 1$:
```
bash run_experiments.sh stack_nm mt vp_exp sqil --exp_name vp_exp --q_over_max_penalty 1.0 --q_regularizer vp
```
