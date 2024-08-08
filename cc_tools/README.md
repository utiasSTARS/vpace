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
```bash
bash single_job.sh --main_task insert_nb_nm --exp_name debug_test --expert_amounts 100
```

If the code runs without error, we can start running slurm jobs.

## Experiments
Run `run_experiments.sh` to kick off an experiment with 5 seeds.
The script takes in four arguments:
- `main_task`: The task to run
- `task_mode`: Single task or multitask
- `exp_name`: The experiment name for logging
- `algo`: The reward model

Optionally, you may include extra arguments for the code.
For example, if we want to use value penalization with $\lambda = 1$:
```
bash run_experiments.sh stack_nm mt vp_exp sqil --q_over_max_penalty 1.0 --q_regularizer vp
```
