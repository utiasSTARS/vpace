# Value-Penalized Auxiliary Control from Examples for Learning without Rewards or Demonstrations

This repository contains data, code, and instructions for reproducing the results from **v**alue-**p**enalized **a**uxiliary **c**ontrol from **e**xamples (VPACE).

- [Installation (Algorithm + Panda Environments)](#installation-algorithm--panda-environments)
  - [Sawyer and Adroit Hand Baselines](#sawyer-and-adroit-hand-baselines)
- [Running](#running)
  - [Quick Start](#quick-start)
  - [Task Options](#task-options)
  - [Algorithm Options](#algorithm-options)
  - [Ablation Study Options](#ablation-study-options)
- [Figures](#figures)

## Installation (Algorithm + Panda Environments)
Our method and code is built on [learning from guided play (LfGP)](https://github.com/vpace-anon/lfgp).
We recommend first setting up a virtual environment (`conda`, `virtualenv`, etc.).
We have tested python 3.7-3.11, but recommend python 3.11.
To install:

1. `git clone git@github.com:vpace-anon/vpace.git && cd vpace`
2. `pip install -r reqs/requirements.txt`
3. `export VPACE_TOP_DIR=$PWD`  (optionally set this elsewhere, but you must also move `expert_data`).

### Sawyer and Adroit Hand Baselines
The above installation only allows you to run our code in the Panda environments originally from LfGP.
To run Sawyer and Adroit Hand baselines, you must also run:

1. `pip install tf-agents==0.19.0`
2. `pip install -r reqs/sawyer_hand_requirements.txt`
   1. This will cause a pip resolver issues about `gym` and `cloudpickle`, which can be safely ignored because we barely use `tf-agents` or `tensorflow-probability`.
3. (if step above alone still results in mujoco issues) `sudo apt install gcc libosmesa6-dev libglew-dev patchelf`

Note that the Sawyer and Adroit Hand environments require an older version of mujoco_py, which can sometimes be a bit of a hassle to install and run.
If you still can't run these environments after following our instructions, try looking up your specific error, and someone else has, most likely, encountered it as well.

## Running

All experiments from our paper can be reproduced with `run_vpace.py` and various argument combinations.
Results will be stored under `top_save_path/results/long_env_name/seed/algo/exp_name/date_time`.
By default, all results will be stored under `vpace/results`, but you can change this with either either the `VPACE_TOP_DIR` environment variable, or the `--top_save_path` argument.

### Quick Start
To train a VPACE-SQIL model for `Unstack-Stack`, you can use:
```bash
python run_vpace.py --main_task unstack_nm --reward_model sqil --exp_name test
```
For `sawyer_box_close`:
```bash
python run_vpace.py --env_type sawyer --env_name sawyer_box_close --reward_model sqil --exp_name test
```
For `relocate-human-v0-najp-dp`:
```bash
python run_vpace.py --env_type hand_dapg --env_name relocate-human-v0-najp-dp --reward_model sqil --exp_name test
```

### Task Options
To choose a task to run, you can use combinations of `--env_type`, `--env_name`, and `--main_task` as follows:
|                            | Options                                                                                                                             | Description                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `--env_type`               | `manipulator_learning` (default, Panda environment), `sawyer`, `hand_dapg`                                                          | Environment type                                |
| `--env_name` (Panda)       | `PandaPlayInsertTrayXYZState` (default)                                                                                             | Environment name for all Panda tasks            |
| `--env_name` (Sawyer)      | `sawyer_drawer_open`, `sawyer_drawer_close`, `sawyer_push`, `sawyer_lift`, `sawyer_box_close`, `sawyer_bin_picking`            | Environment name for Sawyer tasks               |
| `--env_name` (Adroit Hand) | `door-human-v0`, `hammer-human-v0`, `relocate-human-v0`, `door-human-v0-dp`, `hammer-human-v0-dp`, `relocate-human-v0-najp-dp` | Environment name for Adroit Hand tasks          |
| `--main_task`              | `reach`, `lift`, `move`, `stack_nm`, `unstack_nm`, `bring_nm`,`insert_nb_nm`                                                            | Panda env task (applies to Panda env only) |


### Algorithm Options
Common options you can change to reproduce our main results are:
|                        | Options                                 | Description                                                  |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------ |
| `--reward_model`       | `discrimininator`,`sqil`,`rce`,`sparse` | Reward model                                                 |
| `--single_task`        | (Add to turn on, otherwise off)         | Run without ACE/LfGP framework                               |
| `--q_over_max_penalty` | Float (default `10.0`)                  | Strength of value penalization (\lambda from the paper). Set to `0.0` to turn VP off. |

### Ablation Study Options

To reproduce our ablation study results, you can use the following option combinations:
| Ablation                     | Options to Add                                             | Description                                                    |
| ---------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| +Full Trajectories           | `--add_default_full_traj`                                  | Add full trajectory expert data, in addition to exmaples. |
| +Full Trajectories & Actions | `--add_default_full_traj --expert_data_mode obs_act` | Same as above, but include actions as well.               |
| SAC-X (Sparse Rewards)       | `--reward_model sparse`                                    | Use true sparse rewards, instead of examples.             |
| \lambda = 1                  | `--q_over_max_penalty 1.0`                                 | Value penalization strength of 1.                              |
| \lambda = 100                | `--q_over_max_penalty 100.0`                               | Value penalization strength of 100.                            |
| 10 Examples                  | `expert_amounts 10`                                        | Use only 10 examples per task.                                 |
| 100 Examples                 | `expert_amounts 100`                                       | Use only 100 examples per task.                                |
| No Example Augmentation      | `--expert_randomize_factor 0.0`                            | Turn off example augmentation.                                 |
| 10 Examples, No Ex. Aug.     | `--expert_amounts 10 --expert_randomize_factor 0.0` | 10 examples, no ex. aug.                                       |

## Figures
To generate plots and figures, you can use the scripts in `figures` and `plotting`.
You may want to install some dependencies first with:

1. `pip install -r reqs/fig_requirements.txt`