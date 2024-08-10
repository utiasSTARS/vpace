import rl_sandbox.constants as c
import json
import os
import yaml
import configargparse as argparse

import rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state as p_aux
from rl_sandbox.train.train_lfgp_sac import train_lfgp_sac
from rl_sandbox.train.train_dac_sac import train_dac_sac
import rl_sandbox.examples.lfgp.default_configs.eb_common as default_eb
from env_default_configs import get_common_env_defaults

parser = default_eb.get_parser()  # this grabs all the default settings, but will be overwritten by anything below

# Settings are included here if their defaults are different from the defaults set by the previous functions

# RL
parser.add_argument('--seed', type=int, default=100, help="Random seed")
parser.add_argument('--device', type=str, default="cuda:0", help="device to use")
parser.add_argument('--max_steps', type=int, required=False,
                    help="Number of steps to interact with. If not set, set automatically as default based on env config.")
parser.add_argument('--actor_lr', type=float, default=3e-4, help="Actor learning rate.")
parser.add_argument('--no_bootstrap_on_done', action="store_true", help="If set, use dones to prevent bootstrapping on timeouts.")
parser.add_argument('--no_entropy_in_qloss', action="store_true", help="If set, remove entropy from q loss.")
parser.add_argument('--buffer_warmup', type=int, default=5000, help="Buffer warmup before starting training.")
parser.add_argument('--exploration_steps', type=int, default=10000, help="Steps to use random instead of learned policy.")
parser.add_argument('--target_polyak_averaging', type=float, default=1e-3, help="Polyak averaging for updates from target.")
parser.add_argument('--eval_freq', type=int, default=25000, help="Overwrite original default eval frequency.")
parser.add_argument('--save_interval', type=int, default=50000)
parser.add_argument('--buffer_randomize_factor', type=float, default=0.0,
                        help="Factor to randomize each dimension of buffer data by, after normalizing")
parser.add_argument('--reward_scaling', type=float, default=0.1, help="Reward scaling.")

# env
parser.add_argument('--env_type', type=str, choices=['manipulator_learning', 'sawyer', 'hand_dapg', 'panda_rl_envs'],
                        default="manipulator_learning")
parser.add_argument('--env_name', type=str, default="PandaPlayInsertTrayXYZState", help="Env name.")
parser.add_argument('--main_task', type=str, default="stack", help="Main task (for play environment)")
parser.add_argument('--main_intention', type=int, default=2, help="The main intention index, only used for multitask.")
parser.add_argument('--control_hz', type=int, choices=[5, 10, 20], default=5, help="Environment control hz.")
parser.add_argument('--sawyer_grip_pos_in_env', action='store_true', help="Include grip pos in sawyer envs.")
parser.add_argument('--sawyer_vel_in_env', action='store_true', help="Include grip pos in sawyer envs.")
parser.add_argument('--sawyer_aux_tasks', type=str, choices=['reach', 'reach,grasp'], default='reach,grasp',
                    help="Sawyer auxiliary task list.")
parser.add_argument('--hand_dapg_aux_tasks', type=str, choices=['reach', 'reach,grasp'], default='reach,grasp',
                    help="hand_dapg auxiliary task list.")
parser.add_argument('--hand_dapg_dp_kwargs', type=str,
                    # default='control_hz:20,common_control_multiplier:.02,responsive_control:False,rotate_frame_ee:True,lower_mass:True,delta_pos:True,include_vel:False',
                    default='',
                    help="For overriding the defaults: e.g., 'control_hz:5,common_control_multiplier:.05'.")
parser.add_argument('--panda_rl_envs_kwargs', type=str, default='', help="For overriding default env params.")

# expert data
parser.add_argument('--expert_data_mode', type=str, default="obs_only_no_next", help="options are [obs_act, obs_only, obs_only_no_next].")
parser.add_argument('--expert_top_dir', type=str, default=os.environ['VPACE_TOP_DIR'])
parser.add_argument('--expert_dir_rest', type=str, default='expert_data/1200_per_task')
parser.add_argument('--expert_amounts', type=str, default='200',
                    help="Expert amounts per buffe for multitask, or for single task, this value can be multiplied"\
                         " by the number of tasks in the mulitask version.")
parser.add_argument('--expert_randomize_factor', type=float, default=0.1,
                        help="Factor to randomize each dimension of expert data by, after normalizing")
parser.add_argument('--single_task_multiply_amount', action='store_true',
                    help="Increase amount of data for single task to num_tasks*multitask per task amount.")
parser.add_argument('--full_traj_expert_filenames', type=str, required=False,
                    help="Expert filenames for full trajectories, to use in addition to final timesteps.")
parser.add_argument('--ft_expert_dir_rest', type=str, default='expert_data/full_trajectories/200_per_task')
parser.add_argument('--add_default_full_traj', action='store_true',
                    help="If set, add the default expert trajectories as defined in env_default_configs.py")

# data
parser.add_argument('--top_save_path', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'], 'results'),
                    help="Top path for saving results")
parser.add_argument('--exp_name', type=str, required=True, help="String corresponding to the experiment name")
parser.add_argument('--log_interval', type=int, default=1000, help="Log interval for tensorboard.")

# n step
parser.add_argument('--n_step', type=int, default=1, help="If greater than 1, add an n-step loss to the q updates.")
parser.add_argument('--n_step_mode', type=str, default="nth_q_targ",
                    help="N-step modes: options are: [n_rew_only, sum_pad, nth_q_targ].")

# lfgp/discriminator
parser.add_argument('--reward_model', type=str, choices=['discriminator', 'sqil', 'rce', 'sparse'], default="sqil")
parser.add_argument('--expbuf_critic_share_type', type=str, choices=['share', 'no_share'], default='no_share',
    help="Whether all critics learn from all expert buffers or from only their own.")
parser.add_argument('--expbuf_policy_share_type', type=str, choices=['share', 'no_share'], default='no_share',
    help="Whether all policies learn from all expert buffers or from only their own.")
parser.add_argument('--expbuf_size_type', type=str, choices=['match_bs', 'fraction'], default='fraction',
    help="Fraction means each expert buffer samples batch_size / 2 / num_tasks, match_bs means each samples batch_size."\
         " Significantly increases memory usage and processing time, so batch_size should probably be lowered.")
parser.add_argument('--expbuf_model_sample_rate', type=float, default=0.5,
    help="Proportion of mini-batch samples that should be expert samples for q/policy training.")
parser.add_argument('--expbuf_model_sample_decay', type=float, default=1.0,
    help="Decay rate for expbuf_model_sample_rate. .99999 brings close to 0 at 1M.")
parser.add_argument('--expbuf_model_train_mode', type=str, default='critic_only',
    help="Whether expert data trains the critic, or both the actor and critic. Options: [both, critic_only]")
parser.add_argument('--sqil_rce_bootstrap_expert_mode', type=str, choices=['boot', 'no_boot'], default="boot",
                    help="If boot, sqil and rce bootstrap on expert dones (unlike RCE implementation). no_boot"\
                         " means no bootstrapping on expert dones (but bootstrapping on non-expert handled by no_bootstrap_on_done)")
parser.add_argument('--q_type', type=str, default="raw", help="Options: [raw, classifier]")
parser.add_argument('--shared_layers', action="store_true", help="Switch to turn on shared layers for q/policy.")
parser.add_argument('--scheduler', type=str, choices=['wrs_plus_handcraft', 'wrs', 'learned', 'no_sched'],
                    default="wrs_plus_handcraft")

# RCE/SQIL
parser.add_argument('--sqil_policy_reward_label', type=float, choices=[0.0, -1.0], default=-1.0,
                        help="Reward label for policy data in SQIL, if not using classifier.")
parser.add_argument('--move_obj_filename', type=str, choices=['5_move.gz', '5_move_new.gz'], default='5_move_new.gz',
                    help="Name of move-object expert data file. new is a better match for 5hz env.")
parser.add_argument('--threshold_discriminator', action="store_true")
parser.add_argument('--q_regularizer', type=str, choices=['vp', 'cql', 'c2f'], default="vp")
parser.add_argument('--rnd', action="store_true", help="Enable RND")
parser.add_argument('--q_over_max_penalty', type=float, default=10.0,
                        help="If set, a multiplier on the q magnitude over the max possible q based on current expert avg/max, "\
                             "using reward_scaling and discount_factor")
parser.add_argument('--qomp_num_med_filt', type=int, default=50,
                    help="For q over max penalty + discriminator reward, how many max discrim values to use for "
                         "median filter estimate of true discrim max.")
parser.add_argument('--qomp_policy_max_type', type=str, choices=['max_exp', 'avg_exp'], default='avg_exp',
                        help="Whether to use average expert or max expert for q max penalty.")
parser.add_argument('--sawyer_orig_rce_settings', action='store_true', help="Set frame stack to 1, no grip pos in state.")

# config file
parser.add('--alg_cfg_file', required=False, is_config_file=True,
           help="Config file path for overriding many algorithm settings at once.")

args = parser.parse_args()


# check which args were actually set on the command line, because they take precedence over everything
aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
for arg, val in vars(args).items():
    if isinstance(val, bool):
        if val:
            aux_parser.add_argument('--'+arg, action='store_true')
        else:
            aux_parser.add_argument('--'+arg, action='store_false')
    else:
        aux_parser.add_argument('--'+arg, type=type(val))

cli_args, _ = aux_parser.parse_known_args()

# set to defaults if no custom file
if not args.alg_cfg_file:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(cur_dir, 'alg_cfgs', f"{args.reward_model}.yaml")
    with open(yaml_file, 'r') as f:
        def_cfg_args = yaml.safe_load(f)
    for arg, v in def_cfg_args.items():
        setattr(args, arg, v)

####### CUSTOM DEFAULTS #######
# can't handle directly in config files
if args.reward_model in [c.RCE, c.SQIL] and args.q_type == 'classifier':
    args.expert_critic_weight = 1 - args.discount_factor

args.gpu_buffer = True
args.no_shared_layers = not args.shared_layers  # since shared_layers used to be the default

# a few crucial settings different from defaults, specific to envs from RCE paper
if args.env_type in [c.SAWYER, c.HAND_DAPG]:
    # matching eval from RCE paper
    args.eval_freq = 10000
    args.num_evals_per_task = 30

    # defaults for best performance, also from RCE paper
    args.no_entropy_in_qloss = True
    args.n_step = 10

# new for this work
if args.env_type == c.SAWYER and not args.sawyer_orig_rce_settings:
    args.frame_stack = 3
    args.sawyer_grip_pos_in_env = True
if args.sawyer_orig_rce_settings:
    args.expert_randomize_factor = 0.0

##### ENV-SPECIFIC SETTINGS, DEFAULT EXPERT DATA LOCATIONS ######
get_common_env_defaults(args)

##### CFG FILE HANDLING ######
# if custom file, it should take precedence over any setting options from above, but not moreso than command line
if args.alg_cfg_file:
    with open(args.alg_cfg_file, 'r') as f:
        file_cfg_args = yaml.safe_load(f)
    for arg, v in file_cfg_args.items():
        setattr(args, arg, v)

# finally, set all args from command line
for arg, v in vars(cli_args).items():
    setattr(args, arg, v)

    if arg == 'max_steps':  # not going to ever use memory size smaller than max steps
        setattr(args, 'memory_size', v)

# append to exp_name to make sorting easier later
if args.sawyer_orig_rce_settings: args.exp_name += "_rce_orig"

assert args.memory_size == args.max_steps, \
    f"memory size set to {args.memory_size}, max steps to {args.max_steps}, all our testing was done with them equal."

# get the dictionary
experiment_setting = default_eb.get_settings(args=args)

# set expert amounts based on number of tasks -- has to be done after populating dictionary because main_task
# can be fixed by aliases
if args.single_task and experiment_setting[c.ENV_SETTING][c.ENV_TYPE] == c.MANIPULATOR_LEARNING \
        and args.single_task_multiply_amount:
    aux_reward = p_aux.PandaPlayXYZStateAuxiliaryReward(
        experiment_setting[c.ENV_SETTING][c.KWARGS][c.MAIN_TASK], include_main=False)
    num_tasks = aux_reward.num_auxiliary_rewards
    orig_amounts = experiment_setting[c.EXPERT_AMOUNT]
    print("-----------------------")
    print(f"For single task run, multiplying arg {orig_amounts} amount of data by {num_tasks} "\
          f"aux tasks, final amount {orig_amounts * num_tasks}")
    print("-----------------------")
    experiment_setting[c.EXPERT_AMOUNT] = orig_amounts * num_tasks

if args.single_task:
    train_dac_sac(experiment_config=experiment_setting)
else:
    train_lfgp_sac(experiment_config=experiment_setting)
