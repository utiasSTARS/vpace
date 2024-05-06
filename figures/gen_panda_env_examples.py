import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import argparse
import copy

from manipulator_learning.sim.envs import *
import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset

from ast import literal_eval
from functools import partial

import rl_sandbox.constants as c

from rl_sandbox.learning_utils import evaluate_policy
from rl_sandbox.utils import set_seed


# new imports..possibly delete everything above
import os
import cv2
import sys

from rl_sandbox.examples.eval_tools.utils import load_model
import common as fig_common
sys.path.append('..')
import plotting.common as plot_common
import plotting.data_locations as data_locations


parser = argparse.ArgumentParser()
parser.add_argument('--env_seed', type=int, default=0)
parser.add_argument('--model_seed', type=int, default=1)
parser.add_argument('--algo', type=str, default='multi-sqil')
# parser.add_argument('--aux_task', type=int, default=2)
# parser.add_argument('--model', type=str, default='500000.pt')
parser.add_argument('--num_ex', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--top_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'], 'results'))
parser.add_argument('--top_save_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'],
                                                                     'figures', 'reset_success_examples'))
parser.add_argument('--cam_str', type=str, default='panda_play_higher_closer')
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--vid_save_only', action='store_true')
parser.add_argument('--vid_fps', type=float, default=1.0)
parser.add_argument('--resolution', type=str, required=False)
parser.add_argument('--crf', type=int, required=False)
parser.add_argument('--render_on_screen', action='store_true')
args = parser.parse_args()


assert args.vid_fps >= 1.0, "to handle framerate less than 1, need to generate duplicate images..not handling for now"

# use the models defined in data_locations as our source for good models
# task_list = list(data_locations.main.keys())
task_list = [
    # 'open_reach',
    # 'grasp',
    # 'reach_0',
    # 'lift_0',
    # 'move_obj_0',
    # 'stack_no_move_0',
    'unstack_stack_env_only_no_move_0',
    # 'bring_no_move_0',
    # 'insert_no_bring_no_move_0',
]
env = None

for task_i, task in enumerate(task_list):

    all_reset = []
    all_success = []

    for ex_num in range(args.num_ex):

        extra_img_str = ""
        if ex_num > 0:
            env.seed(ex_num + args.env_seed)
            extra_img_str = f"_{ex_num}"

        if task in ['open_reach', 'grasp']:
            load_task = 'stack_no_move_0'
        else:
            load_task = task

        data_path = fig_common.full_path_from_alg_expname(
            args.top_dir, load_task, args.model_seed, data_locations.main[load_task][args.algo])

        config_file = 'lfgp_experiment_setting.pkl' if 'multi' in args.algo else 'dac_experiment_setting.pkl'

        # just use final saved model as model
        model_str = fig_common.PANDA_SETTINGS_DICT[load_task]['last_model']

        if task == 'open_reach':
            main_task_i = 0
        else:
            main_task_i = plot_common.PANDA_TASK_SETTINGS[load_task]['main_task_i']

        if env is None:
            config, env, buffer_preprocess, agent = load_model(
                args.env_seed, os.path.join(data_path, config_file), os.path.join(data_path, model_str),
                main_task_i, args.device, include_disc=False, force_egl=True)
            env.seed(args.env_seed)
        else:
            config, buffer_preprocess, agent = load_model(
                args.env_seed, os.path.join(data_path, config_file), os.path.join(data_path, model_str),
                main_task_i, args.device, include_disc=False, force_egl=True, include_env=False)

        # for weighted random scheduler, this sets deterministic action to run what we want
        agent.high_level_model._intention_i = np.array(main_task_i)

        obs = buffer_preprocess(env.reset())
        h_state = agent.reset()

        # get reset image -- we'll get one per task even though its the same distribution
        reset_img, _ = env.env.render(args.cam_str)  # all the same reset distribution, but we'll generate many anyways
        all_reset.append(copy.deepcopy(reset_img))
        reset_img[:,:,:3] = reset_img[:,:,:3][:,:,::-1]

        if not args.vid_save_only:
            cv2.imwrite(os.path.join(args.top_save_dir, f"{task}_reset{extra_img_str}.png"), reset_img)

        auxiliary_reward, auxiliary_success = fig_common.get_aux_reward_success(config, env)

        done = False
        ts = 0
        while not done:
            if args.render_on_screen:
                env.render()

            if task == 'grasp':
                if ts < 20:
                    agent.high_level_model._intention_i = np.array(4)  # reach
                else:
                    agent.high_level_model._intention_i = np.array(1)
                eval_task = 1
            else:
                eval_task = main_task_i

            if args.stochastic:
                action, h_state, act_info = agent.compute_action(obs=obs, hidden_state=h_state)
            else:
                action, h_state, act_info = agent.deterministic_action(obs=obs, hidden_state=h_state)

            if config[c.CLIP_ACTION]:
                action = np.clip(action, a_min=config[c.MIN_ACTION], a_max=config[c.MAX_ACTION])

            next_obs, reward, done, env_info = env.step(action)
            next_obs = buffer_preprocess(next_obs)

            curr_aux_success = auxiliary_success(observation=obs, action=action, env_info=env_info['infos'][-1])

            obs = next_obs
            ts += 1

            if done and not curr_aux_success[eval_task]:
                print(f"Task didn't successfully complete for {task}, resetting and trying again")
                obs = buffer_preprocess(env.reset())
                ts = 0
                done = False

            elif curr_aux_success[eval_task]:
                suc_img, _ = env.env.render(args.cam_str)
                all_success.append(copy.deepcopy(suc_img))
                suc_img[:,:,:3] = suc_img[:,:,:3][:,:,::-1]
                if not args.vid_save_only:
                    cv2.imwrite(os.path.join(args.top_save_dir, f"{task}_success{extra_img_str}.png"), suc_img)
                    print(f"Images for {task} saved.")
                done = True

        # for _ in range(50000):
        #     env.step(np.zeros(env.action_space.shape))
        #     env.render()

    # save vid
    vid_path = os.path.join(args.top_save_dir, "vids")
    os.makedirs(vid_path, exist_ok=True)
    for vid_suf, imgs in zip(['reset', 'success'], [all_reset, all_success]):
        fig_common.imgs_to_vid(
            vid_path=vid_path, name=f"{task}_{str(args.num_ex).zfill(2)}_eps_{vid_suf}", imgs=imgs, frame_rate=args.vid_fps,
            out_fr=30, resolution=args.resolution, crf=args.crf)
    print(f"Finished saving all eps vid for {task}")