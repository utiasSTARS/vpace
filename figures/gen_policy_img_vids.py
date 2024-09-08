import argparse
import os
import sys
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import cv2
import tqdm

from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler
import rl_sandbox.constants as c
import common as fig_common
import cam_settings

sys.path.append('..')
import plotting.common as plot_common
import plotting.data_locations as data_locations
import plotting.hand_dapg_data_locations as hand_dapg_data_locations
import plotting.rce_env_data_locations as rce_env_data_locations

# for generating performance vids, e.g.
# python gen_policy_img_vids.py --task_list reach_0 --algo_list multi-sqil --num_eps 5 --vid_save_only \
#    --all_eps_one_vid --end_on_suc --vid_speedup 3.0 --resolution 300x300 --crf 28

# for generating exploratory vids, e.g.
# python gen_policy_img_vids.py --algo_list sqil-no-vp,rce,disc --num_eps 2 --vid_save_only --all_eps_one_vid \
#   --vid_speedup 2.0 --task_list insert_no_bring_no_move_0,unstack_stack_env_only_no_move_0 --stochastic \
#   --model_list 100000,200000,300000 --resolution 300x300 --crf 28 --aux_string_cap

parser = argparse.ArgumentParser()
parser.add_argument('--env_seed', type=int, default=0)
parser.add_argument('--model_seed', type=int, default=1)
parser.add_argument('--task_list', type=str, required=False)
parser.add_argument('--algo_list', type=str, default='multi-sqil,multi-sqil-no-vp,sqil,sqil-no-vp,disc,rce')
parser.add_argument('--model_list', type=str, default='last')
parser.add_argument('--aux_task', type=int, required=False)
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--top_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'], 'results'))
parser.add_argument('--top_dir', type=str, default="/media/ssd_2tb/data/lfebp/results")
parser.add_argument('--top_save_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'],
                                                                     'figures', 'policy_img_vids'))
parser.add_argument('--panda_cam_str', type=str, default='panda_play_higher_closer')
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--extra_name', type=str, default='')
parser.add_argument('--num_eps', type=int, default=1)
parser.add_argument('--img_w', type=int, default=500)
parser.add_argument('--img_h', type=int, default=500)
parser.add_argument('--vid_save_only', action='store_true')
parser.add_argument('--img_save_only', action='store_true')
parser.add_argument('--all_eps_one_vid', action='store_true')
parser.add_argument('--img_type', type=str, choices=['jpeg', 'png'], default='jpeg')
parser.add_argument('--vid_speedup', type=float, default=1.0)
parser.add_argument('--end_on_suc', action='store_true')
parser.add_argument('--resolution', type=str, required=False)
parser.add_argument('--crf', type=int, required=False)
parser.add_argument('--aux_string_cap', action='store_true')
parser.add_argument('--sawyer_hand_sched_type', type=str, choices=['true', 'forced'], default='forced')
args = parser.parse_args()

tasks = [
    *list(data_locations.main.keys()),
    *list(hand_dapg_data_locations.main_performance.keys()),
    *list(rce_env_data_locations.main_performance.keys())
]

if args.task_list:
    tasks = args.task_list.split(',')

algo_list = args.algo_list.split(',')
if 'disc' in algo_list:
    tasks.remove('relocate-human-v0')

model_list = args.model_list.split(',')
true_sawyer_hand_fr = 20
true_panda_fr = 20  # not 5 because we render the substeps
sawyer_hand_suc_hold_ts = 20
sto_str = "_sto" if args.stochastic else ""
sawyer_aux_task_dict = {0: "Main", 1: "Reach", 2: "Grasp"}
panda_aux_task_dict = {0: "Release", 1: "Grasp", 2: "Main", 3: "Lift", 4: "Reach"}  # only doing the hard tasks
forced_sawyer_sched = {0: 1, 30: 2, 60: 0}
forced_hand_sched = {0: 1, 40: 2, 80: 0}


for task_i, task in enumerate(tasks):
    env = None
    task_settings_dict, data_loc_dict, main_task_i = fig_common.get_task_settings(task)
    if args.aux_task:
        main_task_i = args.aux_task

    if 'sawyer' in task or 'human' in task:
        frame_rate = round(true_sawyer_hand_fr * args.vid_speedup)
        aux_task_dict = sawyer_aux_task_dict
    else:
        frame_rate = round(true_panda_fr * args.vid_speedup)
        aux_task_dict = panda_aux_task_dict
    speedup_str = str(args.vid_speedup).replace('.','-')

    for model in model_list:
        model_save_str = "" if model == 'last' else f"_{model}steps"

        for algo_i, algo in enumerate(algo_list):
            # optionally take different model seed as default
            if args.model_seed == 1 and algo == 'multi-sqil' and 'multi-sqil-model' in task_settings_dict:
                model_seed = task_settings_dict['multi-sqil-model']
            else:
                model_seed = args.model_seed

            # load agent
            config_file = 'lfgp_experiment_setting.pkl' if 'multi' in algo else 'dac_experiment_setting.pkl'
            model_str = task_settings_dict['last_model'] if model == 'last' else f"{model}.pt"
            data_path = fig_common.full_path_from_alg_expname(args.top_dir, task, model_seed, data_loc_dict[task][algo])
            config, env, buffer_preprocess, agent = fig_common.load_model_and_env_once(
                args.env_seed, os.path.join(data_path, config_file), os.path.join(data_path, model_str),
                main_task_i, args.device, include_disc=False, force_egl=True, env=env
            )
            if 'multi' in algo and not args.stochastic:
                agent.high_level_model._intention_i = np.array(main_task_i)
            else:
                main_task_i = 0

            if 'sawyer' in task and args.sawyer_hand_sched_type == 'forced' and args.stochastic:
                agent.high_level_model = FixedScheduler(forced_sawyer_sched[0], num_tasks=3, max_schedule=200)
                forced_sched = forced_sawyer_sched
            if 'human' in task and args.sawyer_hand_sched_type == 'forced' and args.stochastic:
                agent.high_level_model = FixedScheduler(forced_hand_sched[0], num_tasks=3, max_schedule=200)
                forced_sched = forced_hand_sched

            print(f"loaded agent for task {task}, algo {algo}, seed {args.model_seed}")

            # set camera for mujoco envs
            if 'sawyer' in task or 'human' in task:
                rgb_viewer = env.unwrapped._get_viewer('rgb_array')
                cam_settings.set_cam_settings(rgb_viewer, cam_settings.CAM_SETTINGS[task])
            else:
                # allow rendering substeps in slightly hacky way
                env.env.gripper._internal_substep_render_func = partial(
                    env.env.render, mode=args.panda_cam_str, substep_render=True)

            # success detection
            if 'sawyer' in task or 'human' in task:
                ret_suc_thresh = task_settings_dict['ret_suc']
            else:
                auxiliary_reward, auxiliary_success = fig_common.get_aux_reward_success(config, env)

            # run ep
            all_ep_imgs = []
            for ep_i in range(args.num_eps):
                obs = buffer_preprocess(env.reset())
                h_state = agent.reset()
                done = False
                ts = 0
                ep_obs = []
                ep_acts = []
                ep_imgs = []
                suc_ts = 0
                suc_latch = False
                ret = 0

                while not done:
                    if 'sawyer' in task or 'human' in task:
                        img = env.unwrapped.render('rgb_array', width=args.img_w, height=args.img_h)
                        imgs = [img]
                    else:
                        if ts == 0:
                            img, _ = env.env.render(args.panda_cam_str)
                            imgs = [img]
                        else:
                            imgs = env.env.gripper._rendered_substeps  # see manipulator_wrapper.py

                    if ('sawyer' in task or 'human' in task) and args.sawyer_hand_sched_type == 'forced' and args.stochastic:
                        if ts in forced_sched:
                            agent.high_level_model._intention_i = np.array(forced_sched[ts])

                    if args.stochastic:
                        action, h_state, act_info = agent.compute_action(obs=obs, hidden_state=h_state)
                    else:
                        action, h_state, act_info = agent.deterministic_action(obs=obs, hidden_state=h_state)
                    if config[c.CLIP_ACTION]:
                        action = np.clip(action, a_min=config[c.MIN_ACTION], a_max=config[c.MAX_ACTION])

                    if args.stochastic and 'multi' in algo:
                        print(f"High level action: {agent.curr_high_level_act}")

                    if args.aux_string_cap:
                        capped_imgs = []
                        for im in imgs:
                            if 'multi' in algo:
                                capped_imgs.append(fig_common.img_caption(
                                    im, aux_task_dict[int(agent.curr_high_level_act)]))
                            else:
                                # capped_imgs.append(fig_common.img_caption(im, "Main"))
                                capped_imgs.append(im)  # no cap for single task
                        imgs = capped_imgs

                    ep_imgs.extend(imgs)

                    ep_obs.append(obs)
                    ep_acts.append(action)

                    if 'sawyer' in task or 'human' in task:
                        next_obs, reward, done, env_info = env.step(action)
                    else:
                        next_obs, reward, done, env_info = env.step(action, substep_render_delay=5)
                    next_obs = buffer_preprocess(next_obs)
                    ret += reward

                    # detect success
                    if args.end_on_suc:
                        if 'sawyer' in task or 'human' in task:
                            # print(f'ret: {ret}, thresh: {ret_suc_thresh}')
                            if ret > ret_suc_thresh:
                                suc_latch = True
                            if suc_latch:
                                suc_ts += 1
                            if suc_ts >= sawyer_hand_suc_hold_ts:
                                done = True
                        else:
                            curr_aux_success = auxiliary_success(observation=obs, action=action, env_info=env_info['infos'][-1])
                            if curr_aux_success[main_task_i]:
                                done = True

                        if 'timeout' in task_settings_dict:
                            if ts >= task_settings_dict['timeout']:
                                done = True

                    obs = next_obs
                    ts += 1

                ep_obs = torch.tensor(np.array(ep_obs), dtype=torch.float32).to(args.device)
                ep_acts = torch.tensor(np.array(ep_acts), dtype=torch.float32).to(args.device)

                # save images
                if not args.vid_save_only:
                    img_path = os.path.join(args.top_save_dir, task, algo, str(ep_i))
                    os.makedirs(img_path, exist_ok=True)
                    for img_i in tqdm.trange(len(ep_imgs)):
                        img = ep_imgs[img_i]
                        if 'sawyer' in task or 'human' in task:
                            img = img[:, :, ::-1]
                        else:
                            img[:,:,:3] = img[:,:,:3][:,:,::-1]
                        cv2.imwrite(os.path.join(img_path, f"{str(img_i).zfill(3)}.{args.img_type}"), img)
                    print(f"Finished saving imgs for ep {ep_i}")

                if not args.img_save_only:
                    vid_path = os.path.join(args.top_save_dir, "vids", task, algo)
                    os.makedirs(vid_path, exist_ok=True)

                    if args.all_eps_one_vid:
                        all_ep_imgs.extend(ep_imgs)
                        print(f"ep {ep_i + 1}/{args.num_eps} complete, appending to all images")
                    else:
                        fig_common.imgs_to_vid(vid_path=vid_path,
                                            name=f"{str(ep_i).zfill(3)}_{speedup_str}x{sto_str}{model_save_str}",
                                            imgs=ep_imgs, frame_rate=frame_rate, out_fr=30, resolution=args.resolution,
                                            crf=args.crf)
                        print(f"Finished saving vid for ep {ep_i}")

            if args.all_eps_one_vid:
                fig_common.imgs_to_vid(vid_path=vid_path,
                                       name=f"{str(args.num_eps).zfill(2)}_eps_{speedup_str}x{sto_str}{model_save_str}",
                                       imgs=all_ep_imgs, frame_rate=frame_rate, out_fr=30, resolution=args.resolution,
                                       crf=args.crf)
                print(f"Finished saving all eps vid for {task}, {algo}, {model}")