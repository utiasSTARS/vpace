import argparse
import os
import sys
from ast import literal_eval

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import cv2
import tqdm

from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
import rl_sandbox.constants as c
import common as plot_common
import data_locations
import hand_dapg_data_locations
import rce_env_data_locations

sys.path.append('..')
import figures.common as fig_common


parser = argparse.ArgumentParser()
parser.add_argument('--env_seed', type=int, default=0)
# parser.add_argument('--task', type=str, default='insert_no_bring_no_move_0')
parser.add_argument('--task', type=str, default='unstack_stack_env_only_no_move_0')
parser.add_argument('--us_and_insert', action='store_true')
# parser.add_argument('--all_tasks', action='store_true')
parser.add_argument('--env_plot', type=str, choices=['single', 'panda', 'sawyer', 'hand'], default='single')
# parser.add_argument('--algo_list', type=str, default='multi-sqil,multi-sqil-no-vp,sqil,sqil-no-vp,disc,rce')
parser.add_argument('--algo_lists', type=str, default='sqil,sqil-no-vp,,multi-sqil,multi-sqil-no-vp')
# parser.add_argument('--aux_task', type=int, default=2)
# parser.add_argument('--model', type=str, default='500000.pt')
parser.add_argument('--model_train_prop', type=float, default=1.0)
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--top_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'], 'results'))
parser.add_argument('--top_dir', type=str, default="/media/ssd_2tb/data/lfebp")
parser.add_argument('--top_save_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'],
                                                                     'figures', 'q_traj_values'))
parser.add_argument('--cam_str', type=str, default='panda_play_higher_closer')
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--render_no_plot', action='store_true')
parser.add_argument('--extra_name', type=str, default='')
parser.add_argument('--force_vert_squish', action='store_true')
parser.add_argument('--bigger_labels', action='store_true')
parser.add_argument('--log_y_axis', action='store_true')
parser.add_argument('--compare_max_direct', action='store_true')
parser.add_argument('--model_str', type=str, default='last')

# img options
parser.add_argument('--save_ep_imgs', action='store_true')
parser.add_argument('--panda_cam_str', type=str, default='panda_play_higher_closer')
parser.add_argument('--img_w', type=int, default=500)
parser.add_argument('--img_h', type=int, default=500)

args = parser.parse_args()

seeds = [1,2,3,4,5]
# algo_list = args.algo_list.split(',')

if args.us_and_insert:
    # possible options:
    # - env seed 1, seed 4 (index 3), insert
    # - env seed 0, seed 4 (index 3), unstack

    tasks = ['unstack_stack_env_only_no_move_0', 'insert_no_bring_no_move_0']
    # tasks = ['insert_no_bring_no_move_0', 'unstack_stack_env_only_no_move_0'] # TODO TEMP FOR TESTING, DELETE
    # args.algo_lists = 'multi-sqil,multi-sqil-no-vp'
    # args.algo_lists = 'multi-sqil-no-vp,,multi-sqil'
    args.algo_lists = 'multi-sqil-no-vp,multi-sqil'
else:
    tasks = [args.task]

if args.env_plot != 'single':
    if args.env_plot == 'panda':
        tasks = list(data_locations.main.keys())
    elif args.env_plot == 'sawyer':
        tasks = list(rce_env_data_locations.main_performance.keys())
    elif args.env_plot == 'hand':
        tasks = list(hand_dapg_data_locations.main_performance.keys())

num_timesteps_mean = 10
# num_timesteps_mean = 1
log_y_axis = args.log_y_axis

# do separate algo lists per row
algo_lists = args.algo_lists.split(',,')
algo_lists = [l.split(',') for l in algo_lists]

# not using these parameters for now, just taking seeds as they come regardless of success
want_suc = True
one_suc_one_fail = False
max_tries = 40

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}')

if args.us_and_insert:
    fig_path = os.path.join(args.top_save_dir, "us_and_insert")
    fig_name = f"plot{args.extra_name}.pdf"
    os.makedirs(os.path.join(args.top_save_dir, "us_and_insert"), exist_ok=True)
elif len(tasks) == 1:
    # fig_path = os.path.join(args.top_save_dir, f"{tasks[0][:15]}.pdf")
    fig_path = os.path.join(args.top_save_dir, f"{tasks[0]}")
    fig_name = f"{args.algo_lists}{args.extra_name}.pdf"
else:
    fig_path = os.path.join(args.top_save_dir, f"{tasks[0]}")
    fig_name = f"{args.env_plot}-multi-task.pdf"

if not args.render_no_plot:
    fig_shape, plot_size, num_stds, font_size, _, cmap, linewidth, std_alpha, _, _, _, _ = \
        plot_common.get_fig_defaults(num_plots=len(tasks))
    if args.us_and_insert:
        fig_shape = [len(algo_lists), len(tasks)]
    else:
        fig_shape = [len(tasks), len(algo_lists)]

    if args.force_vert_squish:
        plot_size[0] = 4.2
        font_size += 2

    if args.bigger_labels:
        font_size += 2
        linewidth *= 2

    fig, axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                            figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])

    num_stds = 1

    if fig_shape == [1, 1]:
        axes = np.array([[axes]])
    elif fig_shape[0] == 1 or fig_shape[1] == 1:
        axes = np.array([axes])
    # else:
    #     axes = axes.flatten()

for task_i, task in enumerate(tasks):
    env = None
    expert_buffer = None
    for algo_list_i, algo_list in enumerate(algo_lists):
        for algo_i, algo in enumerate(algo_list):
            task_settings_dict, data_loc_dict, main_task_i = fig_common.get_task_settings(task)
            # if 'sawyer' in task or 'human' in task:
            #     task_settings_dict = fig_common.SAWYER_HAND_SETTINGS_DICT[task]
            #     if 'sawyer' in task:
            #         data_loc_dict = rce_env_data_locations.main_performance
            #     else:
            #         data_loc_dict = hand_dapg_data_locations.main_performance
            #     main_task_i = 0
            # else:
            #     task_settings_dict = fig_common.PANDA_SETTINGS_DICT[task]
            #     main_task_i = plot_common.PANDA_TASK_SETTINGS[task]['main_task_i']
            #     data_loc_dict = data_locations.main

            multitask_algo = 'multi' in algo or 'ace' in algo

            # load model settings
            config_file = 'lfgp_experiment_setting.pkl' if multitask_algo else 'dac_experiment_setting.pkl'

            # choose model based on model_train_prop  TODO not implemented yet, just manually trying earlier models first
            # model_str = '500000.pt'
            if args.model_str == 'last':
                model_str = task_settings_dict['last_model']
            elif args.model_str == 'half':
                model_str = task_settings_dict['half_model']
            else:
                model_str = f"{args.model_str}.pt"

            all_seed_values = []
            all_seed_expert_values = []
            all_ep_imgs = []

            for seed in seeds:
                # load model + env if not yet loaded
                data_path = fig_common.full_path_from_alg_expname(
                    os.path.join(args.top_dir, 'results'), task, seed, data_loc_dict[task][algo])

                config, env, buffer_preprocess, agent = fig_common.load_model_and_env_once(
                    args.env_seed, os.path.join(data_path, config_file), os.path.join(data_path, model_str),
                    main_task_i, args.device, include_disc=False, force_egl=True, env=env
                )

                # auxiliary_reward, auxiliary_success = fig_common.get_aux_reward_success(config, env)
                print(f"loaded agent for task {task}, algo {algo}, seed {seed}")

                if multitask_algo:
                    # for weighted random scheduler, this sets deterministic action to run what we want
                    agent.high_level_model._intention_i = np.array(main_task_i)

                # run model for single episode, recording q values
                env.seed(args.env_seed)  # exact same ep for all tests...might not create enough variety though

                fig_common.set_cam(args, task, env)

                got_suc_or_fail = False
                num_tries = 0
                while not got_suc_or_fail:
                    obs = buffer_preprocess(env.reset())
                    h_state = agent.reset()

                    done = False
                    ts = 0
                    all_obs = []
                    all_acts = []
                    ep_imgs = []
                    print(f"ep {num_tries}")
                    while not done:
                        if args.save_ep_imgs:
                            imgs = fig_common.get_ts_imgs(args, task, env, ts, get_panda_substeps=False)
                            ep_imgs.extend(imgs)
                        if args.render_no_plot:
                            env.render()
                        if args.stochastic:
                            action, h_state, act_info = agent.compute_action(obs=obs, hidden_state=h_state)
                        else:
                            action, h_state, act_info = agent.deterministic_action(obs=obs, hidden_state=h_state)

                        if config[c.CLIP_ACTION]:
                            action = np.clip(action, a_min=config[c.MIN_ACTION], a_max=config[c.MAX_ACTION])

                        # this calculates value including log prob, we could also manually calculate qs with
                        all_obs.append(obs)
                        all_acts.append(action)

                        if 'sawyer' in task or 'human' in task:
                            next_obs, reward, done, env_info = env.step(action)
                        else:
                            next_obs, reward, done, env_info = env.step(action, substep_render_delay=5)

                        next_obs = buffer_preprocess(next_obs)

                        # suc = auxiliary_success(observation=obs, action=action, env_info=env_info['infos'][-1])

                        obs = next_obs
                        ts += 1

                    num_tries += 1

                    all_obs = torch.tensor(np.array(all_obs), dtype=torch.float32).to(args.device)
                    all_acts = torch.tensor(np.array(all_acts), dtype=torch.float32).to(args.device)
                    ep_values, _, _, _ = agent.model.q_vals(all_obs, h_state, all_acts)

                    # print(f"suc? {suc}, avg q: {ep_values[-10:].mean()}")
                    if False:  # for quick removal of this, in case we don't like it
                        if want_suc:
                            if multitask_algo:
                                got_suc_or_fail = suc[main_task_i]
                            else:
                                got_suc_or_fail = suc[0]
                        else:
                            if multitask_algo:
                                got_suc_or_fail = not suc[main_task_i]
                            else:
                                got_suc_or_fail = not suc[0]
                        if num_tries >= max_tries:
                            import ipdb; ipdb.set_trace()
                    else:
                        got_suc_or_fail = True

                if multitask_algo:
                    ep_values = ep_values[:, main_task_i]
                all_seed_values.append(ep_values.detach().cpu().numpy().flatten())

                # also need q values for expert data, average
                if expert_buffer is None:
                    # load the data once
                    if multitask_algo:
                        expert_data_in_path_ind = config[c.EXPERT_BUFFERS][main_task_i].index('expert_data')
                        saved_expert_buffer_path = config[c.EXPERT_BUFFERS][main_task_i]
                        amount = config[c.EXPERT_AMOUNTS][main_task_i]
                    else:
                        expert_data_in_path_ind = config[c.EXPERT_BUFFER].index('expert_data')
                        saved_expert_buffer_path = config[c.EXPERT_BUFFER]
                        amount = config[c.EXPERT_AMOUNT]

                    expert_buffer_path = os.path.join(args.top_dir, saved_expert_buffer_path[expert_data_in_path_ind:])
                    frame_stack = 1
                    for wrap_dict in config[c.ENV_SETTING][c.ENV_WRAPPERS]:
                        if wrap_dict[c.WRAPPER] == FrameStackWrapper:
                            frame_stack = wrap_dict[c.KWARGS][c.NUM_FRAMES]
                    config[c.EXPERT_BUFFER_SETTING][c.KWARGS][c.DEVICE] = torch.device(args.device)
                    expert_buffer = make_buffer(config[c.EXPERT_BUFFER_SETTING], seed, expert_buffer_path, end_idx=amount,
                                                match_load_size=True, frame_stack_load=frame_stack)

                expert_buf_pol_actions, _ = agent.model.act_lprob(expert_buffer.observations, expert_buffer.hidden_states)
                if multitask_algo:
                    expert_buf_pol_actions = expert_buf_pol_actions[:, main_task_i]

                expert_q_vals, _, _, _ = agent.model.q_vals(expert_buffer.observations, h_state,
                                                            expert_buf_pol_actions.detach())
                if multitask_algo:
                    expert_q_vals = expert_q_vals[:, main_task_i]
                all_seed_expert_values.append(expert_q_vals.detach().cpu().numpy().flatten())

                if args.save_ep_imgs:
                    all_ep_imgs.append(ep_imgs)
                    # img_path = os.path.join(fig_path, task, algo, str(seed))
                    # os.makedirs(img_path, exist_ok=True)
                    # for img_i in tqdm.trange(len(ep_imgs)):
                    #     # swap channel order
                    #     img = cv2.cvtColor(ep_imgs[img_i], cv2.COLOR_BGRA2RGBA)

                    #     # caption with value
                    #     img = fig_common.img_caption(
                    #                 img, aux_task_dict[int(agent.curr_high_level_act)])

                    #     cv2.imwrite(os.path.join(img_path, f"{str(img_i).zfill(4)}.png"), img)

            all_seed_values = np.array(all_seed_values)
            mean = all_seed_values.mean(axis=0)
            std = all_seed_values.std(axis=0)
            all_seed_max = all_seed_values.max(axis=0)

            # alternatively, just using difference between q values and expert max, per seed
            all_seed_expert_values = np.array(all_seed_expert_values)
            per_seed_expert_mean = all_seed_expert_values.mean(axis=1)

            if args.compare_max_direct:
                q_minus_expert = all_seed_values
            else:
                q_minus_expert = all_seed_values - per_seed_expert_mean[:, None]
            mean = q_minus_expert.mean(axis=0)
            std = q_minus_expert.std(axis=0)
            all_seed_max = q_minus_expert.max(axis=0)

            if args.save_ep_imgs:
                for seed_i, seed in enumerate(seeds):
                    img_path = os.path.join(fig_path, task, algo, str(seed))
                    os.makedirs(img_path, exist_ok=True)
                    for img_i in tqdm.trange(len(all_ep_imgs[seed_i])):
                        # swap channel order
                        img = cv2.cvtColor(all_ep_imgs[seed_i][img_i], cv2.COLOR_BGRA2RGBA)

                        # caption with value
                        img = fig_common.img_caption(
                                    img, f"{q_minus_expert[seed_i, img_i]:.3f}")

                        cv2.imwrite(os.path.join(img_path, f"{str(img_i).zfill(4)}.png"), img)

            # smooth
            if num_timesteps_mean > 1:
                convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
                mean = convolve1d(mean, convolv_op, axis=0, mode='nearest')
                std = convolve1d(std, convolv_op, axis=0, mode='nearest')
                all_seed_max = convolve1d(all_seed_max, convolv_op, axis=0, mode='nearest')

            # plot setup
            if args.us_and_insert:
                ax = axes[algo_list_i, task_i]
            else:
                ax = axes[task_i, algo_list_i]
            # ax.set_xlabel('Timestep', fontsize=font_size - 2)
            # ax.set_ylabel('Q Value', fontsize=font_size - 2)
            ax.grid(alpha=0.5, which='both')

            # if multitask_algo:
            #     line_style = '-'
            # else:
            #     line_style = '--'
            line_style = '-'
            cmap_i = plot_common.ALGO_TITLE_DICT[algo]['cmap_i']
            label = plot_common.ALGO_TITLE_DICT[algo]['title'] if task_i == 0 else ""

            # plot q values mean + std
            x_vals = np.arange(len(mean))
            ax.set_xlim(x_vals[0], x_vals[-1])
            ax.plot(x_vals, mean, label=label, color=cmap(cmap_i), linewidth=linewidth, linestyle=line_style)

            # fill based on standard deviation
            ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(cmap_i), alpha=std_alpha)

            # fill based on maximum
            # ax.fill_between(x_vals, mean, all_seed_max, facecolor=cmap(cmap_i), alpha=0.15)

            # plot the maximum as separate line
            ax.plot(x_vals, all_seed_max, color=cmap(cmap_i), linewidth=linewidth, linestyle='--')

            # plot horizontal line with current average expert q value estimate
            # mean_expert = all_seed_expert_values.mean(axis=1)
            # std_expert = all_seed_expert_values.mean(axis=1).std()

            # ax.axhline(mean_expert.mean(), linewidth=linewidth, color=cmap(cmap_i), linestyle="-.")

            # exp_mean_rep = np.repeat(mean_expert.mean(), len(x_vals))
            # exp_std_rep = np.repeat(std_expert, len(x_vals))
            # ax.fill_between(x_vals, exp_mean_rep - num_stds * exp_std_rep, exp_mean_rep + num_stds * exp_std_rep,
            #                 facecolor=cmap(cmap_i), alpha=std_alpha)

            # ax.axhline(mean_expert.max(), linewidth=linewidth, color=cmap(cmap_i), linestyle=(0, (3, 1, 1, 1)))

        if args.us_and_insert and algo_list_i == 0:
            ax.set_title(f"{task_settings_dict['title']}", fontsize=font_size)

        if args.compare_max_direct:
            # TODO this doesn't work right now, and is probably a bad idea because each seed has very different expert maximums
            import ipdb; ipdb.set_trace()
            max_val = per_seed_expert_mean[:, None]
            label = r"$\mathbb{E}_{\mathcal{B}^*} \left[ V^{\pi}(s^*) \right]$"
        else:
            max_val = 0
            label = "Ideal max"

        if task_i == 0 and algo_list_i == len(algo_lists) - 1:
            ax.axhline(max_val, linewidth=linewidth, color='black', linestyle='-.', label=label)
        else:
            ax.axhline(max_val, linewidth=linewidth, color='black', linestyle='-.')

        if log_y_axis:
            ax.set_yscale('log')

# add final labelling, legend, and save
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_xlabel(r"Episode Timestep $t$", fontsize=font_size)
# ax.set_ylabel("Q Value", fontsize=font_size)
# ax.set_ylabel("Q Value - Mean Expert", fontsize=font_size)
if args.compare_max_direct:
    ax.set_ylabel(r"$Q(s_t,a_t)", fontsize=font_size, labelpad=10.0)
else:
    # ax.set_ylabel(r"$Q(s_t,a_t) - \mathbb{E}_{\mathcal{B}^*} \left[ Q(s^*,a^*) \right]$", fontsize=font_size, labelpad=10.0)
    ax.set_ylabel(r"$Q(s_t,a_t) - \mathbb{E}_{\mathcal{B}^*} \left[ V^{\pi}(s^*) \right]$", fontsize=font_size, labelpad=10.0)

# per-row titles for task
# grid = plt.GridSpec(fig_shape[0], fig_shape[1])
for row_i in range(fig_shape[0]):
    # row = fig.add_subplot(grid)
    ax = fig.add_subplot(fig_shape[0], 1, row_i + 1, frameon=False)
    task = tasks[row_i]
    if 'sawyer' in task or 'human' in task:
        title = task
    else:
        task_settings_dict = fig_common.PANDA_SETTINGS_DICT[task]
        title = task_settings_dict['title']

    if not args.us_and_insert:
        ax.set_title(f"{title}", fontsize=font_size)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

# if 'sawyer' in task or 'human' in task:
#     title = task
# else:
#     title = task_settings_dict['title']
# ax.set_title(title, fontsize=font_size)



# TODO consider adding a multicolour patch here for describing all maximums
# could not figure this out...doing it manually
# from matplotlib.collections import LineCollection
# from matplotlib.lines import Line2D

# t = np.linspace(0, 10, 200)
# x = np.cos(np.pi * t)
# y = np.sin(t)
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)

# all_algo_colors = []
# split = args.algo_lists.split(',,')
# for l in split:
#     algo_list = l.split(',')
#     for algo in algo_list:
#         all_algo_colors.append(cmap(plot_common.ALGO_TITLE_DICT[algo]['cmap_i']))

# lc = LineCollection(segments, colors=all_algo_colors,
#                     norm=plt.Normalize(0, 10), linewidth=linewidth, label='Max')

# # handler_map = leg.get_legend_handler_map()
# h, l = fig.get_legend_handles_labels()
# line = Line2D([0], [0], label='manual line', color='k')
# h.append(lc)


if len(tasks) == 1:
    if len(algo_lists) == 1:
        bbta_dict = {
            1: (0.475, -.35),
            2: (0.475, -.45),
            3: (0.475, -.55),
            4: (0.475, -.65),
            5: (0.475, -.75),
            6: (0.475, -.85),
        }
        leg = fig.legend(fancybox=True, shadow=True, fontsize=font_size, loc="lower center", ncol=1,
                bbox_to_anchor=bbta_dict[len(algo_list)])
    else:
        # bbta_dict = {
        #     1: (0.475, -.35),
        #     2: (0.475, -.35),
        #     4: (0.475, -.35),
        #     5: (0.475, -.75),
        #     6: (0.475, -.85),
        # }
        leg = fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=3,
                bbox_to_anchor=(0.475, -.4))
else:
    if args.us_and_insert:
        if len(algo_lists) == 1:
            leg = fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=3,
                        bbox_to_anchor=(0.475, -.4))
        elif len(algo_lists) == 2:
            leg = fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=4,
                        bbox_to_anchor=(0.475, -.13))
    else:
        bbta_num_tasks_dict = {
                5: (0.475, 0.01),
                6: (0.475, 0.01),
                7: (0.475, 0.02),
            }
        leg = fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=3,
                    bbox_to_anchor=bbta_num_tasks_dict[len(tasks)])

fig.subplots_adjust(hspace=.35)

os.makedirs(fig_path, exist_ok=True)

if args.log_y_axis:
    fig_name = f"log_y_{fig_name}"
if args.compare_max_direct:
    fig_name = f"compare_max_direct_{fig_name}"

fig_name = f"{args.model_str}_{fig_name}"

fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')