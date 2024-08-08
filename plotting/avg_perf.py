import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import math
import ast
import argparse
import copy
from scipy.ndimage import convolve1d

import data_locations
from rce_env_data_locations import main_performance as rce_data_locations
from hand_dapg_data_locations import main_performance as hand_data_locations
from real_panda_data_locations import main_performance as real_data_locations
import common as plot_common


#### Options ########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--extra_name', type=str, default="")
parser.add_argument('--reload_data', action='store_true')
parser.add_argument('--plot', type=str, default='all',
                    choices=['main', 'rce', 'hand', 'hand_orig', 'hand_dp', 'all', 'all_4_sep', 'all_5_sep', 'rce_env_mods'])
parser.add_argument('--stddev_type', type=str, choices=['none', 'by_task', 'by_seed_mean'], default='by_task',
                    help="by_task is standard deviation of mean task performance, "\
                         "by_seed_mean is the mean of each tasks across-seed std dev.")
parser.add_argument('--force_vert_squish', action='store_true')
parser.add_argument('--real_x_axis', action='store_true')
args = parser.parse_args()

fig_name = f"{args.plot}_envs_avg"
plot_name = fig_name

if args.plot == 'all_4_sep':
    all_valid_task_settings = plot_common.AVG_ENVS_DICT['all']['valid_task_settings']
    valid_algoss = []; valid_task_settingss = []; plot_titles = []; num_timesteps_means = []
    for ptype in ['main', 'rce', 'hand_orig', 'hand_dp']:
        valid_algoss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_algos'])
        valid_task_settingss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_task_settings'])
        plot_titles.append(plot_common.AVG_ENVS_DICT[ptype]['title'])
        num_timesteps_means.append(plot_common.AVG_ENVS_DICT[ptype]['num_timesteps_mean'])
        if 'relocate-human-v0' in valid_task_settingss[-1]:
            del valid_task_settingss['relocate-human-v0']  # since all zeros for everything, and missing some results
else:
    all_valid_task_settings = plot_common.AVG_ENVS_DICT[args.plot]['valid_task_settings']
    valid_algoss = [plot_common.AVG_ENVS_DICT[args.plot]['valid_algos']]
    valid_task_settingss = [plot_common.AVG_ENVS_DICT[args.plot]['valid_task_settings']]
    plot_titles = [plot_common.AVG_ENVS_DICT[args.plot]['title']]
    num_timesteps_means = [plot_common.AVG_ENVS_DICT[args.plot]['num_timesteps_mean']]

    if 'relocate-human-v0' in valid_task_settingss[0]:
        del valid_task_settingss[0]['relocate-human-v0']  # since all zeros for everything, and missing some results

panda_task_settings = {**plot_common.PANDA_TASK_SETTINGS}
real_task_settings = {**plot_common.REAL_PANDA_TASK_SETTINGS}

root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name=fig_name)

# task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use, \
#     st_num_eval_steps_to_use, eval_intervals = \
#     plot_common.get_task_defaults(plot=plot_name)

algo_dir_names, algo_titles, multitask_algos, _, cmap_is = \
    plot_common.get_algo_defaults(plot=plot_name)

if args.plot in ['all_4_sep']:
    num_plots = 4
else:
    num_plots = 1

fig_shape, plot_size, num_stds, font_size, _, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline, _ = \
    plot_common.get_fig_defaults(num_plots=num_plots)
num_stds = 0.25

# hardcoded options for real_x_axis
x_val_scale = 100000
eval_cutoff_env_step = 500000
#####################################################################################################################

if args.force_vert_squish:
    plot_size[0] = 4.2
    font_size += 4

# pretty plotting, allow tex
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# get returns and successes
# all_returns, all_successes = plot_common.get_success_return(
#     reload_data=args.reload_data,
#     task_dir_names=task_dir_names,
#     valid_task=valid_task,
#     algo_dir_names=algo_dir_names,
#     num_eval_steps_to_use=num_eval_steps_to_use,
#     multitask_algos=multitask_algos,
#     st_num_eval_steps_to_use=st_num_eval_steps_to_use,
#     data_locations={**rce_data_locations, **hand_data_locations},
#     experiment_root_dir=experiment_root_dir,
#     seeds=seeds,
#     task_data_filenames=task_data_filenames,
#     num_aux=num_aux,
#     eval_eps_per_task=eval_eps_per_task,
#     fig_path=fig_path,
#     valid_algos=valid_algos
# )

# directly load the data from sawyer and hand plots saved data, so we don't accidentally use diff data..and to save time
all_returns = {}; all_successes = {}
for data_fig_name in ['main_performance', 'rce_performance', 'hand_performance', 'real_performance']:
    data = pickle.load(open(os.path.join(root_dir, 'figures', data_fig_name, 'data', 'data.pkl'), 'rb'))
    for r_type in ['all_returns', 'all_successes']:
        for k, v in data[r_type].items():
            if k in all_valid_task_settings:
                locals()[r_type][k] = v

# make each plot separately, if we're doing a multiplot
# s_fig, s_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
#                              figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])

if hasattr(r_axes, 'shape'):
    r_axes_flat = r_axes.flatten()
else:
    r_axes_flat = [r_axes]

# r_ax = r_axes
for plot_i, r_ax in enumerate(r_axes_flat):
    valid_algos = valid_algoss[plot_i]
    valid_task_settings = valid_task_settingss[plot_i]
    plot_title = plot_titles[plot_i]
    num_timesteps_mean = num_timesteps_means[plot_i]

    # first find the max number of eval timsteps of any task, to decide what we'll interpolate all to
    # also find min eval interval
    max_eval_steps = 0
    max_eval = 0
    min_eval_interval = 25000
    for task in valid_task_settings.keys():
        eval_steps = all_returns[task][valid_algos[0]]['raw'].shape[1]
        max_eval_steps = max(eval_steps, max_eval_steps)
        max_eval = max(max_eval, max_eval_steps * all_valid_task_settings[task]['eval_intervals'])
        min_eval_interval = min(min_eval_interval, all_valid_task_settings[task]['eval_intervals'])

    # all_rets_norm = copy.deepcopy(all_returns)
    across_task_rets = {}
    all_rets_norm = {}
    all_rets_norm_interp = {}
    all_sucs_norm_interp = {}  # don't need norm, since success rates are already normalized from 0 to 1
    # for task_i, task in enumerate(all_returns.keys()):
    for task_i, task in enumerate(valid_task_settings.keys()):

        if task not in panda_task_settings and task not in real_task_settings:
            try:
                min_ret = valid_task_settings[task]['return_ylims'][0]
                max_ret = valid_task_settings[task]['return_ylims'][1]
                all_rets_norm[task] = {}
                all_rets_norm_interp[task] = {}
            except Exception as e:
                print(f"Exception: {e}")
                import ipdb; ipdb.set_trace()
        all_sucs_norm_interp[task] = {}

        if task in real_task_settings:
            loop_valid_algos = ['multi-sqil', 'sqil-no-vp']
        else:
            loop_valid_algos = valid_algos

        for algo in loop_valid_algos:
            # remove non-main task data if multitask
            if algo in multitask_algos:
                if task in panda_task_settings:
                    main_task_idx = panda_task_settings[task]['main_task_i']
                else:
                    main_task_idx = 0

                # hardcode for real robot envs, no seed index
                if task in real_task_settings:
                    all_returns[task][algo]['raw'] = all_returns[task][algo]['raw'][:, main_task_idx, :]
                    all_successes[task][algo]['raw'] = all_successes[task][algo]['raw'][:, main_task_idx, :]
                else:
                    all_returns[task][algo]['raw'] = all_returns[task][algo]['raw'][:, :, main_task_idx, :]
                    all_successes[task][algo]['raw'] = all_successes[task][algo]['raw'][:, :, main_task_idx, :]

                all_returns[task][algo]['mean'] = all_returns[task][algo]['mean'][:, main_task_idx]
                all_returns[task][algo]['std'] = all_returns[task][algo]['std'][:, main_task_idx]

                all_successes[task][algo]['mean'] = all_successes[task][algo]['mean'][:, main_task_idx]
                all_successes[task][algo]['std'] = all_successes[task][algo]['std'][:, main_task_idx]


            # first normalize the returns based on max_ret_norm
            if task not in panda_task_settings and task not in real_task_settings:
                all_rets_norm[task][algo] = {}
                all_rets_norm_interp[task][algo] = {}
                all_rets_norm[task][algo]['raw'] = (all_returns[task][algo]['raw'] - min_ret) / (max_ret - min_ret)
                all_rets_norm[task][algo]['mean'] = (all_returns[task][algo]['mean'] - min_ret) / (max_ret - min_ret)
                per_seed_mean = all_rets_norm[task][algo]['raw'].mean(axis=-1)
                all_rets_norm[task][algo]['std'] = per_seed_mean.std(axis=0)
            else:
                all_sucs_norm_interp[task][algo] = {}

            # hardcode for real robot envs, no seed index
            if task in real_task_settings:
                num_eval_steps = all_returns[task][algo]['raw'].shape[0]
            else:
                num_eval_steps = all_returns[task][algo]['raw'].shape[1]

            if algo not in across_task_rets: across_task_rets[algo] = {'means': [], 'stds': []}

            if args.real_x_axis:
                # 1. interpolate everything to shortest eval intervals, keeping true max timestep the same

                # interpolate everything, but this is a non-op for tasks that already have min_eval_interval
                eval_interval = all_valid_task_settings[task]['eval_intervals']
                orig_final_step = eval_interval * num_eval_steps + 1
                orig_x = np.arange(eval_interval, orig_final_step, eval_interval)
                new_x = np.arange(min_eval_interval, orig_final_step, min_eval_interval)

                if task not in panda_task_settings and task not in real_task_settings:
                    task_rets_mean = all_rets_norm[task][algo]['mean']
                    task_rets_std = all_rets_norm[task][algo]['std']
                else:
                    task_rets_mean = all_successes[task][algo]['mean']
                    task_rets_std = all_successes[task][algo]['std']

                across_task_rets[algo]['means'].append(np.interp(new_x, orig_x, task_rets_mean))
                across_task_rets[algo]['stds'].append(np.interp(new_x, orig_x, task_rets_std))

            else:

                # then normalize the eval timesteps with interpolation and the max_eval_steps
                if num_eval_steps < max_eval_steps:
                    eval_interval = all_valid_task_settings[task]['eval_intervals']
                    # if task in panda_task_settings:
                    #     eval_interval = panda_task_settings[task]['eval_intervals']
                    # elif task in real_task_settings:
                    #     eval_interval = 5000
                    # else:
                    #     eval_interval = 10000

                    orig_x = np.arange(1, num_eval_steps + 1) * eval_interval
                    new_x = np.arange(1, max_eval_steps + 1) * eval_interval

                    # expand orig_x to have same min and max as new x
                    orig_x_new_max = (new_x.max() - new_x.min()) / (orig_x.max() - orig_x.min()) * \
                                    (orig_x - orig_x.min()) + new_x.min()

                    if task not in panda_task_settings and task not in real_task_settings:
                        all_rets_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['mean'])
                        all_rets_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['std'])

                    else:
                        try:
                            all_sucs_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_successes[task][algo]['mean'])
                            all_sucs_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_successes[task][algo]['std'])
                        except:
                            import ipdb; ipdb.set_trace()

                else:
                    if task not in panda_task_settings and task not in real_task_settings:
                        all_rets_norm_interp[task][algo]['mean'] = all_rets_norm[task][algo]['mean']
                        all_rets_norm_interp[task][algo]['std'] = all_rets_norm[task][algo]['std']

                    else:
                        all_sucs_norm_interp[task][algo]['mean'] = all_successes[task][algo]['mean']
                        all_sucs_norm_interp[task][algo]['std'] = all_successes[task][algo]['std']

                # put across-task data together
                if task in panda_task_settings or task in real_task_settings:
                    across_task_rets[algo]['means'].append(all_sucs_norm_interp[task][algo]['mean'])
                    across_task_rets[algo]['stds'].append(all_sucs_norm_interp[task][algo]['std'])
                else:
                    across_task_rets[algo]['means'].append(all_rets_norm_interp[task][algo]['mean'])
                    across_task_rets[algo]['stds'].append(all_rets_norm_interp[task][algo]['std'])

    # plot the means
    for algo in valid_algos:
        if args.real_x_axis:
            # 2. take across-task means at every timestep, regardless of number of tasks that have a value
            #    at that timestep...likely will need to be done in a loop since array would be ragged

            # convolve must be done one task at a time, since each task is a different length now
            if num_timesteps_mean > 1:
                convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
                across_task_rets_smoothed = {'means': [], 'stds': []}
                for task_algo_means, task_algo_stds in zip(across_task_rets[algo]['means'], across_task_rets[algo]['stds']):
                    across_task_rets_smoothed['means'].append(convolve1d(task_algo_means, convolv_op, mode='nearest'))
                    across_task_rets_smoothed['stds'].append(convolve1d(task_algo_stds, convolv_op, mode='nearest'))
                across_task_rets[algo] = across_task_rets_smoothed

            # get per timestep means -- start by getting longest array, should be max_eval / min_eval_interval
            max_new_eval_steps = 0
            for task_algo_means in across_task_rets[algo]['means']:
                max_new_eval_steps = max(max_new_eval_steps, len(task_algo_means))

            # get per timestep mean, since each task is different length
            import awkward as ak
            ak_across_task_rets_means = ak.Array(across_task_rets[algo]['means'])
            ak_across_task_rets_stds = ak.Array(across_task_rets[algo]['stds'])

            mean = np.zeros(max_new_eval_steps,)
            std = np.zeros(max_new_eval_steps,)

            for i in range(max_new_eval_steps):
                timestep_means = ak_across_task_rets_means[ak.num(ak_across_task_rets_means) > i, i].to_numpy()
                mean[i] = timestep_means.mean()
                raw_stds = ak_across_task_rets_stds[ak.num(ak_across_task_rets_stds) > i, i].to_numpy().mean()
                if args.stddev_type != 'none':
                    if args.stddev_type == 'by_task':
                        if len(timestep_means) >= 3:  # only take std of means of we have at least a few valid tasks
                            std[i] = timestep_means.std()
                        else:
                            std[i] = raw_stds.mean()
                    elif args.stddev_type == 'by_seed_mean':
                        std = raw_stds.mean()

            # overwrite max steps if we want to stop it sooner
            if 'eval_cutoff_env_step' in plot_common.AVG_ENVS_DICT[args.plot]:
                eval_cutoff_env_step = plot_common.AVG_ENVS_DICT[args.plot]['eval_cutoff_env_step']
                max_eval = eval_cutoff_env_step
                max_new_eval_steps = int(eval_cutoff_env_step / min_eval_interval)
                mean = mean[:max_new_eval_steps]
                std = std[:max_new_eval_steps]

            norm_eval_steps = np.arange(min_eval_interval, max_new_eval_steps * min_eval_interval + 1,
                                        min_eval_interval) / x_val_scale

            max_tick = max_eval / x_val_scale
            tick_gap = max_tick / 5
            r_ax.set_xticks(np.arange(0.0, max_tick + .1, tick_gap))

        else:
            all_means = np.array(across_task_rets[algo]['means']).T
            all_stds = np.array(across_task_rets[algo]['stds']).T

            if num_timesteps_mean > 1:
                convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
                all_means = convolve1d(all_means, convolv_op, axis=0, mode='nearest')
                all_stds = convolve1d(all_stds, convolv_op, axis=0, mode='nearest')

            norm_eval_steps = np.linspace(0, 1, max_eval_steps)

            mean = all_means.mean(axis=-1)
            if args.stddev_type != 'none':
                if args.stddev_type == 'by_task':
                    std = all_means.std(axis=-1)
                elif args.stddev_type == 'by_seed_mean':
                    std = all_stds.mean(axis=-1)

        if 'multi' in algo:
            line_style = '-'
        # elif 'theirs' in plot_common.ALGO_TITLE_DICT[algo]['title']:
        #     line_style = '-.'
        else:
            line_style = '--'

        color = cmap(plot_common.ALGO_TITLE_DICT[algo]['cmap_i'])

        if plot_i == 0:
            if plot_name == 'all_envs_avg':
                if algo == 'multi-sqil':
                    label = 'VPACE'
                elif algo == 'disc':
                    label = 'DAC'
                else:
                    label = plot_common.ALGO_TITLE_DICT[algo]['title']
            else:
                label = plot_common.ALGO_TITLE_DICT[algo]['title']
        else:
            label = None

        r_ax.plot(norm_eval_steps, mean, label=label, color=color, linewidth=linewidth, linestyle=line_style)
        if args.stddev_type != 'none':
            r_ax.fill_between(norm_eval_steps, mean - num_stds * std, mean + num_stds * std, facecolor=color,
                            alpha=std_alpha)


    # pretty/labels
    r_ax.set_title(plot_title, fontsize=font_size)
    if args.plot != 'all_4_sep':
        if args.real_x_axis:
            r_ax.set_xlabel(r"Env. Steps ($\times$100k)", fontsize=font_size + 2)
        else:
            r_ax.set_xlabel('Steps (normalized)', fontsize=font_size + 2)

    # if args.plot == 'main' or (args.plot == 'all_4_sep' and plot_i == 0):
    #     r_ax.set_ylabel('Success Rate', fontsize=font_size - 2)
    # else:
    if args.plot == 'all_4_sep':
        if plot_i == 0:
            r_ax.set_ylabel('Return (normalized)', fontsize=font_size + 2)
    else:
        r_ax.set_ylabel('Return (normalized)', fontsize=font_size + 2)
    if not args.real_x_axis:
        r_ax.set_xlim([0, 1])
    r_ax.set_ylim([0, 1])
    r_ax.grid(alpha=0.5, which='both')

    r_ax.tick_params(labelsize=font_size-8)

if args.plot == 'all_4_sep':
    ax = r_fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if args.real_x_axis:
        ax.set_xlabel(r"Env. Steps ($\times$100k)", fontsize=font_size + 2)
    else:
        ax.set_xlabel("Steps (normalized)", fontsize=font_size + 2)

bbox_to_anchor_dict = {
    2: (0.45, -.4),
    3: (0.45, -.535),
    4: (0.45, -.65),
    5: (0.45, -.77),
    8: (0.45, -.65),
}

if args.plot == 'all':
    ncol = 2
    if args.force_vert_squish:
        bbox_to_anchor = (0.45, -.55)
    else:
        bbox_to_anchor = (0.45, -.4)
    fsize = font_size - 2
elif args.plot == 'all_4_sep':
    ncol = 4
    bbox_to_anchor = (0.5, -.425)
    fsize = font_size - 2
    if args.force_vert_squish:
        bbox_to_anchor = (0.5, -.54)
else:
    ncol = 1
    bbox_to_anchor_dict = {
        2: (0.45, -.4),
        3: (0.45, -.535),
        4: (0.45, -.65),
        5: (0.45, -.77),
        8: (0.45, -.65),
    }
    bbox_to_anchor = bbox_to_anchor_dict[len(valid_algos)]
    fsize = font_size - 3

r_fig.legend(fancybox=True, shadow=True, fontsize=fsize, loc="lower center", ncol=ncol, bbox_to_anchor=bbox_to_anchor)

fig_path += args.extra_name

fig_name = 'r_fig.pdf'
if args.force_vert_squish:
    fig_name = f"squish_{fig_name}"
if args.real_x_axis:
    fig_name = f"real_x_{fig_name}"

os.makedirs(fig_path, exist_ok=True)
r_fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')