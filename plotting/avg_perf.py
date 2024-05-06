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
import common as plot_common


#### Options ########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--extra_name', type=str, default="")
parser.add_argument('--reload_data', action='store_true')
parser.add_argument('--plot', type=str, default='all',
                    choices=['main', 'rce', 'hand', 'hand_orig', 'hand_dp', 'all', 'all_4_sep', 'rce_env_mods'])
parser.add_argument('--stddev_type', type=str, choices=['none', 'by_task', 'by_seed_mean'], default='by_task',
                    help="by_task is standard deviation of mean task performance, "\
                         "by_seed_mean is the mean of each tasks across-seed std dev.")
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
#####################################################################################################################

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
for data_fig_name in ['main_performance', 'rce_performance', 'hand_performance']:
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
    max_eval_steps = 0
    for task in valid_task_settings.keys():
        eval_steps = all_returns[task][valid_algos[0]]['raw'].shape[1]
        max_eval_steps = max(eval_steps, max_eval_steps)

    # all_rets_norm = copy.deepcopy(all_returns)
    across_task_rets = {}
    all_rets_norm = {}
    all_rets_norm_interp = {}
    all_sucs_norm_interp = {}  # don't need norm, since success rates are already normalized from 0 to 1
    # for task_i, task in enumerate(all_returns.keys()):
    for task_i, task in enumerate(valid_task_settings.keys()):

        if task not in panda_task_settings:
            try:
                min_ret = valid_task_settings[task]['return_ylims'][0]
                max_ret = valid_task_settings[task]['return_ylims'][1]
                all_rets_norm[task] = {}
                all_rets_norm_interp[task] = {}
            except:
                import ipdb; ipdb.set_trace()
        all_sucs_norm_interp[task] = {}

        for algo in valid_algos:
            # remove non-main task data if multitask
            if algo in multitask_algos:
                if task in panda_task_settings:
                    main_task_idx = panda_task_settings[task]['main_task_i']
                else:
                    main_task_idx = 0

                all_returns[task][algo]['raw'] = all_returns[task][algo]['raw'][:, :, main_task_idx, :]
                all_returns[task][algo]['mean'] = all_returns[task][algo]['mean'][:, main_task_idx]
                all_returns[task][algo]['std'] = all_returns[task][algo]['std'][:, main_task_idx]

                all_successes[task][algo]['raw'] = all_successes[task][algo]['raw'][:, :, main_task_idx, :]
                all_successes[task][algo]['mean'] = all_successes[task][algo]['mean'][:, main_task_idx]
                all_successes[task][algo]['std'] = all_successes[task][algo]['std'][:, main_task_idx]


            # first normalize the returns based on max_ret_norm
            if task not in panda_task_settings:
                all_rets_norm[task][algo] = {}
                all_rets_norm_interp[task][algo] = {}
                all_rets_norm[task][algo]['raw'] = (all_returns[task][algo]['raw'] - min_ret) / (max_ret - min_ret)
                all_rets_norm[task][algo]['mean'] = (all_returns[task][algo]['mean'] - min_ret) / (max_ret - min_ret)
                per_seed_mean = all_rets_norm[task][algo]['raw'].mean(axis=-1)
                all_rets_norm[task][algo]['std'] = per_seed_mean.std(axis=0)
            else:
                all_sucs_norm_interp[task][algo] = {}

            # then normalize the eval timesteps with interpolation and the max_eval_steps
            num_eval_steps = all_returns[task][algo]['raw'].shape[1]
            if num_eval_steps < max_eval_steps:
                if task in panda_task_settings:
                    eval_interval = panda_task_settings[task]['eval_intervals']
                else:
                    eval_interval = 10000

                orig_x = np.arange(1, num_eval_steps + 1) * eval_interval
                new_x = np.arange(1, max_eval_steps + 1) * eval_interval

                # expand orig_x to have same min and max as new x
                orig_x_new_max = (new_x.max() - new_x.min()) / (orig_x.max() - orig_x.min()) * \
                                (orig_x - orig_x.min()) + new_x.min()

                if task not in panda_task_settings:
                    all_rets_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['mean'])
                    all_rets_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['std'])

                else:
                    all_sucs_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_successes[task][algo]['mean'])
                    all_sucs_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_successes[task][algo]['std'])

            else:
                if task not in panda_task_settings:
                    all_rets_norm_interp[task][algo]['mean'] = all_rets_norm[task][algo]['mean']
                    all_rets_norm_interp[task][algo]['std'] = all_rets_norm[task][algo]['std']

                else:
                    all_sucs_norm_interp[task][algo]['mean'] = all_successes[task][algo]['mean']
                    all_sucs_norm_interp[task][algo]['std'] = all_successes[task][algo]['std']

            # put across-task data together
            if algo not in across_task_rets: across_task_rets[algo] = {'means': [], 'stds': []}

            if task in panda_task_settings:
                across_task_rets[algo]['means'].append(all_sucs_norm_interp[task][algo]['mean'])
                across_task_rets[algo]['stds'].append(all_sucs_norm_interp[task][algo]['std'])
            else:
                across_task_rets[algo]['means'].append(all_rets_norm_interp[task][algo]['mean'])
                across_task_rets[algo]['stds'].append(all_rets_norm_interp[task][algo]['std'])

    # plot the means
    for algo in valid_algos:
        all_means = np.array(across_task_rets[algo]['means']).T
        all_stds = np.array(across_task_rets[algo]['stds']).T

        if num_timesteps_mean > 1:
            convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
            all_means = convolve1d(all_means, convolv_op, axis=0, mode='nearest')
            all_stds = convolve1d(all_stds, convolv_op, axis=0, mode='nearest')

        norm_eval_steps = np.linspace(0, 1, max_eval_steps)

        if 'multi' in algo:
            line_style = '-'
        # elif 'theirs' in plot_common.ALGO_TITLE_DICT[algo]['title']:
        #     line_style = '-.'
        else:
            line_style = '--'

        mean = all_means.mean(axis=-1)
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
            if args.stddev_type == 'by_task':
                std = all_means.std(axis=-1)
            elif args.stddev_type == 'by_seed_mean':
                std = all_stds.mean(axis=-1)

            r_ax.fill_between(norm_eval_steps, mean - num_stds * std, mean + num_stds * std, facecolor=color,
                            alpha=std_alpha)


    # pretty/labels
    r_ax.set_title(plot_title, fontsize=font_size)
    if args.plot != 'all_4_sep':
        r_ax.set_xlabel('Steps (normalized)', fontsize=font_size + 2)

    # if args.plot == 'main' or (args.plot == 'all_4_sep' and plot_i == 0):
    #     r_ax.set_ylabel('Success Rate', fontsize=font_size - 2)
    # else:
    if args.plot == 'all_4_sep':
        if plot_i == 0:
            r_ax.set_ylabel('Return (normalized)', fontsize=font_size + 2)
    else:
        r_ax.set_ylabel('Return (normalized)', fontsize=font_size + 2)
    r_ax.set_xlim([0, 1])
    r_ax.set_ylim([0, 1])
    r_ax.grid(alpha=0.5, which='both')


if args.plot == 'all_4_sep':
    ax = r_fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
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
    bbox_to_anchor = (0.45, -.4)
    fsize = font_size - 2
elif args.plot == 'all_4_sep':
    ncol = 4
    bbox_to_anchor = (0.5, -.425)
    fsize = font_size - 2
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

os.makedirs(fig_path, exist_ok=True)
r_fig.savefig(os.path.join(fig_path, 'r_fig.pdf'), bbox_inches='tight')