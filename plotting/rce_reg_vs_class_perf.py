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
# parser.add_argument('--algos', type=str, default='sqil,rce,rce_orig,sqil_theirs')
# parser.add_argument('--algos', type=str, default='multi-sqil,disc,sqil,rce_orig,sqil_theirs')
# parser.add_argument('--algos', type=str, default='multi-sqil,sqil,rce_orig,sqil_theirs,disc')
# parser.add_argument('--algos', type=str, default='rce_orig,sqil_theirs,sqil_theirs_nstepoff')
parser.add_argument('--algos', type=str, default='sqil-no-vp,rce_orig,sqil_theirs,sqil_theirs_nstepoff')
parser.add_argument('--stddev_type', type=str, choices=['none', 'by_task', 'by_seed_mean'], default='by_task',
                    help="by_task is standard deviation of mean task performance, "\
                         "by_seed_mean is the mean of each tasks across-seed std dev.")
args = parser.parse_args()

fig_name = f"rce_reg_vs_class_performance"
plot_name = 're_vs_cl'
# valid_tasks = [*plot_common.RCE_TASK_SETTINGS.keys()]
# valid_tasks.extend(['door-human-v0', 'hammer-human-v0'])
valid_task_settings = {**plot_common.RCE_TASK_SETTINGS}
valid_task_settings['door-human-v0'] = plot_common.HAND_TASK_SETTINGS['door-human-v0']
valid_task_settings['hammer-human-v0'] = plot_common.HAND_TASK_SETTINGS['hammer-human-v0']
valid_algos = args.algos.split(',')
main_task_idx = 0
num_timesteps_mean = 5

root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name=fig_name)

task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use, \
    st_num_eval_steps_to_use, eval_intervals, eval_eps_per_task = \
    plot_common.get_task_defaults(plot=plot_name)

# algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task, _, cmap_is = \
algo_dir_names, algo_titles, multitask_algos, _, cmap_is = \
    plot_common.get_algo_defaults(plot=plot_name)

fig_shape, plot_size, num_stds, font_size, _, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline, num_timesteps_mean = \
    plot_common.get_fig_defaults(num_plots=1)
#####################################################################################################################

# pretty plotting, allow tex
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# get returns and successes
all_returns, all_successes = plot_common.get_success_return(
    reload_data=args.reload_data,
    task_dir_names=task_dir_names,
    valid_task=valid_task,
    algo_dir_names=algo_dir_names,
    num_eval_steps_to_use=num_eval_steps_to_use,
    multitask_algos=multitask_algos,
    st_num_eval_steps_to_use=st_num_eval_steps_to_use,
    data_locations={**rce_data_locations, **hand_data_locations},
    experiment_root_dir=experiment_root_dir,
    seeds=seeds,
    task_data_filenames=task_data_filenames,
    num_aux=num_aux,
    eval_eps_per_task=eval_eps_per_task,
    fig_path=fig_path,
    valid_algos=valid_algos
)

# directly load the data from sawyer and hand plots saved data, so we don't accidentally use diff data..and to save time
# all_returns = {}; all_successes = {}
# for data_fig_name in ['rce_performance', 'hand_performance']:
#     data = pickle.load(open(os.path.join(root_dir, 'figures', data_fig_name, 'data', 'data.pkl'), 'rb'))
#     for r_type in ['all_returns', 'all_successes']:
#         for k, v in data[r_type].items():
#             if k in valid_task_settings:
#                 locals()[r_type][k] = v

# from now on, ignoring successes since they're all 0 for these envs
# now going to normalize return values from 0 to 1, and timesteps to be on same scale with simple linear interpolation
# finally, add all to single numpy array
# standard deviation across all data will be meaningless, but could take the mean of each (return 0-1 normalized)
# standard deviation across seeds for each individual task
# alternatively, could just do standard deviation (across tasks) of mean performance at each timestep instead.

# first find the max number of eval timsteps of any task, to decide what we'll interpolate all to
max_eval_steps = 0
for task in all_returns.keys():
    eval_steps = all_returns[task][valid_algos[0]]['raw'].shape[1]
    max_eval_steps = max(eval_steps, max_eval_steps)

# all_rets_norm = copy.deepcopy(all_returns)
across_task_rets = {}
all_rets_norm = {}
all_rets_norm_interp = {}
for task_i, task in enumerate(all_returns.keys()):
    min_ret = valid_task_settings[task]['return_ylims'][0]
    max_ret = valid_task_settings[task]['return_ylims'][1]
    all_rets_norm[task] = {}
    all_rets_norm_interp[task] = {}

    for algo in valid_algos:
        all_rets_norm[task][algo] = {}
        all_rets_norm_interp[task][algo] = {}

        # remove non-main task data if multitask
        if algo in multitask_algos:
            all_returns[task][algo]['raw'] = all_returns[task][algo]['raw'][:, :, main_task_idx, :]
            all_returns[task][algo]['mean'] = all_returns[task][algo]['mean'][:, main_task_idx]
            all_returns[task][algo]['std'] = all_returns[task][algo]['std'][:, main_task_idx]

        # first normalize the returns based on max_ret_norm
        all_rets_norm[task][algo]['raw'] = (all_returns[task][algo]['raw'] - min_ret) / (max_ret - min_ret)
        all_rets_norm[task][algo]['mean'] = (all_returns[task][algo]['mean'] - min_ret) / (max_ret - min_ret)
        per_seed_mean = all_rets_norm[task][algo]['raw'].mean(axis=-1)
        all_rets_norm[task][algo]['std'] = per_seed_mean.std(axis=0)

        # then normalize the eval timesteps with interpolation and the max_eval_steps
        num_eval_steps = all_rets_norm[task][algo]['raw'].shape[1]
        if num_eval_steps < max_eval_steps:
            orig_x = np.arange(1, num_eval_steps + 1) * eval_intervals[task_i]
            new_x = np.arange(1, max_eval_steps + 1) * eval_intervals[task_i]

            # expand orig_x to have same min and max as new x
            orig_x_new_max = (new_x.max() - new_x.min()) / (orig_x.max() - orig_x.min()) * \
                             (orig_x - orig_x.min()) + new_x.min()

            all_rets_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['mean'])
            all_rets_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['mean'])

        else:
            all_rets_norm_interp[task][algo]['mean'] = all_rets_norm[task][algo]['mean']
            all_rets_norm_interp[task][algo]['std'] = all_rets_norm[task][algo]['std']

        # put across-task data together
        if algo not in across_task_rets: across_task_rets[algo] = {'means': [], 'stds': []}
        across_task_rets[algo]['means'].append(all_rets_norm_interp[task][algo]['mean'])
        across_task_rets[algo]['stds'].append(all_rets_norm_interp[task][algo]['std'])

# plot the means
s_fig, s_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
r_ax = r_axes

for algo in valid_algos:
    all_means = np.array(across_task_rets[algo]['means']).T
    all_stds = np.array(across_task_rets[algo]['stds']).T

    if num_timesteps_mean > 1:
        # convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
        # mean_padded = np.pad(mean, num_timesteps_mean // 2, mode='edge')
        # mean = np.convolve(convolv_op, mean_padded, mode='valid')
        # all_means_padded = np.pad(all_means, ((num_timesteps_mean // 2, num_timesteps_mean // 2), (0,0)), mode='edge')
        convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
        all_means = convolve1d(all_means, convolv_op, axis=0, mode='nearest')
        all_stds = convolve1d(all_stds, convolv_op, axis=0, mode='nearest')

    norm_eval_steps = np.linspace(0, 1, max_eval_steps)

    if 'muliti' in algo:
        line_style = '-'
    # elif 'theirs' in plot_common.ALGO_TITLE_DICT[algo]['title']:
    #     line_style = '-.'
    else:
        line_style = '--'

    mean = all_means.mean(axis=-1)
    color = cmap(plot_common.ALGO_TITLE_DICT[algo]['cmap_i'])
    r_ax.plot(norm_eval_steps, mean, label=plot_common.RCE_REG_VS_CLASS_DICT[algo]['title'],
              color=color, linewidth=linewidth, linestyle=line_style)

    if args.stddev_type != 'none':

        if args.stddev_type == 'by_task':
            std = all_means.std(axis=-1)
        elif args.stddev_type == 'by_seed_mean':
            std = all_stds.mean(axis=-1)

        r_ax.fill_between(norm_eval_steps, mean - num_stds * std, mean + num_stds * std, facecolor=color,
                          alpha=std_alpha)


# pretty/labels
r_ax.set_title('Sawyer \\& Adroit (Average)', fontsize=font_size)
r_ax.set_xlabel('Steps (normalized)', fontsize=font_size - 2)
r_ax.set_ylabel('Return (normalized)', fontsize=font_size - 2)
r_ax.set_xlim([0, 1])
r_ax.set_ylim([0, 1])
r_ax.grid(alpha=0.5, which='both')

# bbox_to_anchor_dict = {
#     2: (0.5, -.45),
#     3: (0.5, -.575),
#     4: (0.5, -.7),
# }
bbox_to_anchor_dict = {
    2: (0.45, -.4),
    3: (0.45, -.535),
    4: (0.45, -.65),
}

# r_fig.legend(fancybox=True, shadow=True, fontsize=font_size-2,
r_fig.legend(fancybox=True, shadow=True, fontsize=font_size-3,
             loc="lower center", ncol=1, bbox_to_anchor=bbox_to_anchor_dict[len(valid_algos)])

fig_path += args.extra_name

os.makedirs(fig_path, exist_ok=True)
r_fig.savefig(os.path.join(fig_path, 'r_fig.pdf'), bbox_inches='tight')