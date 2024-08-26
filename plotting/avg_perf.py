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
from avg_common import get_between_task_mean_std


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
parser.add_argument('--bigger_labels', action='store_true')
parser.add_argument('--real_x_axis', action='store_true')
parser.add_argument('--use_rliable', action='store_true')
parser.add_argument('--rliable_num_reps', type=int, default=20000)
parser.add_argument('--side_legend', action='store_true')
args = parser.parse_args()

fig_name = f"{args.plot}_envs_avg"
plot_name = fig_name

between_task_mean_std, valid_algoss, plot_titles = get_between_task_mean_std(
    args, fig_name=fig_name, get_plot_params=True)

root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name=fig_name)

algo_dir_names, algo_titles, multitask_algos, _, cmap_is = \
    plot_common.get_algo_defaults(plot=plot_name)

if args.plot in ['all_4_sep']:
    num_plots = 4
else:
    num_plots = 1

fig_shape, plot_size, num_stds, font_size, _, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline, _ = \
    plot_common.get_fig_defaults(num_plots=num_plots)
# num_stds = 0.25

#####################################################################################################################

if args.force_vert_squish:
    plot_size[0] = 4.2
    font_size += 4

if args.bigger_labels:
    font_size += 2
    linewidth *= 2

# pretty plotting, allow tex
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# make each plot separately, if we're doing a multiplot
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])

if hasattr(r_axes, 'shape'):
    r_axes_flat = r_axes.flatten()
else:
    r_axes_flat = [r_axes]

# r_ax = r_axes
for plot_i, r_ax in enumerate(r_axes_flat):
    valid_algos = valid_algoss[plot_i]
    plot_title = plot_titles[plot_i]

    # plot the means
    for algo in valid_algos:

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
                # elif algo == 'sqil-rnd':
                #     label = "RND"
                else:
                    label = plot_common.ALGO_TITLE_DICT[algo]['title']
            else:
                label = plot_common.ALGO_TITLE_DICT[algo]['title']
        else:
            label = None

        mean = between_task_mean_std[plot_title][algo]['means']
        std = between_task_mean_std[plot_title][algo]['stds']
        norm_eval_steps = between_task_mean_std[plot_title][algo]['norm_eval_steps']
        if args.use_rliable:
            mean = between_task_mean_std[plot_title][algo]['iqm_scores']
            iqm_cis = between_task_mean_std[plot_title][algo]['iqm_cis']

        r_ax.plot(norm_eval_steps, mean, label=label, color=color, linewidth=linewidth, linestyle=line_style)

        if args.stddev_type != 'none':
            if args.use_rliable:
                r_ax.fill_between(norm_eval_steps, iqm_cis[:, 0], iqm_cis[:, 1], facecolor=color,
                                alpha=std_alpha)
            else:
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
    if args.side_legend:
        ncol = 1
        if len(valid_algos) == 4:
            bbox_to_anchor = (1.15, .15)
        elif len(valid_algos) == 5:
            bbox_to_anchor = (1.2, .05)
        fsize = font_size - 2
    else:
        ncol = 2
        if args.force_vert_squish:
            if args.bigger_labels:
                bbox_to_anchor = (0.45, -.61)
            else:
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
if args.use_rliable:
    fig_name = f"rliable_{fig_name}"
if args.side_legend:
    fig_name = f"sideleg_{fig_name}"

os.makedirs(fig_path, exist_ok=True)
r_fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')