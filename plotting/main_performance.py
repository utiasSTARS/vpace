import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import math
import argparse

import data_locations
from rce_env_data_locations import main_performance as rce_data_locations
from hand_dapg_data_locations import main_performance as hand_data_locations
import common as plot_common


#### Options ########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--reload_data', action='store_true')
parser.add_argument('--plot', type=str, default='main',
                    choices=['main', 'rce', 'hand', 'abl_expert', 'abl_alg', 'abl_dquant', 'rce_hand_theirs',
                             'abl_all', 'abl_exaug', 'hardest', 'hardest_4'])
parser.add_argument('--extra_name', type=str, default="")
args = parser.parse_args()

fig_name = f"{args.plot}_performance"

root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name=fig_name)

task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use, \
    st_num_eval_steps_to_use, eval_intervals, eval_eps_per_task = \
    plot_common.get_task_defaults(plot=args.plot)

# algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task, valid_algos, cmap_is = \
algo_dir_names, algo_titles, multitask_algos, valid_algos, cmap_is = \
    plot_common.get_algo_defaults(plot=args.plot)

fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline, num_timesteps_mean = \
    plot_common.get_fig_defaults(num_plots=sum(valid_task), plot=args.plot)

include_expert_baseline = False  # not going to use
side_legend = True

if args.plot == 'rce':
    data_locs = rce_data_locations
elif args.plot == 'hand':
    data_locs = hand_data_locations
elif args.plot == 'rce_hand_theirs':
    data_locs = {**rce_data_locations, **hand_data_locations}
elif 'hardest' in args.plot:
    data_locs = {**data_locations.main, **rce_data_locations, **hand_data_locations}
else:
    data_locs = getattr(data_locations, args.plot)
#####################################################################################################################

# pretty plotting, allow tex
# plt.rcParams.update({"font.family": "serif", "font.serif": "Times", "text.usetex": True, "pgf.rcfonts": False})
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

s_fig, s_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])

if sum(valid_task) == 1:
    s_axes_flat = [s_axes]
    r_axes_flat = [r_axes]
else:
    s_axes_flat = s_axes.flatten()
    r_axes_flat = r_axes.flatten()

# delete subplots we don't need
if len(s_axes_flat) > sum(valid_task):
    for i in range(len(s_axes_flat) - 1, sum(valid_task) - 1, -1):
        s_fig.delaxes(s_axes_flat[i])
        r_fig.delaxes(r_axes_flat[i])

# get returns and successes
all_returns, all_successes = plot_common.get_success_return(
    reload_data=args.reload_data,
    task_dir_names=task_dir_names,
    valid_task=valid_task,
    algo_dir_names=algo_dir_names,
    num_eval_steps_to_use=num_eval_steps_to_use,
    multitask_algos=multitask_algos,
    st_num_eval_steps_to_use=st_num_eval_steps_to_use,
    data_locations=data_locs,
    experiment_root_dir=experiment_root_dir,
    seeds=seeds,
    task_data_filenames=task_data_filenames,
    num_aux=num_aux,
    eval_eps_per_task=eval_eps_per_task,
    fig_path=fig_path,
    valid_algos=valid_algos
)

# skip even considering success rate for combined plot
if 'hardest' in args.plot:
    for task in all_successes.keys():
        if 'sawyer' in task or 'human' in task:
            all_successes[task] = all_returns[task]

skipped_tasks = 0
# plotting
for task_i, task in enumerate(task_dir_names):
    if not valid_task[task_i]:
        print(f"Task {task} set to false in valid_task, skipping in plotting")
        skipped_tasks += 1
        continue
    ax_i = task_i - skipped_tasks
    s_ax = s_axes_flat[ax_i]
    r_ax = r_axes_flat[ax_i]
    for algo_i, algo in enumerate(algo_dir_names):
        # if not valid_algos[algo_i]:
        if algo not in valid_algos:
            print(f"algo {algo} labelled as not valid, skipping.")
            continue
        if algo not in all_successes[task].keys():
            print(f"No data for algo {algo} and task {task}, skipping")
            continue
        for ax, task_algo_data in zip([s_ax, r_ax], [all_successes[task][algo], all_returns[task][algo]]):
            try:
                if algo in multitask_algos:
                    mean = task_algo_data['mean'][..., main_task_i[task_i]]
                    std = task_algo_data['std'][..., main_task_i[task_i]]
                else:
                    mean = task_algo_data['mean']
                    std = task_algo_data['std']

                if num_timesteps_mean > 1:
                    # shape for multitask is raw is seed, eval step, aux task, eval ep
                    # for single is seed, eval step, eval ep
                    smooth_means = []
                    smooth_stds = []
                    for eval_step in range(task_algo_data['raw'].shape[1]):
                        half_ts = num_timesteps_mean // 2
                        bottom_ind = max(0, eval_step - half_ts)
                        top_ind = min(task_algo_data['raw'].shape[1], eval_step + half_ts + 1)
                        if algo in multitask_algos:
                            seed_means = task_algo_data['raw'][:, bottom_ind:top_ind, main_task_i[task_i], :].mean(axis=(1,2))
                        else:
                            seed_means = task_algo_data['raw'][:, bottom_ind:top_ind, :].mean(axis=(1,2))
                        smooth_means.append(seed_means.mean())
                        smooth_stds.append(seed_means.std())
                    mean = np.array(smooth_means)
                    std = np.array(smooth_stds)

                if algo in multitask_algos:
                    line_style = '-'
                # elif 'theirs' in plot_common.ALGO_TITLE_DICT[algo]['title']:
                #     line_style = '-.'
                else:
                    line_style = '--'

                x_vals = np.array(range(eval_intervals[task_i], eval_intervals[task_i] * len(mean) + 1,
                                        eval_intervals[task_i] * subsample_rate))
                x_vals = x_vals / x_val_scale

                # test combining data at subsample rate to see if it reduces noise
                if subsample_rate > 1:
                    new_mean = []
                    new_std = []
                    for samp_start in range(0, task_algo_data['raw'].shape[1], subsample_rate):
                        new_samp = task_algo_data['raw'][:, samp_start:samp_start+subsample_rate, ...]
                        if algo in multitask_algos:
                            new_mean.append(new_samp.mean(axis=(1, -1))[:, main_task_i[task_i]].mean())
                            new_std.append(new_samp.mean(axis=(1, -1))[:, main_task_i[task_i]].std())
                        else:
                            new_mean.append(new_samp.mean(axis=(1, -1)).mean())
                            new_std.append(new_samp.mean(axis=(1, -1)).std())
                    mean = np.array(new_mean)
                    std = np.array(new_std)

                if args.plot == 'hand':
                    label = algo_titles[algo_i] if task_i == 4 else ""
                else:
                    label = algo_titles[algo_i] if task_i == 0 else ""

                ax.plot(x_vals, mean, label=label,
                        # color=cmap(algo_i), linewidth=linewidth, linestyle=line_style)
                        color=cmap(cmap_is[algo_i]), linewidth=linewidth, linestyle=line_style)
                # ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(algo_i),
                ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(cmap_is[algo_i]),
                                alpha=std_alpha)
            except Exception as e:
                print(f"Error when processing task {task}, algo: {algo}:")
                print(e)
                print("skipping in plotting")

    # pretty up plot, add expert baselines
    for plot_type, ax in zip(['s', 'r'], [s_ax, r_ax]):

        # pretty
        ax.grid(alpha=0.5)
        if args.plot == 'abl_expert':
            ax.set_title("Reward Variations", fontsize=font_size)
        elif args.plot == 'abl_alg':
            ax.set_title("Algorithm Variations", fontsize=font_size)
        elif args.plot == 'abl_dquant':
            ax.set_title("Expert Quantity Variations", fontsize=font_size)

        else:
            ax.set_title(task_titles[task_i], fontsize=font_size)

        # if args.plot == 'hardest_4':
        #     if task_i == 2:
        #         ax.set_ylabel('Episode Return', fontsize=font_size + 2)

        if plot_type == 's':
            if 'hardest' in args.plot and ('sawyer' in task or 'human' in task):
                all_task_settings = {**plot_common.RCE_TASK_SETTINGS, **plot_common.HAND_TASK_SETTINGS}
                ax.set_ylim(*all_task_settings[task]["return_ylims"])
            else:
                ax.set_ylim(-.01, 1.01)
        else:
            # if custom defined limits, set them
            if args.plot == 'rce':
                ax.set_ylim(*plot_common.RCE_TASK_SETTINGS[task]["return_ylims"])
            elif args.plot == 'hand':
                ax.set_ylim(*plot_common.HAND_TASK_SETTINGS[task]["return_ylims"])
            elif args.plot == 'rce_hand_theirs':
                all_task_settings = {**plot_common.RCE_TASK_SETTINGS, **plot_common.HAND_TASK_SETTINGS}
                ax.set_ylim(*all_task_settings[task]["return_ylims"])

        ax.grid(alpha=0.5, which='both')

        # set xticks based on eval intervals + num eval steps..also choose max between single and multi
        mt_nestu = num_eval_steps_to_use[task_i]
        st_nestu = st_num_eval_steps_to_use[task_i]
        max_eval = max(mt_nestu, st_nestu) * eval_intervals[task_i]
        max_tick = max_eval / x_val_scale
        tick_gap = max_tick / 5
        ax.set_xticks(np.arange(0.0, max_tick + .1, tick_gap))

        # set y ticks to smaller scale
        if args.plot == 'hand' and plot_type == 'r':
            from matplotlib.axis import Axis
            import matplotlib.ticker as ticker
            def divide_formatter(x, pos):
                return f"{x * .001}"

            Axis.set_major_formatter(ax.yaxis, ticker.FuncFormatter(divide_formatter))



fig_path += args.extra_name

for fig, fig_name in zip([s_fig, r_fig], ['s_fig.pdf', 'r_fig.pdf']):

    # if side_legend:
    #     fig_name = "sideleg_" + fig_name

    if sum(valid_task) == 1:
        label_font_size = font_size - 2
        y_label_pad = 2.0
    else:
        label_font_size = font_size + 2
        y_label_pad = 8.0

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # ax.set_xlabel("Environment Steps (millions)", fontsize=font_size)
    # ax.set_xlabel("Environment Steps (hundred thousands)", fontsize=font_size)
    ax.set_xlabel(r"Environment Steps ($\times$100k)", fontsize=label_font_size)
    # ax.xaxis.set_label_coords(.57, -.15)  # if we have the 10^6 scientific notation

    if 's_fig' in fig_name:
        if args.plot == 'hardest':
            s_axes[0, 0].set_ylabel("Success Rate", fontsize=label_font_size)
            s_axes[1, 0].set_ylabel("Episode Return", fontsize=label_font_size, labelpad=8.0)
        elif args.plot == 'hardest_4':
            ax.set_ylabel("Success Rate or Return", fontsize=label_font_size - 3, labelpad=y_label_pad)
        else:
            ax.set_ylabel("Success Rate", fontsize=label_font_size, labelpad=y_label_pad)
    else:
        if args.plot == 'hand':
            ax.set_ylabel(r"Episode Return ($\times$1000)", fontsize=label_font_size, labelpad=y_label_pad)
        else:
            ax.set_ylabel("Episode Return", fontsize=label_font_size, labelpad=y_label_pad)

    if args.plot == 'hardest_4':
        pass
    elif args.plot == 'hardest':
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.99, 0.075))
    elif fig_shape == [2, 4] and sum(valid_task) == 8:
        # legend underneath
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.11))
    elif fig_shape[0] == 3:
        print("TODO need to position legend properly for this case!!!")
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.81, 0.075))
    elif fig_shape == [2, 3]:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(1.03, 0.3))
    elif fig_shape[0] == 2:
        if len(valid_algos) == 6:
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.81, 0.075))
        else:
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.81, 0.005))
    elif sum(valid_task) == 1:
        if len(valid_algos) == 4:
            bbtoa = (0.475, -0.575)
        elif len(valid_algos) == 7:
            bbtoa = (0.475, -0.875)
        elif len(valid_algos) == 8:
            bbtoa = (0.475, -0.95)
        elif len(valid_algos) == 9:
            bbtoa = (0.475, -1.05)
        elif len(valid_algos) == 10:
            if side_legend:
                bbtoa = (1.32, -.025)
            else:
                bbtoa = (0.475, -1.05)
        else:
            bbtoa = (0.475, -0.5)
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-5, loc="lower center", ncol=1, bbox_to_anchor=bbtoa)
    else:
        if side_legend:
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.98, 0.15))
        else:
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                    ncol=int(math.ceil((len(algo_dir_names) + 1))),
                    bbox_to_anchor=(0.5, -0.3))

    fig.subplots_adjust(hspace=.35)
    # fig.tight_layout()
    # fig.subplots_adjust(top=.8, bottom=0.0)
    # plt.subplots_adjust(top=.65)

    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')