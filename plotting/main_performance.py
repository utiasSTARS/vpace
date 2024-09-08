from argparse import Namespace
import time
import pickle
import os
import glob
import math

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import convolve1d

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import data_locations
from rce_env_data_locations import main_performance as rce_data_locations
from hand_dapg_data_locations import main_performance as hand_data_locations
from real_panda_data_locations import main_performance as real_data_locations
import common as plot_common
from avg_common import get_between_task_mean_std


#### Options ########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--reload_data', action='store_true')
parser.add_argument('--plot', type=str, default='main',
                    choices=['main', 'rce', 'hand', 'abl_expert', 'abl_alg', 'abl_dquant', 'rce_hand_theirs',
                             'abl_all', 'abl_exaug', 'hardest', 'hardest_4', 'real', 'hardest_5', 'panda_3_overall',
                             'panda_3_hardest_overall', 'best_4_overall', 'panda_2_and_avgs', 'abl_reg', 'abl_rew_model',
                             'abl_lambda', 'panda_2_and_all_avgs', 'all_sep', 'abl_dquant_lambda', 'all_avgs'])
parser.add_argument('--extra_name', type=str, default="")
parser.add_argument('--vertical_plot', action='store_true')
parser.add_argument('--force_vert_squish', action='store_true')
parser.add_argument('--constrained_layout', action='store_true')
parser.add_argument('--bigger_labels', action='store_true')
parser.add_argument('--custom_algo_list', type=str, default="",
                    choices=['overall', 'overall_and_ace', 'overall_and_ace_and_rnd', 'ace_variations',
                             'ace_variations_and_vpsqil', 'vp_variations', 'vpace_ace_sqil'])
parser.add_argument('--bottom_legend', action='store_true')
parser.add_argument('--use_rliable', action='store_true')
parser.add_argument('--rliable_num_reps', type=int, default=20000)
parser.add_argument('--table_timestep', type=int, default=300000)
parser.add_argument('--print_table', action='store_true')
parser.add_argument('--table_type', type=str, default='md', choices=['md', 'latex'])
parser.add_argument('--table_valid_algos', type=str, default='', choices=['', 'overall_and_ace_and_rnd'])
parser.add_argument('--one_row', action='store_true')
args = parser.parse_args()

# since this is the plot we're using
if args.plot in ['panda_2_and_avgs', 'panda_3_overall', 'panda_2_and_all_avgs', 'all_avgs']:
    args.custom_algo_list = 'overall_and_ace_and_rnd'
    # args.custom_algo_list = 'overall_and_ace'

fig_name = f"{args.plot}_performance"

root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name=fig_name)

task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use, \
    st_num_eval_steps_to_use, eval_intervals, eval_eps_per_task = \
    plot_common.get_task_defaults(plot=args.plot)

algo_dir_names, algo_titles, multitask_algos, valid_algos, cmap_is = \
    plot_common.get_algo_defaults(plot=args.plot)

if args.custom_algo_list != "":
    valid_algos = plot_common.CUSTOM_ALGO_LIST_DICT[args.custom_algo_list]

fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline, num_timesteps_mean = \
    plot_common.get_fig_defaults(num_plots=sum(valid_task), plot=args.plot)

include_expert_baseline = False  # not going to use
side_legend = not args.bottom_legend
if args.vertical_plot:
    fig_shape = [fig_shape[1], fig_shape[0]]

if args.plot == 'rce':
    data_locs = rce_data_locations
elif args.plot == 'hand':
    data_locs = hand_data_locations
elif args.plot == 'real':
    data_locs = real_data_locations
elif args.plot == 'rce_hand_theirs':
    data_locs = {**rce_data_locations, **hand_data_locations}
elif 'hardest' in args.plot or 'panda_3' in args.plot or 'best' in args.plot or 'avgs' in args.plot or 'all' in args.plot:
    # data_locs = {**data_locations.main, **rce_data_locations, **hand_data_locations}
    data_locs = {**data_locations.main, **rce_data_locations, **hand_data_locations, **real_data_locations}
else:
    data_locs = getattr(data_locations, args.plot)
#####################################################################################################################

# pretty plotting, allow tex
# plt.rcParams.update({"font.family": "serif", "font.serif": "Times", "text.usetex": True, "pgf.rcfonts": False})
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

if args.one_row:
    fig_shape = [1, len(task_dir_names)]

if args.force_vert_squish:
    if args.constrained_layout:
        plot_size = [3.2, 2.7]
    else:
        plot_size[0] = 4.2
    font_size += 2

if 'abl' in args.plot and side_legend:
    plot_size[0] -= 0.7

if args.plot == 'panda_2_and_all_avgs':
    fig_shape = [1, 5]
    plot_size[0] -= .7

if args.plot == 'all_avgs':
    fig_shape = [1, 3]
    plot_size[0] -= .7

if args.bigger_labels:
    font_size += 2
    linewidth *= 2

s_fig, s_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]],
                             constrained_layout=args.constrained_layout)
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]],
                             constrained_layout=args.constrained_layout)

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

# get between task data if necessary
if 'panda_3_overall' in args.plot or 'avgs' in args.plot:
    # data_path = os.path.join(fig_path, 'data')
    # data_file = os.path.join(data_path, 'between_task_mean_std.pkl')

    # if args.reload_data:
    between_task_args = Namespace()
    # between_task_args.plot = 'all_4_sep'
    if args.plot == 'panda_2_and_avgs':
        between_task_args.plot = 'panda_sawyer_sep'
    elif args.plot == 'panda_2_and_all_avgs' or args.plot == 'all_avgs':
        between_task_args.plot = 'all_3_sep'
    elif args.plot == 'panda_3_overall':
        between_task_args.plot = 'main'
    else:
        raise NotImplementedError(f"avgs not implemented for plot arg {args.plot}")
    between_task_args.real_x_axis = True
    between_task_args.stddev_type = 'by_task'
    between_task_args.use_rliable = args.use_rliable
    between_task_args.rliable_num_reps = args.rliable_num_reps
    between_task_args.reload_data = args.reload_data
    between_task_mean_std = get_between_task_mean_std(
        between_task_args, fig_name=fig_name, valid_algos_custom=valid_algos)

    #     os.makedirs(data_path, exist_ok=True)
    #     pickle.dump(between_task_mean_std, open(data_file, 'wb'))
    # else:
    #     between_task_mean_std = pickle.load(open(data_file, 'rb'))

    # make dummy ones for this to make running easier..ignored later anyways
    for task in between_task_mean_std.keys():
        all_successes[task] = {}
        all_returns[task] = {}
        for algo in valid_algos:
            all_successes[task][algo] = None
            all_returns[task][algo] = None

def rm_lead_zero(f):
    # Convert to string and remove the leading zero if necessary
    # return f"{f:.2f}".lstrip('0') if f < 1 and f > -1 else f"{f:.2f}"
    return f"{f:.2f}" if f < 1 and f > -1 else f"{f:.2f}"

if args.print_table:
    if args.table_valid_algos == '':
        table_valid_algos = valid_algos
    else:
        table_valid_algos = plot_common.CUSTOM_ALGO_LIST_DICT[args.table_valid_algos]

    data_path = os.path.join(fig_path, 'data', 'rliable')
    lines = []
    if args.table_type == 'md':
        lines.append(f"|  | {' | '.join(plot_common.ALGO_TITLE_DICT[algo]['title'] for algo in table_valid_algos)} |")
        lines.append(f"| - |{' - |' * len(table_valid_algos)}")
        print(f"|  | {' | '.join(plot_common.ALGO_TITLE_DICT[algo]['title'] for algo in table_valid_algos)} |")  # header line
        print(f"| - |{' - |' * len(table_valid_algos)}")
    else:
        lines.append(f"& {' & '.join(plot_common.ALGO_TITLE_DICT[algo]['title'] for algo in table_valid_algos)} \\\\")
        lines.append('\\midrule')
        print(f"{' & '.join(plot_common.ALGO_TITLE_DICT[algo]['title'] for algo in table_valid_algos)} \\\\")
        print('\\midrule')
    for task_i, (task_title, task) in enumerate(zip(task_titles, task_dir_names)):
        if args.table_type == 'md':
            line = f"| {task_title} |"
        else:
            line = f" {task_title} &"
        for algo_i, algo in enumerate(table_valid_algos):
            algo_title = plot_common.ALGO_TITLE_DICT[algo]['title']
            if 'Main Tasks' in task:
                # all_algo_data = between_task_mean_std[task]
                # for algo in valid_algos:
                data = between_task_mean_std[task][algo]
                iqm_scores = data['iqm_scores']
                iqm_cis = data['iqm_cis'].T
            else:
                eval_interval = eval_intervals[task_i]
                # data = all_successes[task]
                # for algo in valid_algos:
                if args.plot in ['abl_reg', 'abl_rew_model', 'rce', 'hand']:
                    data_file = os.path.join(data_path, f'r-{task}-{algo}.pkl')
                else:
                    data_file = os.path.join(data_path, f's-{task}-{algo}.pkl')
                data = pickle.load(open(data_file, 'rb'))
                iqm_scores = data['iqm_scores']
                iqm_cis = data['iqm_cis']
                # algo_title = plot_common.ALGO_TITLE_DICT[algo]['title']

            if 'human' in task:
                iqm_scores /= 1000
                iqm_cis /= 1000
                iqm_scores = np.maximum(iqm_scores, 0)
                iqm_cis = np.maximum(iqm_cis, 0)

            # data_i = args.table_timestep // eval_interval

            # TODO potentially take data as an average across multiple timesteps here

            data_i = len(iqm_scores) // 2
            if args.table_type == 'md':
                line += f" {iqm_scores[data_i]:.2f} [{iqm_cis[0, data_i]:.2f}, {iqm_cis[1, data_i]:.2f}] |"
            else:
                # score_str = f"{iqm_scores[data_i]:.2f}".lstrip('0')
                if algo_i < len(table_valid_algos) - 1:
                    line += f" {rm_lead_zero(iqm_scores[data_i])} [{rm_lead_zero(iqm_cis[0, data_i])}, "\
                            f"{rm_lead_zero(iqm_cis[1, data_i])}] &"
                else:
                    line += f" {rm_lead_zero(iqm_scores[data_i])} [{rm_lead_zero(iqm_cis[0, data_i])}, "\
                            f"{rm_lead_zero(iqm_cis[1, data_i])}] \\\\"
        print(line)
        lines.append(line)
    table_file = "table.md" if args.table_type == 'md' else "latex-table.txt"
    with open(os.path.join(fig_path, table_file), 'w') as f:
        for l in lines:
            f.write(f"{l}\n")
    import ipdb; ipdb.set_trace()

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
        if task not in plot_common.AVGS_TASK_LIST:
            if algo not in all_successes[task].keys():
                print(f"No data for algo {algo} and task {task}, skipping")
                continue

        for ax_str, ax, task_algo_data in zip(['s', 'r'], [s_ax, r_ax], [all_successes[task][algo], all_returns[task][algo]]):
            # try:
            if task in plot_common.AVGS_TASK_LIST:
                mean = between_task_mean_std[task][algo]['means']
                std = between_task_mean_std[task][algo]['stds']

                if args.use_rliable:
                    iqm_scores = between_task_mean_std[task][algo]['iqm_scores']
                    iqm_cis = between_task_mean_std[task][algo]['iqm_cis'].T
            else:
                if args.use_rliable and not task in real_data_locations:
                    data_path = os.path.join(fig_path, 'data', 'rliable')
                    os.makedirs(data_path, exist_ok=True)
                    data_file = os.path.join(data_path, f'{ax_str}-{task}-{algo}.pkl')
                    if args.reload_data:
                        # shape for multitask raw is seed, eval step, aux task, eval ep
                        # shape for sample efficiency curve from rliable is (seed x num envs x eval step)
                        # for single task, num envs will just be 1
                        if algo in multitask_algos:
                            per_seed_scores = task_algo_data['raw'].mean(axis=-1)[..., main_task_i[task_i]]
                        else:
                            per_seed_scores = task_algo_data['raw'].mean(axis=-1)

                        # for smoothing, convolve the raw data across time dimension instead of how we do it without rliable
                        if num_timesteps_mean > 1:
                            convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
                            per_seed_scores_smoothed = convolve1d(per_seed_scores, convolv_op, axis=-1, mode='nearest')
                            per_seed_scores = per_seed_scores_smoothed

                        # now need a dummy axis for num envs
                        per_seed_scores = np.expand_dims(per_seed_scores, axis=1)
                        scores_dict = {'dummy': per_seed_scores}
                        iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                                                    for frame in range(scores.shape[-1])])
                        print(f"Starting 95% Bootstrap CIs calc for {ax_str} plot, task {task}, algo {algo}")
                        ci_calc_start = time.time()
                        iqm_scores, iqm_cis = rly.get_interval_estimates(scores_dict, iqm, reps=args.rliable_num_reps)
                        print(f"Finished 95%% Bootstrap CIs calc for task {task}, algo {algo}, took {time.time() - ci_calc_start:.3f}s")
                        iqm_scores = iqm_scores['dummy']
                        iqm_cis = iqm_cis['dummy']

                        # save the data for more quickly recreating the figure for format-only fixes
                        data = {'iqm_scores': iqm_scores, 'iqm_cis': iqm_cis}
                        pickle.dump(data, open(data_file, 'wb'))
                    else:
                        data = pickle.load(open(data_file, 'rb'))
                        iqm_scores = data['iqm_scores']
                        iqm_cis = data['iqm_cis']

                if algo in multitask_algos:
                    mean = task_algo_data['mean'][..., main_task_i[task_i]]
                    std = task_algo_data['std'][..., main_task_i[task_i]]
                else:
                    mean = task_algo_data['mean']
                    std = task_algo_data['std']

                if num_timesteps_mean > 1 and not task in real_data_locations:
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
                # if 'no-vp' in algo:
                #     line_style = '--'
                # else:
                #     line_style = '-'
            # elif 'theirs' in plot_common.ALGO_TITLE_DICT[algo]['title']:
            #     line_style = '-.'
            else:
                line_style = '--'
                # if 'no-vp' in algo:
                #     line_style = ':'
                # else:
                #     line_style = '-.'

            x_vals = np.array(range(eval_intervals[task_i], eval_intervals[task_i] * len(mean) + 1,
                                    eval_intervals[task_i] * subsample_rate))
            x_vals = x_vals / x_val_scale

            if args.plot == 'hand':
                label = algo_titles[algo_i] if task_i == 4 else ""
            else:
                label = algo_titles[algo_i] if task_i == 0 else ""

            # hardcode replace VP-DAC with DAC, VPACE-SQIL with VPACE
            if 'panda_3' in args.plot or 'avgs' in args.plot or 'real' in args.plot:
                if task_i == 0:
                    if algo == 'disc':
                        label = "DAC"
                    elif algo == 'multi-sqil':
                        label = "VPACE"
                    elif algo == 'multi-sqil-no-vp':
                        label = "ACE"
            elif 'abl_rew_model' in args.plot:
                if task_i == 0:
                    if algo == 'disc':
                        label = "DAC"

            if args.use_rliable and not task in real_data_locations:
                ax.plot(x_vals, iqm_scores, label=label,
                        color=cmap(cmap_is[algo_i]), linewidth=linewidth, linestyle=line_style)
                ax.fill_between(x_vals, iqm_cis[0], iqm_cis[1], facecolor=cmap(cmap_is[algo_i]),
                                alpha=std_alpha)
            else:
                ax.plot(x_vals, mean, label=label,
                        color=cmap(cmap_is[algo_i]), linewidth=linewidth, linestyle=line_style)
                ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(cmap_is[algo_i]),
                                alpha=std_alpha)

            if task == 'Sawyer Main Tasks':
                ax.set_ylabel('Return (norm)', fontsize=font_size + 2)

            # except Exception as e:
            #     print(f"Error when processing task {task}, algo: {algo}:")
            #     print(e)
            #     print("skipping in plotting")

    # pretty up plot, add expert baselines
    for plot_type, ax in zip(['s', 'r'], [s_ax, r_ax]):

        # pretty
        ax.grid(alpha=0.5)
        # if args.plot == 'abl_expert':
        #     ax.set_title("Reward Variations", fontsize=font_size)
        if args.plot == 'abl_alg':
            ax.set_title("Algorithm Variations", fontsize=font_size)
        # elif args.plot == 'abl_dquant':
        #     ax.set_title("Expert Quantity Variations", fontsize=font_size)

        else:
            ax.set_title(task_titles[task_i], fontsize=font_size)


        if plot_type == 's':
            if 'hardest' in args.plot and ('sawyer' in task or 'human' in task):
                all_task_settings = {**plot_common.RCE_TASK_SETTINGS, **plot_common.HAND_TASK_SETTINGS}
                ax.set_ylim(*all_task_settings[task]["return_ylims"])
            else:
                if args.plot == 'real':
                    ax.set_ylim(-.03, 1.03)
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

        if args.bigger_labels:
            ax.tick_params(labelsize=font_size-7)
        else:
            ax.tick_params(labelsize=font_size-8)

        # set y ticks to smaller scale
        if args.plot == 'hand' and plot_type == 'r':
            from matplotlib.axis import Axis
            import matplotlib.ticker as ticker
            def divide_formatter(x, pos):
                return f"{x * .001}"

            Axis.set_major_formatter(ax.yaxis, ticker.FuncFormatter(divide_formatter))


fig_path += args.extra_name

for fig, fig_name in zip([s_fig, r_fig], ['s_fig.pdf', 'r_fig.pdf']):

    if sum(valid_task) == 1:
        label_font_size = font_size - 2
        y_label_pad = 10.0
    else:
        label_font_size = font_size + 2
        y_label_pad = 8.0

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if args.constrained_layout:
        if args.bigger_labels:
            fig.supxlabel(r"Env. Steps ($\times$100k)", fontsize=label_font_size)
        else:
            fig.supxlabel(r"Environment Steps ($\times$100k)", fontsize=label_font_size)
    else:
        if args.bigger_labels:
            ax.set_xlabel(r"Env. Steps ($\times$100k)", fontsize=label_font_size)
        else:
            ax.set_xlabel(r"Environment Steps ($\times$100k)", fontsize=label_font_size)
    # ax.xaxis.set_label_coords(.57, -.15)  # if we have the 10^6 scientific notation

    if 's_fig' in fig_name:
        if args.constrained_layout:
            # fig.text(-.05, 0.5, "Success Rate", va='center', rotation='vertical')
            fig.supylabel("Success Rate", fontsize=label_font_size)
        else:
            if args.plot == 'hardest':
                s_axes[0, 0].set_ylabel("Success Rate", fontsize=label_font_size)
                s_axes[1, 0].set_ylabel("Episode Return", fontsize=label_font_size, labelpad=8.0)
            elif args.plot == 'hardest_4':
                ax.set_ylabel("Success Rate or Return", fontsize=label_font_size - 3, labelpad=y_label_pad)
            else:
                ax.set_ylabel("Success Rate", fontsize=label_font_size, labelpad=y_label_pad)

    else:
        if args.constrained_layout:
            if args.plot == "hand":
                fig.supylabel(r"Episode Return ($\times$1k)", fontsize=label_font_size)
            else:
                fig.supylabel("Episode Return", fontsize=label_font_size)
        else:
            if args.plot == 'hand':
                ax.set_ylabel(r"Episode Return ($\times$1k)", fontsize=label_font_size, labelpad=y_label_pad)
            else:
                ax.set_ylabel("Episode Return", fontsize=label_font_size, labelpad=y_label_pad)

    if args.plot == 'hardest_4':
        pass
    elif args.plot == 'hardest':
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.99, 0.075))
    elif fig_shape == [1, 2] or fig_shape == [1, 1]:
        if len(valid_algos) > 2:
            num_col = np.ceil(len(valid_algos) / 2)
        else:
            num_col = len(valid_algos)
        if side_legend:
            bbta = (1.15, 0.25)
            if args.plot == 'abl_exaug':
                bbta = (1.22, 0.2)
            elif args.plot == 'abl_expert':
                bbta = (1.19, 0.22)
            elif args.plot == 'abl_dquant_lambda':
                bbta = (1.19, 0.17)
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                        ncol=1, bbox_to_anchor=bbta)
        else:
            if len(valid_algos) == 2:
                bbta = (0.5, -.2)
            else:
                # bbta = (0.5, -.53)
                bbta = (0.5, -.34)
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                        ncol=num_col, bbox_to_anchor=bbta)
    elif fig_shape == [1, 3] and len(valid_algos) > 3:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                        ncol=int(math.ceil((len(valid_algos) + 1)) / 2),
                        bbox_to_anchor=(0.5, -0.33))
    elif fig_shape == [2, 4] and sum(valid_task) == 8:
        # legend underneath
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.11))
    elif fig_shape[0] == 3:
        print("TODO need to position legend properly for this case!!!")
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.81, 0.075))
    elif fig_shape == [2, 3]:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(1.15, 0.2))
    elif fig_shape == [2, 1]:
        if args.bigger_labels:
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.18))
        else:
            fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.16))
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
            if args.constrained_layout:
                valid_algos_bbta_dict = {4: (1.08, 0.23), 5: (1.08, 0.18)}
                fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1,
                        bbox_to_anchor=valid_algos_bbta_dict[len(valid_algos)])
            else:
                fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1,
                           bbox_to_anchor=(0.98, 0.15))
        else:
            if args.constrained_layout:
                fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                        ncol=int(math.ceil((len(valid_algos) + 1))),
                        bbox_to_anchor=(0.5, -0.21))
            else:
                fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                        ncol=int(math.ceil(len(valid_algos))),
                        bbox_to_anchor=(0.5, -0.3))

    if args.vertical_plot:
        fig_name = f"vert_{fig_name}"
    if args.force_vert_squish:
        fig_name = f"squish_{fig_name}"
    if args.constrained_layout:
        fig_name = f"constrained_{fig_name}"
    if not side_legend:
        fig_name = f"bottom_legend_{fig_name}"
    if args.use_rliable:
        fig_name = f"rliable_{fig_name}"
    if args.one_row:
        fig_name = f"one_row_{fig_name}"

    if args.custom_algo_list != "":
        fig_name = f"{args.custom_algo_list}_{fig_name}"

    # if args.custom_algo_list == 'ace_variations':
    #     fig_name = f"ace_var_{fig_name}"
    # if args.custom_algo_list == 'ace_variations_and_vpsqil':
    #     fig_name = f"ace_var_vpsqil_{fig_name}"
    # if args.custom_algo_list == 'vp_variations':
    #     fig_name = f"vp_var_{fig_name}"

    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')