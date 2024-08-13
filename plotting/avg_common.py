from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import math
import ast
import argparse
import copy
import tqdm
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import tqdm.tk

import data_locations
from rce_env_data_locations import main_performance as rce_data_locations
from hand_dapg_data_locations import main_performance as hand_data_locations
from real_panda_data_locations import main_performance as real_data_locations
import common as plot_common


# see avg_perf.py for args
def get_between_task_mean_std(args: Namespace, fig_name, valid_algos_custom=None, get_plot_params=False):
    if args.use_rliable:
        assert args.real_x_axis, "use_rliable only implemented for real x axis for now"

    # fig_name = f"{args.plot}_envs_avg"
    plot_name = fig_name

    if args.plot == 'panda_sawyer_sep':
        all_valid_task_settings = {**plot_common.PANDA_TASK_SETTINGS, **plot_common.RCE_TASK_SETTINGS}
        valid_algoss = []; valid_task_settingss = []; plot_titles = []; num_timesteps_means = []
        for ptype in ['main', 'rce']:
            valid_algoss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_algos'])
            valid_task_settingss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_task_settings'])
            plot_titles.append(plot_common.AVG_ENVS_DICT[ptype]['title'])
            num_timesteps_means.append(plot_common.AVG_ENVS_DICT[ptype]['num_timesteps_mean'])
    elif args.plot == 'all_3_sep':
        all_valid_task_settings = plot_common.AVG_ENVS_DICT['all']['valid_task_settings']
        valid_algoss = []; valid_task_settingss = []; plot_titles = []; num_timesteps_means = []
        for ptype in ['main', 'rce', 'hand']:
            valid_algoss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_algos'])
            valid_task_settingss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_task_settings'])
            plot_titles.append(plot_common.AVG_ENVS_DICT[ptype]['title'])
            num_timesteps_means.append(plot_common.AVG_ENVS_DICT[ptype]['num_timesteps_mean'])
            if 'relocate-human-v0' in valid_task_settingss[-1]:
                del valid_task_settingss[-1]['relocate-human-v0']  # since all zeros for everything, and missing some results
    elif args.plot == 'all_4_sep':
        all_valid_task_settings = plot_common.AVG_ENVS_DICT['all']['valid_task_settings']
        valid_algoss = []; valid_task_settingss = []; plot_titles = []; num_timesteps_means = []
        for ptype in ['main', 'rce', 'hand_orig', 'hand_dp']:
            valid_algoss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_algos'])
            valid_task_settingss.append(plot_common.AVG_ENVS_DICT[ptype]['valid_task_settings'])
            plot_titles.append(plot_common.AVG_ENVS_DICT[ptype]['title'])
            num_timesteps_means.append(plot_common.AVG_ENVS_DICT[ptype]['num_timesteps_mean'])
            if 'relocate-human-v0' in valid_task_settingss[-1]:
                del valid_task_settingss[-1]['relocate-human-v0']  # since all zeros for everything, and missing some results
    else:
        all_valid_task_settings = plot_common.AVG_ENVS_DICT[args.plot]['valid_task_settings']
        valid_algoss = [plot_common.AVG_ENVS_DICT[args.plot]['valid_algos']]
        valid_task_settingss = [plot_common.AVG_ENVS_DICT[args.plot]['valid_task_settings']]
        plot_titles = [plot_common.AVG_ENVS_DICT[args.plot]['title']]
        num_timesteps_means = [plot_common.AVG_ENVS_DICT[args.plot]['num_timesteps_mean']]

        if 'relocate-human-v0' in valid_task_settingss[0]:
            del valid_task_settingss[0]['relocate-human-v0']  # since all zeros for everything, and missing some results

    if valid_algos_custom is not None:
        valid_algoss = []
        for _ in range(len(valid_task_settingss)):
            valid_algoss.append(valid_algos_custom)

    panda_task_settings = {**plot_common.PANDA_TASK_SETTINGS}
    real_task_settings = {**plot_common.REAL_PANDA_TASK_SETTINGS}

    root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
        plot_common.get_path_defaults(fig_name=fig_name)

    algo_dir_names, algo_titles, multitask_algos, _, cmap_is = \
        plot_common.get_algo_defaults(plot=plot_name)

    if args.plot in ['all_4_sep']:
        num_plots = 4
    elif args.plot in ['all_3_sep']:
        num_plots = 3
    elif args.plot in ['panda_sawyer_sep']:
        num_plots = 2
    else:
        num_plots = 1

    fig_shape, plot_size, num_stds, font_size, _, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
        include_expert_baseline, _ = \
        plot_common.get_fig_defaults(num_plots=num_plots)
    num_stds = 0.25

    # hardcoded options for real_x_axis
    x_val_scale = 100000

    #####################################################################################################################

    data_path = os.path.join(fig_path, 'data')
    data_file = os.path.join(data_path, 'between_task_mean_std.pkl')

    if args.reload_data:

        # directly load the data from sawyer and hand plots saved data, so we don't accidentally use diff data..and to save time
        all_returns = {}; all_successes = {}
        for data_fig_name in ['main_performance', 'rce_performance', 'hand_performance', 'real_performance']:
            data = pickle.load(open(os.path.join(root_dir, 'figures', data_fig_name, 'data', 'data.pkl'), 'rb'))
            for r_type in ['all_returns', 'all_successes']:
                for k, v in data[r_type].items():
                    if k in all_valid_task_settings:
                        locals()[r_type][k] = v

        return_data = {}

        for plot_i in range(num_plots):

            valid_algos = valid_algoss[plot_i]
            valid_task_settings = valid_task_settingss[plot_i]
            plot_title = plot_titles[plot_i]
            num_timesteps_mean = num_timesteps_means[plot_i]

            return_data[plot_title] = {}

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

                    if algo not in across_task_rets: across_task_rets[algo] = {'means': [], 'stds': [], 'per_seed_score': []}

                    if args.real_x_axis:
                        # 1. interpolate everything to shortest eval intervals, keeping true max timestep the same

                        # interpolate everything, but this is a non-op for tasks that already have min_eval_interval
                        eval_interval = all_valid_task_settings[task]['eval_intervals']
                        orig_final_step = eval_interval * num_eval_steps + 1
                        orig_x = np.arange(eval_interval, orig_final_step, eval_interval)
                        new_x = np.arange(min_eval_interval, orig_final_step, min_eval_interval)

                        # at this point, all_successes/all_rets_norm shape is (seed, eval timesteps, ep reward/success)
                        if task not in panda_task_settings and task not in real_task_settings:
                            task_rets_per_seed_score = all_rets_norm[task][algo]['raw'].mean(axis=-1)
                            task_rets_mean = all_rets_norm[task][algo]['mean']
                            task_rets_std = all_rets_norm[task][algo]['std']
                        else:
                            task_rets_per_seed_score = all_successes[task][algo]['raw'].mean(axis=-1)
                            task_rets_mean = all_successes[task][algo]['mean']
                            task_rets_std = all_successes[task][algo]['std']

                        # now per seed score is (seed, eval_timesteps)
                        if len(task_rets_per_seed_score.shape) == 1:  # only 1 seed, only true for 2 real envs, fake 5 seeds
                            task_rets_per_seed_score = np.tile(np.expand_dims(task_rets_per_seed_score, 1), [1, 5]).T

                        per_seed_score_interp_f = interp1d(orig_x, task_rets_per_seed_score, axis=1, fill_value='extrapolate')
                        across_task_rets[algo]['per_seed_score'].append(per_seed_score_interp_f(new_x))
                        across_task_rets[algo]['means'].append(np.interp(new_x, orig_x, task_rets_mean))
                        across_task_rets[algo]['stds'].append(np.interp(new_x, orig_x, task_rets_std))

                    else:

                        # then normalize the eval timesteps with interpolation and the max_eval_steps
                        if num_eval_steps < max_eval_steps:
                            eval_interval = all_valid_task_settings[task]['eval_intervals']

                            orig_x = np.arange(1, num_eval_steps + 1) * eval_interval
                            new_x = np.arange(1, max_eval_steps + 1) * eval_interval

                            # expand orig_x to have same min and max as new x
                            orig_x_new_max = (new_x.max() - new_x.min()) / (orig_x.max() - orig_x.min()) * \
                                            (orig_x - orig_x.min()) + new_x.min()

                            if task not in panda_task_settings and task not in real_task_settings:
                                all_rets_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['mean'])
                                all_rets_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_rets_norm[task][algo]['std'])

                            else:
                                all_sucs_norm_interp[task][algo]['mean'] = np.interp(new_x, orig_x_new_max, all_successes[task][algo]['mean'])
                                all_sucs_norm_interp[task][algo]['std'] = np.interp(new_x, orig_x_new_max, all_successes[task][algo]['std'])

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
                    if num_timesteps_mean > 1 and task not in real_task_settings:
                        convolv_op = np.ones(num_timesteps_mean) / num_timesteps_mean
                        across_task_rets_smoothed = {'means': [], 'stds': [], 'per_seed_score': []}
                        for task_algo_per_seed_score, task_algo_means, task_algo_stds in \
                                zip(across_task_rets[algo]['per_seed_score'], across_task_rets[algo]['means'],
                                    across_task_rets[algo]['stds']):
                            across_task_rets_smoothed['per_seed_score'].append(
                                convolve1d(task_algo_per_seed_score, convolv_op, mode='nearest'))
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
                        if 'human' in task and 'hand_eval_cutoff_env_step' in plot_common.AVG_ENVS_DICT[args.plot]:
                            eval_cutoff_env_step = plot_common.AVG_ENVS_DICT[args.plot]['hand_eval_cutoff_env_step']
                        elif '_0' in task and 'panda_eval_cutoff_env_step' in plot_common.AVG_ENVS_DICT[args.plot]:
                            eval_cutoff_env_step = plot_common.AVG_ENVS_DICT[args.plot]['panda_eval_cutoff_env_step']
                        else:
                            eval_cutoff_env_step = plot_common.AVG_ENVS_DICT[args.plot]['eval_cutoff_env_step']
                        max_eval = eval_cutoff_env_step
                        max_new_eval_steps = int(eval_cutoff_env_step / min_eval_interval)
                        mean = mean[:max_new_eval_steps]
                        std = std[:max_new_eval_steps]

                    if args.use_rliable:
                        iqm = lambda x: np.array([metrics.aggregate_iqm(x)])
                        iqm_scores = np.zeros(max_new_eval_steps,)
                        iqm_cis = np.zeros((max_new_eval_steps, 2))

                        print(f"Calculating 95% CI with rliable for avg plot {plot_title}, {algo}")
                        for i in tqdm.trange(max_new_eval_steps):
                            # generate rliable-friendly array for this timestep
                            timestep_scores = []
                            for task_i in range(len(across_task_rets[algo]['per_seed_score'])):
                                if across_task_rets[algo]['per_seed_score'][task_i].shape[-1] > i:
                                    timestep_scores.append(across_task_rets[algo]['per_seed_score'][task_i][:, i])

                            # now shape is (num_valid_tasks, num_seeds), rliable wants (seed, num envs), so transpose
                            timestep_scores = np.array(timestep_scores).T
                            scores_dict = {'dummy': timestep_scores}
                            iqm_score, iqm_ci = rly.get_interval_estimates(scores_dict, iqm, reps=args.rliable_num_reps)
                            iqm_scores[i] = iqm_score['dummy']
                            iqm_cis[i] = iqm_ci['dummy'].flatten()

                    norm_eval_steps = np.arange(min_eval_interval, max_new_eval_steps * min_eval_interval + 1,
                                                min_eval_interval) / x_val_scale

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

                return_data[plot_title][algo] = {'means': mean, 'stds': std, 'norm_eval_steps': norm_eval_steps}
                if args.use_rliable:
                    return_data[plot_title][algo]['iqm_scores'] = iqm_scores
                    return_data[plot_title][algo]['iqm_cis'] = iqm_cis

        os.makedirs(data_path, exist_ok=True)
        pickle.dump(return_data, open(data_file, 'wb'))
    else:
        return_data = pickle.load(open(data_file, 'rb'))

    if get_plot_params:
        return return_data, valid_algoss, plot_titles
    else:
        return return_data
