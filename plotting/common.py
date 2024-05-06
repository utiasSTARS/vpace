"""
Common things for many plotting files.
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict


RCE_REG_VS_CLASS_DICT = OrderedDict({
    'multi-sqil':{
        'title': 'VPACE (reg-loss)'
    },
    'sqil':{
        'title': 'VP-SQIL (reg-loss)'
    },
    'disc':{
        'title': 'VP-DAC (no abs. states)'
    },
    'rce_orig':{
        'title': 'RCE (theirs)'
    },
    'sqil_theirs':{
        'title': 'SQIL-class (theirs)'
    },
    'sqil_theirs_nstepoff':{
        'title': 'SQIL-class-no-nstep (theirs)'
    }
})


ALGO_TITLE_DICT = OrderedDict({
    'multi-sqil':{
        'title': 'VPACE-SQIL',
        'plots': {'main', 'rce', 'abl_expert', 'abl_alg', 'abl_dquant', 'hand', 'abl_all', 'abl_exaug', 'hardest', 'hardest_4'},
        'cmap_i': 0,
    },
    'multi-disc':{
        'title': 'VPACE-DAC',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4'},
        # 'plots': {'main', 'rce'},
        'cmap_i': 2,
    },
    'multi-sqil-no-vp':{
        'title': 'ACE-SQIL',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4'},
        'cmap_i': 6,
    },
    'multi-rce':{
        'title': 'ACE-RCE',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4'},
        'cmap_i': 4,
    },
    'sqil':{
        'title': 'VP-SQIL',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4', 'abl_all'},
        # 'plots': {'main', 'rce'},
        'cmap_i': 1,
    },
    'disc':{
        'title': 'VP-DAC',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4'},
        'cmap_i': 3,
    },
    'sqil-no-vp':{
        'title': 'SQIL',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4', 'rce_hand_theirs'},
        # 'plots': {'main', 'rce'},
        'cmap_i': 7,
    },
    'rce':{
        'title': 'RCE',
        'plots': {'main', 'rce', 'hand', 'hardest', 'hardest_4', 'rce_env_mods'},
        # 'plots': {'main', 'rce'},
        'cmap_i': 5,
    },
    'rce_orig':{
        'title': 'RCE (theirs)',
        'plots': {'re_vs_cl', 'rce_hand_theirs', 'rce_env_mods'},
        'cmap_i': 9,
    },
    'sqil_theirs':{
        'title': 'SQIL-BCE (theirs)',
        'plots': {'re_vs_cl', 'rce_hand_theirs'},
        'cmap_i': 11,
    },
    'sqil_theirs_nstepoff':{
        'title': 'SQIL-BCE-no-nstep (theirs)',
        'plots': {'re_vs_cl', 'rce_hand_theirs'},
        'cmap_i': 13,
    },
    'full_trajs':{
        'title': '+Full Trajectories',
        'plots': {'abl_expert', 'abl_all'},
        'cmap_i': 8,
    },
    'full_trajs_wa':{
        'title': '+Full Trajectories \\& Actions',
        'plots': {'abl_expert', 'abl_all'},
        'cmap_i': 17,
    },
    'full_trajs_st':{
        'title': 'VP-SQIL +Full Trajectories',
        'plots': {'abl_expert', 'abl_all'},
        'cmap_i': 14,
    },
    'sparse_rew':{
        'title': 'SAC-X (Sparse Rewards)',
        'plots': {'abl_expert', 'abl_all'},
        'cmap_i': 10,
    },
    # 'no_q_over_max':{
    #     'title': 'ACE-SQIL (No VP)',
    #     'plots': {'abl_alg'},
    #     'cmap_i': 12,
    # },
    'qomp1':{
        'title': r'$\lambda=1$',
        'plots': {'abl_alg', 'abl_all'},
        'cmap_i': 12,
    },
    'qomp100':{
        'title': r'$\lambda=100$',
        'plots': {'abl_alg', 'abl_all'},
        'cmap_i': 15,
    },
    'no_exp_random':{
        'title': 'No Example Augmentation',
        'plots': {'abl_alg', 'abl_exaug'},
        'cmap_i': 14,
    },
    # '20_data':{
    #     'title': '20 Examples',
    #     'plots': {'abl_dquant', 'abl_all', 'abl_exaug'},
    #     'cmap_i': 16,
    # },
    '10_data':{
        'title': '10 Examples',
        'plots': {'abl_dquant', 'abl_all', 'abl_exaug'},
        'cmap_i': 16,
    },
    '100_data':{
        'title': '100 Examples',
        'plots': {'abl_dquant', 'abl_all'},
        'cmap_i': 18,
    },
    # '20_data_no_exp_random':{
    #     'title': '20 Examples, No Ex. Aug.',
    #     'plots': {'abl_dquant', 'abl_exaug'},
    #     'cmap_i': 19,
    # }
    '10_data_no_exp_random':{
        'title': '10 Examples, No Ex. Aug.',
        'plots': {'abl_dquant', 'abl_exaug'},
        'cmap_i': 19,
    }
})


TASK_LIST = [
    "reach_0",
    "lift_0",
    "move_obj_0",
    "stack_no_move_0",
    "unstack_stack_env_only_no_move_0",
    "bring_no_move_0",
    "insert_no_bring_no_move_0"
]

PANDA_TASK_SETTINGS = OrderedDict({
    "reach_0": {'main_task_i': 1, 'eval_intervals': 10000},
    "lift_0": {'main_task_i': 2, 'eval_intervals': 10000},
    "move_obj_0": {'main_task_i': 4, 'eval_intervals': 10000},
    "stack_no_move_0": {'main_task_i': 2, 'eval_intervals': 25000},
    "unstack_stack_env_only_no_move_0": {'main_task_i': 2, 'eval_intervals': 25000},
    "bring_no_move_0": {'main_task_i': 2, 'eval_intervals': 25000},
    "insert_no_bring_no_move_0": {'main_task_i': 2, 'eval_intervals': 25000}
})


# return y_lims are best attempt to match original RCE paper
RCE_TASK_SETTINGS = OrderedDict({
    "sawyer_drawer_open":{
        "valid": True,
        "return_ylims": [-.01, .14]
    },
    "sawyer_drawer_close":{
        "valid": True,
        "return_ylims": [-.01, .2]
    },
    "sawyer_push": {
        "valid": True,
        "return_ylims": [-.01, .26]
    },
    "sawyer_lift": {
        "valid": True,
        "return_ylims": [-.01, .07]
    },
    "sawyer_box_close": {
        "valid": True,
        "return_ylims": [-.01, .27]
    },
    "sawyer_bin_picking":{
        "valid": True,
        "return_ylims": [-.01, .22]
    },
})

HAND_TASK_SETTINGS = OrderedDict({
    "door-human-v0-dp": {
        "valid": True,
        "return_ylims": [-.1e3, 1.5e3]
    },
    "hammer-human-v0-dp": {
        "valid": True,
        "return_ylims": [-.4e3, 3e3]
    },
    "relocate-human-v0-najp-dp": {
        "valid": True,
        "return_ylims": [-.1e3, 2.3e3]
    },
    "door-human-v0": {
        "valid": True,
        "return_ylims": [-.1e3, 3.1e3]
    },
    "hammer-human-v0": {
        "valid": True,
        "return_ylims": [-.1e4, 1e4]
    },
    "relocate-human-v0": {
        "valid": True,
        "return_ylims": [-.1e3, 2.5e3]
    }
})

AVG_ENVS_DICT = OrderedDict({
    'all': {
        'valid_task_settings': {**PANDA_TASK_SETTINGS, **RCE_TASK_SETTINGS, **HAND_TASK_SETTINGS},
        'valid_algos': ['multi-sqil', 'sqil-no-vp', 'rce', 'disc'],
        # 'valid_algos': ['multi-sqil', 'multi-sqil-no-vp', 'sqil', 'sqil-no-vp', 'rce'],
        'title': "All Envs/Tasks (Average)",
        'num_timesteps_mean': 5,
    },
    'main': {
        'valid_task_settings': {**PANDA_TASK_SETTINGS},
        # 'valid_algos': ['multi-sqil', 'multi-disc', 'multi-sqil-no-vp', 'multi-rce', 'sqil', 'disc', 'sqil-no-vp', 'rce'],
        'valid_algos': ['multi-sqil', 'sqil', 'multi-disc', 'disc', 'multi-sqil-no-vp', 'sqil-no-vp', 'multi-rce', 'rce'],
        'title': "Panda Main Tasks",
        'num_timesteps_mean': 10,
    },
    'rce': {
        'valid_task_settings': {**RCE_TASK_SETTINGS},
        # 'valid_algos': ['multi-sqil', 'multi-disc', 'multi-sqil-no-vp', 'multi-rce', 'sqil', 'disc', 'sqil-no-vp', 'rce'],
        'valid_algos': ['multi-sqil', 'sqil', 'multi-disc', 'disc', 'multi-sqil-no-vp', 'sqil-no-vp', 'multi-rce', 'rce'],
        'title': "Sawyer Main Tasks",
        'num_timesteps_mean': 10,
    },
    'hand': {
        'valid_task_settings': {**HAND_TASK_SETTINGS},
        # 'valid_algos': ['multi-sqil', 'multi-disc', 'multi-sqil-no-vp', 'multi-rce', 'sqil', 'disc', 'sqil-no-vp', 'rce'],
        'valid_algos': ['multi-sqil', 'sqil', 'multi-disc', 'disc', 'multi-sqil-no-vp', 'sqil-no-vp', 'multi-rce', 'rce'],
        'title': "Adroit Main Tasks",
        'num_timesteps_mean': 15,
    },
    'hand_orig': {
        'valid_task_settings': {
            'door-human-v0': HAND_TASK_SETTINGS['door-human-v0'],
            'hammer-human-v0': HAND_TASK_SETTINGS['hammer-human-v0'],
        },
        # 'valid_algos': ['multi-sqil', 'multi-disc', 'multi-sqil-no-vp', 'multi-rce', 'sqil', 'disc', 'sqil-no-vp', 'rce'],
        'valid_algos': ['multi-sqil', 'sqil', 'multi-disc', 'disc', 'multi-sqil-no-vp', 'sqil-no-vp', 'multi-rce', 'rce'],
        'title': "Adroit Main Tasks",
        'num_timesteps_mean': 15,
    },
    'hand_dp': {
        'valid_task_settings': {
            'door-human-v0-dp': HAND_TASK_SETTINGS['door-human-v0-dp'],
            'hammer-human-v0-dp': HAND_TASK_SETTINGS['hammer-human-v0-dp'],
            'relocate-human-v0-najp-dp': HAND_TASK_SETTINGS['relocate-human-v0-najp-dp'],
        },
        # 'valid_algos': ['multi-sqil', 'multi-disc', 'multi-sqil-no-vp', 'multi-rce', 'sqil', 'disc', 'sqil-no-vp', 'rce'],
        'valid_algos': ['multi-sqil', 'sqil', 'multi-disc', 'disc', 'multi-sqil-no-vp', 'sqil-no-vp', 'multi-rce', 'rce'],
        'title': "Adroit DP Main Tasks",
        'num_timesteps_mean': 15,
    },
    'rce_env_mods': {
        'valid_task_settings': {
            **RCE_TASK_SETTINGS,
            'door-human-v0': HAND_TASK_SETTINGS['door-human-v0'],
            'hammer-human-v0': HAND_TASK_SETTINGS['hammer-human-v0'],
        },
        # 'valid_algos': ['multi-sqil', 'multi-disc', 'multi-sqil-no-vp', 'multi-rce', 'sqil', 'disc', 'sqil-no-vp', 'rce'],
        'valid_algos': ['rce', 'rce_orig'],
        'title': "RCE Sawyer and Hand Tasks",
        'num_timesteps_mean': 15,
    },
})

def get_success_return(
    reload_data,
    task_dir_names,
    valid_task,
    algo_dir_names,
    num_eval_steps_to_use,
    multitask_algos,
    st_num_eval_steps_to_use,
    data_locations,
    experiment_root_dir,
    seeds,
    task_data_filenames,
    num_aux,
    eval_eps_per_task,
    fig_path,
    valid_algos
):
    if reload_data:
        all_returns = dict.fromkeys(task_dir_names)
        all_successes = dict.fromkeys(task_dir_names)
        for task_i, task in enumerate(task_dir_names):
            if not valid_task[task_i]:
                print(f"Task {task} set to false in valid_task, skipping")
                continue
            all_successes[task] = { algo : dict(raw=[], mean=[], std=[]) for algo in algo_dir_names }
            all_returns[task] = { algo : dict(raw=[], mean=[], std=[]) for algo in algo_dir_names }

            num_eval_steps_to_use_task = num_eval_steps_to_use[task_i] if task in multitask_algos else st_num_eval_steps_to_use[task_i]

            for algo_i, algo in enumerate(algo_dir_names):
                # if not valid_algos[algo_i]:
                if algo not in valid_algos:
                    # print(f"algo {algo} labelled as not valid, skipping.")
                    continue

                # folder structure is task/seed/algo/experiment_name/datetime
                algo_dir, experiment_name = data_locations[task][algo].split('/')

                data_path = os.path.join(experiment_root_dir, task, '1', algo_dir)
                if not os.path.exists(data_path):
                    print("No path found at %s for task %s algo %s, moving on in data cleaning" % (data_path, task, algo))
                    continue
                for seed in seeds:
                    # data_path = os.path.join(root_dir, top_task_dirs[task_i],  seed, algo, experiment_name)
                    data_path = os.path.join(experiment_root_dir, task, seed, algo_dir, experiment_name)

                    # find datetime folder
                    try:
                        dirs = sorted([os.path.join(data_path, found) for found in os.listdir(data_path)
                                    if os.path.isdir(os.path.join(data_path, found))])
                        if len(dirs) > 1:
                            print(f"WARNING: multiple folders found at {data_path}, using {dirs[-1]}")
                        data_path = dirs[-1]
                    except:
                        print(f"Error at data_path {data_path}")
                        # import ipdb; ipdb.set_trace()

                    if not os.path.exists(data_path):
                        print("No path found at %s for task %s, moving on in data cleaning" % (data_path, task))
                        continue

                    data_file = os.path.join(data_path, task_data_filenames[task_i])
                    if not os.path.isfile(data_file):
                        data_file = os.path.join(data_path, 'train.pkl')  # default for cases where we didn't want to rename
                    if not os.path.isfile(data_file):
                        print("No train.pkl file at %s for task %s, moving on in data cleaning" % (data_file, task))
                        continue

                    data = pickle.load(open(data_file, 'rb'))

                    suc_data = np.array(data['evaluation_successes_all_tasks']).squeeze()
                    ret_data = np.array(data['evaluation_returns']).squeeze()

                    if algo in multitask_algos:
                        expected_num_eval = num_eval_steps_to_use[task_i]
                    else:
                        expected_num_eval = st_num_eval_steps_to_use[task_i]

                    if suc_data.shape[0] < expected_num_eval:
                        print(f"Data for task {task}, algo {algo}, seed {seed} only has {suc_data.shape[0]} evals, "\
                              f"expected {expected_num_eval}. Skipping.")
                        continue

                    # adjust arrays to compensate for recycle scheduler
                    if algo in multitask_algos:
                        suc_fixed = []
                        ret_fixed = []

                        for aux_i in range(num_aux[task_i]):
                            rets_slice = slice(aux_i * eval_eps_per_task[task_i],
                                               aux_i * eval_eps_per_task[task_i] + eval_eps_per_task[task_i])
                            # for all algos, -1 index is episode, -2 is which aux, 0 is eval step index
                            suc_fixed.append(suc_data[..., aux_i, rets_slice])
                            ret_fixed.append(ret_data[..., aux_i, rets_slice])

                        suc_data = np.array(suc_fixed)
                        ret_data = np.array(ret_fixed)
                        # if not 'bc' in algo:
                        #     suc_data = np.swapaxes(suc_data, 0, 1)
                        #     ret_data = np.swapaxes(ret_data, 0, 1)

                        # now order is eval step index, aux index, eval ep index
                        suc_data = np.swapaxes(suc_data, 0, 1)
                        ret_data = np.swapaxes(ret_data, 0, 1)

                    # remove extra eval step indices if there are any
                    suc_data = suc_data[..., :num_eval_steps_to_use_task, :]
                    ret_data = ret_data[..., :num_eval_steps_to_use_task, :]

                    all_successes[task][algo]['raw'].append(suc_data)
                    all_returns[task][algo]['raw'].append(ret_data)

                if all_returns[task][algo]['raw'] == []:
                    print(f"No data for task {task}, algo {algo}, skipping.")
                    continue

                try:
                    all_returns[task][algo]['raw'] = np.array(all_returns[task][algo]['raw']).squeeze()
                    all_successes[task][algo]['raw'] = np.array(all_successes[task][algo]['raw']).squeeze()

                    # take along episode axis, then along seed axis..matters for std
                    for dat in [all_returns[task][algo], all_successes[task][algo]]:

                        # first cut out undesired eval steps
                        dat['raw'] = dat['raw'][:, :num_eval_steps_to_use_task]

                        # dat['mean'] = dat['raw'][:, :num_eval_steps_to_use[task_i]].mean(axis=-1).mean(axis=0)
                        # dat['std'] = dat['raw'][:, :num_eval_steps_to_use[task_i]].mean(axis=-1).std(axis=0)
                        dat['mean'] = dat['raw'].mean(axis=-1).mean(axis=0)
                        dat['std'] = dat['raw'].mean(axis=-1).std(axis=0)

                except Exception as e:
                    print(f"Exception: {e}")
                    print(f"For task {task}, algo {algo}, skipping because sizes didn't match..is there an unfinished run?")
                    # import ipdb; ipdb.set_trace()

        # save the data for more quickly recreating the figure for format-only fixes
        data = {'all_returns': all_returns, 'all_successes': all_successes}
        data_path = os.path.join(fig_path, 'data')
        os.makedirs(data_path, exist_ok=True)
        pickle.dump(data, open(os.path.join(data_path, 'data.pkl'), 'wb'))
    else:
        data = pickle.load(open(os.path.join(fig_path, 'data', 'data.pkl'), 'rb'))
        all_returns = data['all_returns']
        all_successes = data['all_successes']

    return all_returns, all_successes


def get_path_defaults(fig_name, task_inds=(0,1,2,3)):
    root_dir = os.environ['VPACE_TOP_DIR']
    fig_path = os.path.join(root_dir, "figures", fig_name)
    experiment_root_dir = root_dir
    seeds = ['1','2','3','4','5']
    expert_root = os.path.join(root_dir, "expert-data")

    out_epf = None
    out_epf_mti = None

    return root_dir, fig_path, experiment_root_dir, seeds, expert_root, out_epf, out_epf_mti


def get_task_defaults(plot='main'):
    if plot == 'rce':
        valid_task = [RCE_TASK_SETTINGS[task_key]['valid'] for task_key in RCE_TASK_SETTINGS.keys()]
        task_titles = [*RCE_TASK_SETTINGS]
        main_task_i = [0] * len(RCE_TASK_SETTINGS.keys())
        num_aux = [3] * len(RCE_TASK_SETTINGS.keys())
        task_data_filenames = ['train.pkl'] * len(RCE_TASK_SETTINGS.keys())
        num_eval_steps_to_use = [30, 30, 50, 50, 50, 30]
        single_task_nestu = [30, 30, 50, 50, 50, 30]
        eval_intervals = [10000] * len(RCE_TASK_SETTINGS.keys())
        task_list = task_titles
        eval_eps_per_task = [30] * len(RCE_TASK_SETTINGS.keys())
    elif plot == 'main':
        valid_task = [True, True, True, True, True, True, True]
        task_titles = ["Reach", "Lift", "Move-Block", "Stack", "Unstack-Stack", "Bring", "Insert"]
        main_task_i = [1, 2, 4, 2, 2, 2, 2]
        # num_aux = [2, 4, 5, 6, 6, 6, 6]
        num_aux = [2, 4, 5, 5, 5, 5, 5]
        task_data_filenames = ['train.pkl', 'train.pkl', 'train.pkl', 'train.pkl', 'train.pkl', 'train.pkl', 'train.pkl']
        num_eval_steps_to_use = [20, 20, 20, 20, 20, 20, 40]
        # single_task_nestu = [20, 20, 20, 20, 40, 20, 40]
        single_task_nestu = [20, 20, 20, 20, 20, 20, 40]
        eval_intervals = [10000, 10000, 10000, 25000, 25000, 25000, 25000]
        task_list = TASK_LIST
        eval_eps_per_task = [50] * len(task_titles)
    elif 'abl' in plot:
        valid_task = [True]
        task_titles = ["Unstack-Stack"]
        main_task_i = [2]
        num_aux = [5]
        task_data_filenames = ['train.pkl']
        num_eval_steps_to_use = [20]
        single_task_nestu = [20]
        eval_intervals = [25000]
        task_list = [TASK_LIST[4]]
        eval_eps_per_task = [50] * len(task_titles)
    elif plot == 'hand':
        valid_task = [HAND_TASK_SETTINGS[task_key]['valid'] for task_key in HAND_TASK_SETTINGS.keys()]
        task_titles = [*HAND_TASK_SETTINGS]
        main_task_i = [0] * len(HAND_TASK_SETTINGS.keys())
        num_aux = [3] * len(HAND_TASK_SETTINGS.keys())
        task_data_filenames = ['train.pkl'] * len(HAND_TASK_SETTINGS.keys())
        num_eval_steps_to_use = [50, 100, 150, 30, 50, 150]
        single_task_nestu = [50, 100, 150, 30, 50, 150]
        eval_intervals = [10000] * len(HAND_TASK_SETTINGS.keys())
        task_list = task_titles
        eval_eps_per_task = [30] * len(task_titles)
    elif plot in ['re_vs_cl', 'rce_hand_theirs']:
        valid_task = [RCE_TASK_SETTINGS[task_key]['valid'] for task_key in RCE_TASK_SETTINGS.keys()]
        valid_task.extend([HAND_TASK_SETTINGS[task_key]['valid'] for task_key in ['door-human-v0', 'hammer-human-v0']])
        task_titles = [*RCE_TASK_SETTINGS]
        task_titles.extend(['door-human-v0', 'hammer-human-v0'])
        num_tasks = len(task_titles)
        main_task_i = [0] * num_tasks
        num_aux = [3] * num_tasks
        task_data_filenames = ['train.pkl'] * num_tasks
        num_eval_steps_to_use = [30, 30, 50, 50, 50, 30, 30, 50]
        single_task_nestu = [30, 30, 50, 50, 50, 30, 30, 50]
        eval_intervals = [10000] * num_tasks
        task_list = task_titles
        eval_eps_per_task = [30] * len(task_titles)
    elif plot == 'hardest':
        task_titles = ['Stack',
                       'Unstack-Stack',
                       'Bring',
                       'Insert',
                       'sawyer_box_close',
                       'sawyer_bin_picking',
                       'hammer-human-v0-dp',
                       'relocate-human-v0-dp']
        num_tasks = len(task_titles)
        valid_task = [True] * len(task_titles)
        main_task_i = [2, 2, 2, 2, 0, 0, 0, 0]
        num_aux = [5, 5, 5, 5, 3, 3, 3, 3]
        task_data_filenames = ['train.pkl'] * num_tasks
        num_eval_steps_to_use = [20, 20, 20, 40, 50, 30, 100, 150]
        single_task_nestu = num_eval_steps_to_use
        eval_intervals = [25000, 25000, 25000, 25000, 10000, 10000, 10000, 10000]
        task_list = ['stack_no_move_0',
                       'unstack_stack_env_only_no_move_0',
                       'bring_no_move_0',
                       'insert_no_bring_no_move_0',
                       'sawyer_box_close',
                       'sawyer_bin_picking',
                       'hammer-human-v0-dp',
                       'relocate-human-v0-najp-dp']
        eval_eps_per_task = [50, 50, 50, 50, 30, 30, 30, 30]
    elif plot == 'hardest_4':
        task_titles = ['Unstack-Stack',
                       'Insert',
                       'sawyer_box_close',
                       'relocate-human-v0-dp']
        num_tasks = len(task_titles)
        valid_task = [True] * len(task_titles)
        main_task_i = [2, 2, 0, 0]
        num_aux = [5, 5, 3, 3]
        task_data_filenames = ['train.pkl'] * num_tasks
        num_eval_steps_to_use = [20, 40, 50, 150]
        single_task_nestu = num_eval_steps_to_use
        eval_intervals = [25000, 25000, 10000, 10000]
        task_list = ['unstack_stack_env_only_no_move_0',
                       'insert_no_bring_no_move_0',
                       'sawyer_box_close',
                       'relocate-human-v0-najp-dp']
        eval_eps_per_task = [50, 50, 30, 30]

    out_tdn = []
    out_vt = []
    out_tt = []
    out_mti = []
    out_na = []
    out_tdf = []
    out_nestu = []
    out_stnestu = []
    out_ei = []
    out_eept = []
    for i in range(len(task_list)):
        if valid_task[i]:
            out_tdn.append(task_list[i])
            out_vt.append(valid_task[i])
            out_tt.append(task_titles[i])
            out_mti.append(main_task_i[i])
            out_na.append(num_aux[i])
            out_tdf.append(task_data_filenames[i])
            out_nestu.append(num_eval_steps_to_use[i])
            out_stnestu.append(single_task_nestu[i])
            out_ei.append(eval_intervals[i])
            out_eept.append(eval_eps_per_task[i])

    return out_tdn, out_vt, out_tt, out_mti, out_na, out_tdf, out_nestu, out_stnestu, out_ei, out_eept


def get_algo_defaults(plot='main'):
    algo_dir_names = []
    algo_titles = []
    multitask_algos = []
    valid_algos = []
    cmap_is = []
    for k in ALGO_TITLE_DICT.keys():
        # if plot in ALGO_TITLE_DICT[k]['plots']:
        algo_dir_names.append(k)
        if 'abl' in plot and k == 'multi-sqil':
            algo_titles.append("VPACE-SQIL, no ablations")
        elif 'abl' in plot and k == 'sqil':
            algo_titles.append("VP-SQIL, no ablations")
        else:
            algo_titles.append(ALGO_TITLE_DICT[k]['title'])
        if 'multi' in k or 'abl' in plot:
            if 'st' not in k and k != 'sqil':
                multitask_algos.append(k)
        if plot in ALGO_TITLE_DICT[k]['plots']:
            valid_algos.append(k)
        cmap_is.append(ALGO_TITLE_DICT[k]['cmap_i'])

    # eval_eps_per_task = 30 if plot in ['rce', 'hand'] else 50

    # return algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task, valid_algos, cmap_is
    return algo_dir_names, algo_titles, multitask_algos, valid_algos, cmap_is


def get_fig_defaults(num_plots=4, plot='main'):
    if num_plots <= 4:
        fig_shape = [1, num_plots]  # row x col
    elif 4 < num_plots <= 6:
        fig_shape = [2, 3]
    elif 6 < num_plots < 8:
        fig_shape = [2, num_plots // 2 + 1]
    elif num_plots == 8:
        fig_shape = [2, 4]
    else:
        fig_shape = [3, num_plots // 3 + 1]

    plot_size = [3.2, 2.4]
    # num_stds = 1
    num_stds = .5
    font_size = 16
    eval_interval = 25000  # now task dependent
    # cmap = plt.get_cmap("tab10")
    cmap = plt.get_cmap("tab20")
    linewidth = 1.5
    # std_alpha = .5
    std_alpha = .25
    x_val_scale = 1e5
    subsample_rate = 1  # 1 for no subsample
    include_expert_baseline = True
    num_timesteps_mean = 5

    return fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
        include_expert_baseline, num_timesteps_mean