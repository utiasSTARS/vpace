# Since our default folder structure is task/seed/algo/experiment_name/datetime,
# we have algo/experiment_name for each task/algo combo

algo_dir_names = ['multi-sqil', 'multi-dac', 'multi-rce', 'sqil', 'dac', 'rce']

main = {
    'reach_0': {
        'multi-sqil':        "multi-sqil/feb24_newqmax_fixed",
        'multi-disc':        "multi-disc/feb24_newqmax_fixed",
        'multi-rce':         "multi-rce/feb24_newqmax_fixed",
        'sqil':              "sqil/feb24_newqmax_fixed",
        'disc':              "disc/feb24_newqmax_fixed",
        'rce':               "rce/feb24_newqmax_fixed",
        'multi-sqil-no-vp':          "multi-sqil/apr4_noqovermax",
        'sqil-no-vp':        "sqil/apr4_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
        # 'disc-no-vp':        "disc/feb24_newqmax_fixed",
    },
    'lift_0': {
        'multi-sqil':        "multi-sqil/feb24_newqmax_fixed",
        'multi-disc':        "multi-disc/feb24_newqmax_fixed",
        'multi-rce':         "multi-rce/feb24_newqmax_fixed",
        'sqil':              "sqil/feb24_newqmax_fixed",
        'disc':              "disc/feb24_newqmax_fixed",
        'rce':               "rce/feb24_newqmax_fixed",
        'multi-sqil-no-vp':          "multi-sqil/apr4_noqovermax",
        'sqil-no-vp':        "sqil/apr4_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
        # 'disc-no-vp':        "disc/feb24_newqmax_fixed",
    },
    'move_obj_0': {
        'multi-sqil':        "multi-sqil/feb24_newqmax_fixed",
        'multi-disc':        "multi-disc/feb24_newqmax_fixed",
        'multi-rce':         "multi-rce/feb24_newqmax_fixed",
        'sqil':              "sqil/feb24_newqmax_fixed",
        'disc':              "disc/feb24_newqmax_fixed",
        'rce':               "rce/feb24_newqmax_fixed",
        'multi-sqil-no-vp':          "multi-sqil/apr4_noqovermax",
        'sqil-no-vp':        "sqil/apr4_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
        # 'disc-no-vp':        "disc/feb24_newqmax_fixed",
    },
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':        "multi-sqil/apr15_envfix",
        'multi-disc':        "multi-disc/apr15_envfix",
        'multi-rce':         "multi-rce/apr15_envfix",
        'sqil':              "sqil/apr15_envfix",
        'disc':              "disc/apr15_envfix",
        'rce':               "rce/apr15_envfix",
        'multi-sqil-no-vp':          "multi-sqil/apr15_envfix_noqovermax",
        'sqil-no-vp':        "sqil/apr15_envfix_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
        'qreg-ace-sqil':         "multi-sqil/c2f",
        'cql-ace-sqil':          "multi-sqil/cql",
    },
    'stack_no_move_0': {
        'multi-sqil':        "multi-sqil/feb26_avgexpqmax",
        'multi-disc':        "multi-disc/feb26_avgexpqmax",
        'multi-rce':         "multi-rce/feb26_avgexpqmax",
        'sqil':              "sqil/feb26_avgexpqmax",
        'disc':              "disc/feb26_avgexpqmax",
        'rce':               "rce/feb26_avgexpqmax",
        'multi-sqil-no-vp':          "multi-sqil/apr4_noqovermax",
        'sqil-no-vp':        "sqil/apr4_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
    },
    'bring_no_move_0': {
        'multi-sqil':        "multi-sqil/feb26_avgexpqmax",
        'multi-disc':        "multi-disc/feb26_avgexpqmax",
        'multi-rce':         "multi-rce/feb26_avgexpqmax",
        'sqil':              "sqil/feb26_avgexpqmax",
        'disc':              "disc/feb26_avgexpqmax",
        'rce':               "rce/feb26_avgexpqmax",
        'multi-sqil-no-vp':          "multi-sqil/apr4_noqovermax",
        'sqil-no-vp':        "sqil/apr4_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
    },
    'insert_no_bring_no_move_0': {
        'multi-sqil':        "multi-sqil/feb26_avgexpqmax",
        'multi-disc':        "multi-disc/feb26_avgexpqmax",
        'multi-rce':         "multi-rce/feb26_avgexpqmax",
        'sqil':              "sqil/feb26_avgexpqmax",
        'disc':              "disc/feb26_avgexpqmax",
        'rce':               "rce/feb26_avgexpqmax",
        'multi-sqil-no-vp':          "multi-sqil/apr4_noqovermax",
        'sqil-no-vp':        "sqil/apr4_noqovermax",
        'sqil-rnd':          "sqil/sqil_rnd",
        'disc-no-vp':        "disc/dac",
    },
}

abl_expert = {
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':       "multi-sqil/apr15_envfix",
        'full_trajs':       "multi-sqil/apr15_envfix_full_trajs",
        'full_trajs_wa':       "multi-sqil/apr15_envfix_full_trajs_with_actions",
        'sparse_rew':       "multi-spar/apr15_envfix_sparse_rew",
    },
    'insert_no_bring_no_move_0': {
        'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        'full_trajs':       "multi-sqil/aug5_full_trajs",
        'full_trajs_wa':       "multi-sqil/aug5_full_trajs_with_act",
        'sparse_rew':       "multi-spar/aug5_sparse_rew",
    }
}

abl_dquant_lambda = {
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':       "multi-sqil/apr15_envfix",
        '10_data':          "multi-sqil/apr15_envfix_10data",
        '100_data':         "multi-sqil/apr15_envfix_100data",
        'qomp1':    "multi-sqil/apr15_envfix_qomp1",
        'qomp100':    "multi-sqil/apr15_envfix_qomp100",
    },
    'insert_no_bring_no_move_0': {
        'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        '10_data':          "multi-sqil/aug5_10data",
        '100_data':         "multi-sqil/aug7_100data",
        'qomp1':    "multi-sqil/aug25_qomp1",
        'qomp100':    "multi-sqil/aug25_qomp100",
    }
}

abl_reg = {
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':       "multi-sqil/apr15_envfix",
        'cql-ace-sqil':          "multi-sqil/cql",
        'qreg-ace-sqil':         "multi-sqil/c2f",
    },
    'insert_no_bring_no_move_0': {
        'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        'cql-ace-sqil':          "multi-sqil/cql",
        'qreg-ace-sqil':         "multi-sqil/c2f",
    }
}

abl_rew_model = {
    'unstack_stack_env_only_no_move_0': {
        'sqil':       "sqil/apr15_envfix",
        'sqil-no-vp':       "sqil/apr15_envfix_noqovermax",
        'disc':              "disc/apr15_envfix",
        'disc-no-vp':        "disc/dac",
        'rce':               "rce/apr15_envfix",
        'ember':               "disc/ember",
        'rce-cql':               "rce/rce-cql",
    },
    'insert_no_bring_no_move_0': {
        'sqil':       "sqil/feb26_avgexpqmax",
        'sqil-no-vp':       "sqil/apr4_noqovermax",
        'disc':              "disc/feb26_avgexpqmax",
        'disc-no-vp':        "disc/dac",
        'rce':               "rce/feb26_avgexpqmax",
        'ember':         "disc/ember",
        'rce-cql':         "rce/rce-cql",
    }
}

abl_lambda = {
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':       "multi-sqil/apr15_envfix",
        'qomp1':    "multi-sqil/apr15_envfix_qomp1",
        'qomp100':    "multi-sqil/apr15_envfix_qomp100",
    }
}

abl_alg = {
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        # 'no_q_over_max':    "multi-sqil/mar4_noqovermax",
        'qomp1':    "multi-sqil/apr9_qomp1",
        'qomp100':    "multi-sqil/apr9_qomp100",
        'no_exp_random':    "multi-sqil/mar4_noerf",
    }
}

abl_dquant = {
    'unstack_stack_env_only_no_move_0': {
        'multi-sqil':       "multi-sqil/apr15_envfix",
        '10_data':          "multi-sqil/apr15_envfix_10data",
        '100_data':         "multi-sqil/apr15_envfix_100data",
    },
    'insert_no_bring_no_move_0': {
        'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        '10_data':          "multi-sqil/aug5_10data",
        '100_data':         "multi-sqil/aug7_100data",
    }
}

abl_all = {
    'unstack_stack_env_only_no_move_0': {
        # 'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        # 'full_trajs':       "multi-sqil/mar4_full_trajs",
        # 'sparse_rew':       "multi-spar/mar4_sparse_rew",
        # 'qomp1':    "multi-sqil/apr9_qomp1",
        # 'qomp100':    "multi-sqil/apr9_qomp100",
        # '20_data':          "multi-sqil/mar4_20data",
        # '100_data':         "multi-sqil/mar4_100data",
        'sqil':             "sqil/apr15_envfix",
        'multi-sqil':       "multi-sqil/apr15_envfix",
        'full_trajs':       "multi-sqil/apr15_envfix_full_trajs",
        'full_trajs_st':       "sqil/apr15_envfix_full_trajs",
        'full_trajs_wa':       "multi-sqil/apr15_envfix_full_trajs_with_actions",
        'sparse_rew':       "multi-spar/apr15_envfix_sparse_rew",
        'qomp1':    "multi-sqil/apr15_envfix_qomp1",
        'qomp100':    "multi-sqil/apr15_envfix_qomp100",
        '10_data':          "multi-sqil/apr15_envfix_10data",
        '100_data':         "multi-sqil/apr15_envfix_100data",
    }
}

abl_exaug = {
    'unstack_stack_env_only_no_move_0': {
        # 'multi-sqil':       "multi-sqil/feb26_avgexpqmax",
        # 'no_exp_random':    "multi-sqil/mar4_noerf",
        # '20_data':          "multi-sqil/mar4_20data",
        # '20_data_no_exp_random':         "multi-sqil/mar15_noerf_20data",
        'multi-sqil':       "multi-sqil/apr15_envfix",
        'no_exp_random':    "multi-sqil/apr15_envfix_noerf",
        '10_data':          "multi-sqil/apr15_envfix_10data",
        '10_data_no_exp_random':         "multi-sqil/apr15_envfix_noerf_10data",
    }
}