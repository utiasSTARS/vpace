import rl_sandbox.constants as c


def get_common_env_defaults(args):
    move_file = args.move_obj_filename
    ust = '_unstack-stack'

    if args.env_type == c.MANIPULATOR_LEARNING:
        ##### scheduler period, control freq
        if args.control_hz == 5:
            args.env_control_hz = 5
            args.scheduler_period = 12
        elif args.control_hz == 10:
            args.env_control_hz = 10
            args.scheduler_period = 23
        if args.control_hz in [5, 10]:
            args.max_episode_length = 8 * args.scheduler_period
            args.env_max_real_time = args.max_episode_length / args.env_control_hz

        ##### expert data filenames, max steps, scheduler type, save interval, eval freq (if diff from defaults)
        if 'unstack' in args.main_task and 'with-us-aux' in args.main_task:
            if args.single_task:
                args.expert_filenames = "2_stack.gz"
                args.max_steps = 1600000
            else:
                args.expert_filenames = f"0_stack-open.gz,1_close.gz,2_stack.gz,3_unstack.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.max_steps = 800000
        elif 'unstack' in args.main_task:
            if args.single_task:
                args.expert_filenames = "2_stack.gz"
                # args.max_steps = 1000000
                args.max_steps = 500000
            else:
                args.expert_filenames = f"0_stack-open.gz,1_close.gz,2_stack.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.max_steps = 500000
        elif 'stack' in args.main_task:
            if args.single_task:
                args.expert_filenames = "2_stack.gz"
                args.max_steps = 500000
            else:
                args.expert_filenames = f"0_stack-open.gz,1_close.gz,2_stack.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.max_steps = 500000
        elif 'bring' in args.main_task:
            if args.single_task:
                args.expert_filenames = "2_bring.gz"
                args.max_steps = 500000
            else:
                args.expert_filenames = f"0_bring-open.gz,1_close.gz,2_bring.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.max_steps = 500000
        elif args.main_task in ['insert', 'insert_0']:
            args.exploration_steps = 20000
            if args.single_task:
                args.expert_filenames = "2_insert.gz"
                args.max_steps = 1600000
            else:
                args.expert_filenames = f"0_insert-open.gz,1_close.gz,2_insert.gz,2_bring.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.max_steps = 800000
        elif 'insert_nb' in args.main_task:
            args.exploration_steps = 20000
            if args.single_task:
                args.expert_filenames = "2_insert.gz"
                args.max_steps = 1000000
            else:
                args.expert_filenames = f"0_insert-open.gz,1_close.gz,2_insert.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.max_steps = 1000000
        elif 'move' in args.main_task:
            if args.single_task:
                args.expert_filenames = f"{move_file}"
                args.max_steps = 200000
            else:
                args.expert_filenames = f"0_stack-open.gz,1_close.gz,3_lift.gz,4_reach.gz,{move_file}"
                args.main_intention = 4
                args.scheduler = 'wrs'
                args.max_steps = 200000
            args.save_interval = 50000
            args.eval_freq = 10000
        elif 'lift' in args.main_task:
            if args.single_task:
                args.expert_filenames = "3_lift.gz"
                args.max_steps = 200000
            else:
                args.expert_filenames = "0_stack-open.gz,1_close.gz,3_lift.gz,4_reach.gz"
                args.scheduler = 'wrs'
                args.max_steps = 200000
            args.save_interval = 50000
            args.eval_freq = 10000
        elif 'reach' in args.main_task:
            if args.single_task:
                args.expert_filenames = "4_reach.gz"
                args.max_steps = 200000
            else:
                args.expert_filenames = "0_stack-open.gz,4_reach.gz"
                args.main_intention = 1
                args.scheduler = 'wrs'
                args.max_steps = 200000
            args.save_interval = 50000
            args.eval_freq = 10000
        else:
            raise NotImplementedError("Expert data not ready yet!")

        # handle no move, aka nm, versions of envs
        if 'nm' in args.main_task and not args.single_task:
            args.expert_filenames = args.expert_filenames.split(move_file)[0][:-1]  # up to last removes trailing comma

        # handle full trajs as option, unstack has special names
        if args.add_default_full_traj:
            if 'unstack' in args.main_task:
                if args.single_task:
                    args.full_traj_expert_filenames = f"2{ust}.gz"
                else:
                    ft_us_str = f"0{ust}-open.gz,1{ust}-close.gz,2{ust}.gz,3{ust}-lift.gz,4{ust}-reach.gz"
                    if 'nm' not in args.main_task: ft_us_str += f",5{ust}-move.gz"
                    args.full_traj_expert_filenames = ft_us_str
            else:
                args.full_traj_expert_filenames = args.expert_filenames

    elif args.env_type == c.SAWYER:
        ##### scheduler period
        if args.sawyer_aux_tasks == 'reach,grasp':
            args.scheduler_period = 30
        else:
            args.scheduler_period = 50
        if args.action_repeat == 2:
            args.scheduler_period = 38

        ##### max steps
        if args.env_name == 'sawyer_lift':
            args.max_steps = 500000
        elif args.env_name == 'sawyer_push':
            args.max_steps = 500000
        elif args.env_name == 'sawyer_box_close':
            args.max_steps = 500000
        elif args.env_name == 'sawyer_bin_picking':
            args.max_steps = 300000
        elif args.env_name == 'sawyer_drawer_open':
            args.max_steps = 300000
        elif args.env_name == 'sawyer_drawer_close':
            args.max_steps = 300000

        ##### expert data filenames
        if args.expert_dir_rest == "":
            args.expert_dir_rest = 'sawyer'
        else:
            args.expert_dir_rest = 'expert_data/1200_per_task/sawyer'

        if args.single_task:
            args.expert_filenames = f'{args.env_name}.gz'
        else:
            args.main_intention = 0
            args.expert_filenames = f"{args.env_name}.gz,{args.env_name}_random_reach.gz"

            if args.sawyer_aux_tasks == 'reach,grasp':
                args.expert_filenames += f",{args.env_name}_random_grasp.gz"

        if args.env_name == 'sawyer_reach' or args.env_name == 'sawyer_air_reach':
            args.max_episode_length = 50
        else:
            args.max_episode_length = 150

        if args.sawyer_vel_in_env:
            assert args.sawyer_grip_pos_in_env, \
                "No data with vel in env but without grip pos, sawyer_grip_pos_in_env must be true for sawyer_vel_in_env"
            args.expert_dir_rest += '/with_vel'

        if not args.sawyer_grip_pos_in_env:
            args.expert_dir_rest += '/no_grip_pos'

    elif args.env_type == c.HAND_DAPG:
        ##### scheduler period
        if args.hand_dapg_aux_tasks == 'reach,grasp':
            args.scheduler_period = 40
        else:
            args.scheduler_period = 67

        data_env_name = args.env_name
        if args.env_name == 'door-human-v0':
            args.max_steps = 300000
        elif args.env_name == 'door-human-v0-dp':
            args.max_steps = 500000
            data_env_name = 'door-human-v0'
        elif args.env_name == 'hammer-human-v0-dp':
            args.max_steps = 1000000
            data_env_name = 'hammer-human-v0'
        elif args.env_name == 'hammer-human-v0':
            args.max_steps = 500000
        elif args.env_name in ['relocate-human-v0', 'relocate-human-v0-dp']:
            args.max_steps = 1500000
            data_env_name = 'relocate-human-v0'
        elif args.env_name in ['relocate-human-v0-najp', 'relocate-human-v0-najp-dp']:
            args.max_steps = 1500000
            data_env_name = 'relocate-human-v0-najp'

        args.expert_dir_rest = "expert_data/1200_per_task/hand_dapg"
        if 'include_vel:True' in args.hand_dapg_dp_kwargs:
            args.expert_dir_rest += "/with_vel"

        if args.single_task:
            args.expert_filenames = f'{data_env_name}.gz'
        else:
            args.main_intention = 0
            args.expert_filenames = f"{data_env_name}.gz,{data_env_name}_reach.gz"

            if args.hand_dapg_aux_tasks == 'reach,grasp':
                args.expert_filenames += f",{data_env_name}_grasp.gz"

        args.max_episode_length = 200

    else:
        raise NotImplementedError("Expert data not ready yet!")

    if args.load_max_buffer_index:
        args.memory_size = args.max_steps + args.load_max_buffer_index
    else:
        args.memory_size = args.max_steps