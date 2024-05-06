import argparse
import numpy as np
import os

from rl_sandbox.buffers.utils import make_buffer

import rl_sandbox.constants as c
import rl_sandbox.envs.rce_envs as rce_envs

parser = argparse.ArgumentParser("Download data")
parser.add_argument("--env_name", type=str, required=True)
parser.add_argument("--data_amount", type=int, default=1200)  # rce used 200, but our script can change size on the fly
parser.add_argument("--no_grip_pos_in_env", action='store_true', help="Remove added grip pos from envs")
parser.add_argument("--vel_in_env", action='store_true', help="Add vel (both ee and obj) to envs")
args = parser.parse_args()

if args.env_name == 'all':
    env_names = [
        'sawyer_reach',
        'sawyer_air_reach',
        'sawyer_drawer_open',
        'sawyer_push',
        'sawyer_lift',
        'sawyer_box_close',
        'sawyer_bin_picking',
        'door-human-v0',
        'hammer-human-v0',
        # 'relocate-human-v0'
    ]

    for saw_env in ['push', 'lift', 'box_close', 'bin_picking', 'drawer_open', 'drawer_close']:
        env_names.extend([f'sawyer_{saw_env}_reach', f'sawyer_{saw_env}_random_reach'])
        env_names.extend([f'sawyer_{saw_env}_grasp', f'sawyer_{saw_env}_random_grasp'])

    for hand_dapg_env in ['door', 'hammer']:
        env_names.extend([f"{hand_dapg_env}-human-v0_reach", "{hand_dapg_env}-human-v0_grasp"])

else:
    env_names = [args.env_name]

for env_name in env_names:
    tf_env = rce_envs.load_env(env_name, grip_pos_in_env=not args.no_grip_pos_in_env, vel_in_env=args.vel_in_env)

    expert_obs = rce_envs.get_data(tf_env.env, env_name=env_name, num_expert_obs=args.data_amount)

    (memory_size, obs_dim) = expert_obs.shape
    action_dim = tf_env.action_space.shape[0]

    buffer_settings = {
        c.KWARGS: {
            c.MEMORY_SIZE: memory_size,
            c.OBS_DIM: (obs_dim,),
            c.H_STATE_DIM: (1,),
            c.ACTION_DIM: (action_dim,),
            c.REWARD_DIM: (1,),
            c.INFOS: {c.MEAN: ((action_dim,), np.float32),
                        c.VARIANCE: ((action_dim,), np.float32),
                        c.ENTROPY: ((action_dim,), np.float32),
                        c.LOG_PROB: ((1,), np.float32),
                        c.VALUE: ((1,), np.float32),
                        c.DISCOUNTING: ((1,), np.float32)},
            c.CHECKPOINT_INTERVAL: 0,
            c.CHECKPOINT_PATH: None,
        },
        c.STORAGE_TYPE: c.RAM,
        c.BUFFER_TYPE: c.STORE_NEXT_OBSERVATION,
        c.BUFFER_WRAPPERS: [],
        c.LOAD_BUFFER: False,
    }
    buffer = make_buffer(buffer_settings)

    for obs in expert_obs:
        info = {c.DISCOUNTING: 1}
        buffer.push(obs, np.zeros(1), np.zeros(action_dim), 0., True, info, next_obs=obs, next_h_state=np.zeros(1))

    os.makedirs('exp_data', exist_ok=True)
    if args.vel_in_env:
        os.makedirs('exp_data/with_vel', exist_ok=True)
        buffer.save(f"exp_data/with_vel/{env_name}.gz", end_with_done=False)
    elif args.no_grip_pos_in_env:
        os.makedirs('exp_data/no_grip_pos', exist_ok=True)
        buffer.save(f"exp_data/no_grip_pos/{env_name}.gz", end_with_done=False)
    else:
        buffer.save(f"exp_data/{env_name}.gz", end_with_done=False)
