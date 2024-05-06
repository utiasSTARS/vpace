import argparse

import matplotlib.pyplot as plt
from rl_sandbox.envs.rce_envs import load_env

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='sawyer_box_close')
args = parser.parse_args()

env = load_env(args.env)
obs = env.reset()
viewer = env.unwrapped._get_viewer('human')

for i in range(10000):
    img = env.render()
    env.step(env.action_space.sample())
    print(
        f"NEW_CAM_SETTINGS = {{\n"
        f"    'distance': {viewer.cam.distance:.3f},\n"
        f"    'lookat': [{viewer.cam.lookat[0]:.3f}, {viewer.cam.lookat[1]:.3f}, {viewer.cam.lookat[2]:.3f}],\n"
        f"    'elevation': {viewer.cam.elevation:.3f},\n"
        f"    'azimuth': {viewer.cam.azimuth:.3f},\n"
        f"}}\n"
    )

import ipdb; ipdb.set_trace()

