import argparse
import os
import cv2
import copy

import matplotlib.pyplot as plt
from rl_sandbox.envs.rce_envs import load_env
from rl_sandbox.envs.rce_multitask_envs import hand_dapg_get_dataset
import cam_settings
import common as fig_common

parser = argparse.ArgumentParser()
parser.add_argument('--img_w', type=int, default=500)
parser.add_argument('--img_h', type=int, default=500)
parser.add_argument('--num_ex', type=int, default=1)
parser.add_argument('--no_aux', action='store_true')
parser.add_argument('--top_save_dir', type=str, default=os.path.join(os.environ['VPACE_TOP_DIR'],
                                                                     'figures', 'reset_success_examples'))
parser.add_argument('--vid_save_only', action='store_true')
parser.add_argument('--vid_fps', type=float, default=1.0)
parser.add_argument('--resolution', type=str, required=False)
parser.add_argument('--crf', type=int, required=False)
parser.add_argument('--render_on_screen', action='store_true')
args = parser.parse_args()

env_list = [
    # 'sawyer_drawer_open',
    # 'sawyer_drawer_close',
    # 'sawyer_push',
    # 'sawyer_lift',
    # 'sawyer_box_close',
    # 'sawyer_bin_picking',
    'door-human-v0',
    # 'hammer-human-v0',
    # 'relocate-human-v0',
]

os.makedirs(args.top_save_dir, exist_ok=True)

for env_str in env_list:

    all_reset = []
    all_reach = []
    all_grasp = []
    all_success = []

    for ex_num in range(args.num_ex):

        env = load_env(env_str)
        extra_img_str = ""
        if ex_num > 0:
            env.seed(ex_num)
            extra_img_str = f"_{ex_num}"
        obs = env.reset()
        rgb_viewer = env.unwrapped._get_viewer('rgb_array')

        cam_settings.set_cam_settings(rgb_viewer, cam_settings.CAM_SETTINGS[env_str])

        # reset img
        reset_img = env.render('rgb_array', width=args.img_w, height=args.img_h)
        all_reset.append(reset_img)
        if not args.vid_save_only:
            cv2.imwrite(os.path.join(args.top_save_dir, f"{env_str}_reset{extra_img_str}.png"), reset_img[:, :, ::-1])

        # success img for main task
        if 'sawyer' in env_str:
            exp_ds = env.get_dataset(num_obs=1)
            success_img = env.render('rgb_array', width=args.img_w, height=args.img_h)
        else:
            exp_ds, exp_imgs = hand_dapg_get_dataset(env, env, mode='final', max_num_demos=1,
                                                    get_imgs=True, img_wh=(args.img_w, args.img_h), start_demo=ex_num)
            # terminal offset of 50, so take a random one from the middle
            success_img = exp_imgs[25]

        all_success.append(success_img)
        if not args.vid_save_only:
            cv2.imwrite(os.path.join(args.top_save_dir, f"{env_str}_success{extra_img_str}.png"), success_img[:, :, ::-1])

        if 'sawyer' in env_str:
            del env

        # success imgs for aux tasks
        if not args.no_aux:
            for aux_str in ['reach', 'grasp']:
                aux_env_str = f"{env_str}_{aux_str}"
                if 'sawyer' in env_str:
                    env = load_env(aux_env_str)
                    obs = env.reset()
                    rgb_viewer = env.unwrapped._get_viewer('rgb_array')
                    cam_settings.set_cam_settings(rgb_viewer, cam_settings.CAM_SETTINGS[env_str])

                    exp_ds = env.get_dataset(num_obs=1)
                    success_img = env.render('rgb_array', width=args.img_w, height=args.img_h)
                else:
                    exp_ds, exp_imgs = hand_dapg_get_dataset(env, env, mode=aux_str, max_num_demos=1,
                                                        get_imgs=True, img_wh=(args.img_w, args.img_h), start_demo=ex_num)
                    success_img = exp_imgs[0]

                if aux_str == 'reach':
                    all_reach.append(success_img)
                else:
                    all_grasp.append(success_img)

                if not args.vid_save_only:
                    cv2.imwrite(os.path.join(args.top_save_dir, f"{aux_env_str}_success{extra_img_str}.png"), success_img[:, :, ::-1])

                if 'sawyer' in env_str:
                    del env

        if not 'sawyer' in env_str:
            del env

        if not args.vid_save_only:
            print(f"Saved all images for {env_str}, example number {ex_num}")

    # save vid
    vid_path = os.path.join(args.top_save_dir, "vids")
    os.makedirs(vid_path, exist_ok=True)
    for vid_suf, imgs in zip(['reset', 'reach', 'grasp', 'success'], [all_reset, all_reach, all_grasp, all_success]):
        fig_common.imgs_to_vid(
            vid_path=vid_path, name=f"{env_str}_{str(args.num_ex).zfill(2)}_eps_{vid_suf}", imgs=imgs, frame_rate=args.vid_fps,
            out_fr=30, resolution=args.resolution, crf=args.crf)
    print(f"Finished saving all eps vids for {env_str}")