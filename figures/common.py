import sys
import os
import numpy as np
from functools import partial

from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import skvideo.io
from PIL import Image, ImageDraw, ImageFont

from rl_sandbox.examples.eval_tools.utils import load_model
import rl_sandbox.constants as c

sys.path.append('..')
import plotting.common as plot_common
import plotting.data_locations as data_locations
import plotting.rce_env_data_locations as rce_env_data_locations
import plotting.hand_dapg_data_locations as hand_dapg_data_locations


PANDA_SETTINGS_DICT = {
    'reach_0': {'last_model': '200000.pt', 'title': 'Reach', 'main': 1},
    'lift_0': {'last_model': '200000.pt', 'title': 'Lift', 'main': 2},
    'move_obj_0': {'last_model': '200000.pt', 'title': 'Move', 'main': 4, 'multi-sqil-model': 2, 'timeout': 40},
    'unstack_stack_env_only_no_move_0': {'last_model': '500000.pt', 'title': 'Unstack-Stack', 'main': 2},
    'stack_no_move_0': {'last_model': '500000.pt', 'title': 'Stack', 'main': 2},
    'bring_no_move_0': {'last_model': '500000.pt', 'title': 'Bring', 'main': 2},
    'insert_no_bring_no_move_0': {'last_model': '1000000.pt', 'title': 'Insert', 'main': 2},
}

SAWYER_HAND_SETTINGS_DICT = {
    'sawyer_drawer_open': {'last_model': '300000.pt', 'ret_suc': 0.1},
    'sawyer_drawer_close': {'last_model': '300000.pt', 'ret_suc': 0.15, 'multi-sqil-model': 2},
    'sawyer_push': {'last_model': '500000.pt', 'ret_suc': 0.2},
    'sawyer_lift': {'last_model': '500000.pt', 'ret_suc': 0.05},
    'sawyer_box_close': {'last_model': '500000.pt', 'ret_suc': 0.2},
    'sawyer_bin_picking': {'last_model': '300000.pt', 'ret_suc': 0.12, 'multi-sqil-model': 2},
    'door-human-v0': {'last_model': '300000.pt', 'ret_suc': 2000, 'timeout': 100},
    'hammer-human-v0': {'last_model': '500000.pt', 'ret_suc': 2500, 'timeout': 100},
    'relocate-human-v0': {'last_model': '1500000.pt', 'ret_suc': 1000},
    'door-human-v0-dp': {'last_model': '500000.pt', 'ret_suc': 1000, 'timeout': 100, 'multi-sqil-model': 5},
    'hammer-human-v0-dp': {'last_model': '1000000.pt', 'ret_suc': 700},
    'relocate-human-v0-najp-dp': {'last_model': '1500000.pt', 'ret_suc': 1000, 'multi-sqil-model': 5},
}


def default_reward(reward, **kwargs):
    return np.array([reward])

def full_path_from_alg_expname(top_path, task, seed, alg_exp_name_str):
    data_path = os.path.join(top_path, task, str(seed), alg_exp_name_str)

    # find datetime folder
    try:
        dirs = sorted([os.path.join(data_path, found) for found in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, found))])
        if len(dirs) > 1:
            print(f"WARNING: multiple folders found at {data_path}, using {dirs[-1]}")
        data_path = dirs[-1]
    except:
        print(f"Error at data_path {data_path}")
        import ipdb; ipdb.set_trace()

    return data_path

def get_aux_reward_success(config, env ):
    if c.AUXILIARY_REWARDS in config:
        auxiliary_reward = config[c.AUXILIARY_REWARDS].reward
        if hasattr(config[c.AUXILIARY_REWARDS], 'set_aux_rewards_str'):
            config[c.AUXILIARY_REWARDS].set_aux_rewards_str()
    else:
        auxiliary_reward = lambda reward, **kwargs: np.array([reward])

    if hasattr(env, 'get_task_successes') and c.AUXILIARY_REWARDS in config and \
            hasattr(config[c.AUXILIARY_REWARDS], '_aux_rewards_str'):
        auxiliary_success = partial(env.get_task_successes, tasks=config[c.AUXILIARY_REWARDS]._aux_rewards_str)
    elif hasattr(env, 'VALID_AUX_TASKS') and auxiliary_reward.__qualname__ in env.VALID_AUX_TASKS:
        auxiliary_success = partial(env.get_task_successes, tasks=[auxiliary_reward.__qualname__])
    elif hasattr(env, 'get_task_successes') and hasattr(env, 'VALID_AUX_TASKS') and \
            (auxiliary_reward.__qualname__ in env.VALID_AUX_TASKS or
             auxiliary_reward.__qualname__ == 'get_aux_reward_success.<locals>.<lambda>'):

        if auxiliary_reward.__qualname__ == 'get_aux_reward_success.<locals>.<lambda>':
            # auxiliary_success = partial(env.get_task_successes, tasks=['main'])
            auxiliary_success = partial(env.get_task_successes, tasks=[env.unwrapped.main_task])
        else:
            auxiliary_success = partial(env.get_task_successes, tasks=[auxiliary_reward.__qualname__])
    else:
        auxiliary_success = None

    return auxiliary_reward, auxiliary_success

def load_model_and_env_once(*args, env=None, **kwargs):
    if env is None:
        config, env, buffer_preprocess, agent = load_model(*args, include_env=True, **kwargs)
        env.seed(args[0])
    else:
        config, buffer_preprocess, agent = load_model(*args, include_env=False, **kwargs)

    return config, env, buffer_preprocess, agent

class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap,
                     transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]

def get_task_settings(task):
    if 'sawyer' in task or 'human' in task:
        task_settings_dict = SAWYER_HAND_SETTINGS_DICT[task]
        if 'sawyer' in task:
            data_loc_dict = rce_env_data_locations.main_performance
        else:
            data_loc_dict = hand_dapg_data_locations.main_performance
        main_task_i = 0
    else:
        task_settings_dict = PANDA_SETTINGS_DICT[task]
        main_task_i = plot_common.PANDA_TASK_SETTINGS[task]['main_task_i']
        data_loc_dict = data_locations.main

    return task_settings_dict, data_loc_dict, main_task_i

def imgs_to_vid(vid_path, name, imgs, frame_rate=20, out_fr=30, hevc_encode=False,
                crf=None, resolution=None, extra_output_args={}):
    # inputdict and outputdict just use regular ffmpeg flags
    input_dict = {"-framerate": f"{frame_rate}"}
    output_dict = {"-pix_fmt": "yuv420p", "-r": str(out_fr)}
    if hevc_encode:
        output_dict["-c:v"] = "libx265"
    if crf is not None:
        output_dict["-crf"] = str(crf)
    if resolution is not None:
        output_dict["-s"] = resolution  # format: widthxheight
    for k, v in extra_output_args.items():
        output_dict[k] = v
    skvideo.io.vwrite(os.path.join(vid_path, name + ".mp4"), np.asarray(imgs),
                    inputdict=input_dict, outputdict=output_dict)

def img_caption(img, text, position=(250, 440), font_size=130):
    image = Image.fromarray(img)

    font = ImageFont.truetype("cmunrm.ttf", font_size)
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = draw.textbbox(position, text, font=font, anchor="mm")
    draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white", outline="black", width=2)
    draw.text(position, text, font=font, fill="black", anchor="mm")

    # image.show()  # for testing
    # import ipdb; ipdb.set_trace()

    return np.array(image)
