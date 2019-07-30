import numpy as np
import os
import shutil
from os import path
from os import listdir
from PIL import Image
from OpenGL import GL
from gym.envs.mujoco.mujoco_env import MujocoEnv
from utils.math import *
import glfw
import cv2


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))


def out_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../out'))


def log_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../logs'))


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


def load_img(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        I = Image.open(f)
        img = I.resize((224, 224), Image.ANTIALIAS).convert('RGB')
        return img


def save_screen_shots(window, file_name, transparent=False):
    import pyautogui
    xpos, ypos = glfw.get_window_pos(window)
    width, height = glfw.get_window_size(window)
    image = pyautogui.screenshot(region=(xpos*2, ypos*2, width*2, height*2))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA if transparent else cv2.COLOR_RGB2BGR)
    if transparent:
        image[np.all(image >= [240, 240, 240, 240], axis=2)] = [255, 255, 255, 0]
    cv2.imwrite(file_name, image)


"""mujoco helper"""


def get_body_qposaddr(model):
    body_qposaddr = dict()
    for i, body_name in enumerate(model.body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr


def align_human_state(qpos, qvel, ref_qpos):
    qpos[:2] = ref_qpos[:2]
    hq = get_heading_q(ref_qpos[3:7])
    qpos[3:7] = quaternion_multiply(hq, qpos[3:7])
    qvel[:3] = quat_mul_vec(hq, qvel[:3])
