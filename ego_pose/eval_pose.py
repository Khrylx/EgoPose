import argparse
import os
import sys
import pickle
import math
import time
import numpy as np
sys.path.append(os.getcwd())

from ego_pose.utils.metrics import *
from ego_pose.utils.tools import *
from ego_pose.utils.egomimic_config import Config
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from envs.visual.humanoid_vis import HumanoidVisEnv


parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_1205_vis_double_v1')
parser.add_argument('--multi-vis-model', default='humanoid_1205_vis_estimate_v1')
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--egomimic-cfg', default='subject_03')
parser.add_argument('--statereg-cfg', default='subject_03')
parser.add_argument('--egomimic-iter', type=int, default=3000)
parser.add_argument('--statereg-iter', type=int, default=100)
parser.add_argument('--egomimic-tag', default='')
parser.add_argument('--data', default='test')
parser.add_argument('--mode', default='vis')
args = parser.parse_args()


def compute_metrics(results, meta, algo):
    if results is None:
        return

    print('=' * 10 + ' %s ' % algo + '=' * 10)
    g_pose_dist = 0
    g_vel_dist = 0
    g_smoothness = 0
    traj_orig = results['traj_orig']
    traj_pred = results['traj_pred']

    for take in traj_pred.keys():
        traj = traj_pred[take]
        traj_gt = traj_orig[take]
        # compute gt stats
        angs_gt = get_joint_angles(traj_gt)
        vels_gt = get_joint_vels(traj_gt, dt)
        # compute pred stats
        angs = get_joint_angles(traj)
        vels = get_joint_vels(traj, dt)
        accels = get_joint_accels(vels, dt)
        # compute metrics
        pose_dist = get_mean_dist(angs, angs_gt)
        vel_dist = get_mean_dist(vels, vels_gt)
        smoothness = get_mean_abs(accels)
        g_pose_dist += pose_dist
        g_vel_dist += vel_dist
        g_smoothness += smoothness

        # print('%s - pose dist: %.4f, vel dist: %.4f, accels: %.4f' %
        #       (take, pose_dist, vel_dist, smoothness))

    g_pose_dist /= len(traj_pred)
    g_vel_dist /= len(traj_pred)
    g_smoothness /= len(traj_pred)

    print('-' * 60)
    print('all - pose dist: %.4f, vel dist: %.4f, accels: %.4f' % (g_pose_dist, g_vel_dist, g_smoothness))
    print('-' * 60 + '\n')


cfg = Config(args.egomimic_cfg)
dt = 1 / 30.0

res_base_dir = 'results'
em_res_path = '%s/egomimic/%s/results/iter_%04d_%s%s.p' % (res_base_dir, args.egomimic_cfg, args.egomimic_iter, args.data, args.egomimic_tag)
sr_res_path = '%s/statereg/%s/results/iter_%04d_%s.p' % (res_base_dir, args.statereg_cfg, args.statereg_iter, args.data)
em_res, em_meta = pickle.load(open(em_res_path, 'rb')) if args.egomimic_cfg is not None else (None, None)
sr_res, sr_meta = pickle.load(open(sr_res_path, 'rb')) if args.statereg_cfg is not None else (None, None)
# some mocap data we captured have very noisy hands,
# we set hand angles to 0 during training and we do not use it in evaluation
remove_noisy_hands(em_res)
remove_noisy_hands(sr_res)

if args.mode == 'stats':
    compute_metrics(em_res, em_meta, 'ego mimic')
    compute_metrics(sr_res, sr_meta, 'state reg')

elif args.mode == 'vis':
    """visualization"""

    def key_callback(key, action, mods):
        global T, fr, paused, stop, reverse, algo_ind, take_ind, ss_ind, show_gt, mfr_int

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            T *= 1.5
        elif key == glfw.KEY_F:
            T = max(1, T / 1.5)
        elif key == glfw.KEY_R:
            stop = True
        elif key == glfw.KEY_W:
            fr = 0
            update_pose()
        elif key == glfw.KEY_S:
            reverse = not reverse
        elif key == glfw.KEY_Z:
            fr = 0
            take_ind = (take_ind - 1) % len(takes)
            load_take()
            update_pose()
        elif key == glfw.KEY_C:
            fr = 0
            take_ind = (take_ind + 1) % len(takes)
            load_take()
            update_pose()
        elif key == glfw.KEY_X:
            save_screen_shots(env_vis.viewer.window, 'out/%04d.png' % ss_ind)
            ss_ind += 1
        elif glfw.KEY_1 <= key < glfw.KEY_1 + len(algos):
            algo_ind = key - glfw.KEY_1
            load_take()
            update_pose()
        elif key == glfw.KEY_0:
            show_gt = not show_gt
            update_pose()
        elif key == glfw.KEY_MINUS:
            mfr_int -= 1
            update_pose()
        elif key == glfw.KEY_EQUAL:
            mfr_int += 1
            update_pose()
        elif key == glfw.KEY_RIGHT:
            if fr < traj_orig.shape[0] - 1:
                fr += 1
            update_pose()
        elif key == glfw.KEY_LEFT:
            if fr > 0:
                fr -= 1
            update_pose()
        elif key == glfw.KEY_SPACE:
            paused = not paused
        else:
            return False

        return True

    def update_pose():
        print('take_ind: %d, fr: %d, mfr int: %d' % (take_ind, fr, mfr_int))
        if args.multi:
            nq = 59
            traj = traj_orig if show_gt else traj_pred
            num_model = env_vis.model.nq // nq
            hq = get_heading_q(traj_orig[fr, 3:7])
            rel_q = quaternion_multiply(hq, quaternion_inverse(get_heading_q(traj[fr, 3:7])))
            vec = quat_mul_vec(hq, np.array([0, -1, 0]))[:2]
            for i in range(num_model):
                fr_m = min(fr + i * mfr_int, traj.shape[0] - 1)
                env_vis.data.qpos[i*nq: (i+1)*nq] = traj[fr_m, :]
                env_vis.data.qpos[i*nq + 3: i*nq + 7] = quaternion_multiply(rel_q, traj[fr_m, 3:7])
                env_vis.data.qpos[i*nq: i*nq + 2] = traj_orig[fr, :2] + vec * 0.8 * i
        else:
            nq = env_vis.model.nq // 2
            env_vis.data.qpos[:nq] = traj_pred[fr, :]
            env_vis.data.qpos[nq:] = traj_orig[fr, :]
            # add x offset
            env_vis.data.qpos[nq] += 1.0
        env_vis.sim_forward()

    def load_take():
        global traj_pred, traj_orig
        algo_res, algo = algos[algo_ind]
        if algo_res is None:
            return
        take = takes[take_ind]
        print('%s ---------- %s' % (algo, take))
        traj_pred = algo_res['traj_pred'][take]
        traj_orig = algo_res['traj_orig'][take]

    traj_pred = None
    traj_orig = None
    vis_model_file = 'assets/mujoco_models/%s.xml' % (args.multi_vis_model if args.multi else args.vis_model)
    env_vis = HumanoidVisEnv(vis_model_file, 1, focus=not args.multi)
    env_vis.set_custom_key_callback(key_callback)
    takes = cfg.takes[args.data]
    algos = [(em_res, 'ego mimic'), (sr_res, 'state reg')]
    algo_ind = 0
    take_ind = 0
    ss_ind = 0
    mfr_int = 10
    show_gt = False
    load_take()

    """render or select part of the clip"""
    T = 10
    fr = 0
    paused = False
    stop = False
    reverse = False

    update_pose()
    t = 0
    while not stop:
        if t >= math.floor(T):
            if not reverse and fr < traj_orig.shape[0] - 1:
                fr += 1
                update_pose()
            elif reverse and fr > 0:
                fr -= 1
                update_pose()
            t = 0

        env_vis.render()
        if not paused:
            t += 1

