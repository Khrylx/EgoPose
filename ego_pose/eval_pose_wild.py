import argparse
import os
import sys
import pickle
import math
import time
import glob
import cv2
import yaml
import numpy as np
sys.path.append(os.getcwd())

from ego_pose.utils.metrics import *
from ego_pose.utils.egomimic_config import Config
from ego_pose.utils.pose2d import Pose2DContext
from envs.visual.humanoid_vis import HumanoidVisEnv


parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_1205_vis_single_v1')
parser.add_argument('--multi-vis-model', default='humanoid_1205_vis_estimate_v1')
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--egomimic-cfg', default='cross_01')
parser.add_argument('--statereg-cfg', default='cross_01')
parser.add_argument('--egomimic-iter', type=int, default=6000)
parser.add_argument('--statereg-iter', type=int, default=100)
parser.add_argument('--data', default='wild_01')
parser.add_argument('--take-ind', type=int, default=0)
parser.add_argument('--mode', default='vis')
parser.add_argument('--tpv', action='store_true', default=True)
parser.add_argument('--stats-vis', action='store_true', default=False)
args = parser.parse_args()


cfg = Config(args.egomimic_cfg, False)
dt = 1 / 30.0
data_dir = 'datasets'
meta = yaml.load(open('%s/meta/meta_%s.yml' % (data_dir, args.data)))

res_base_dir = 'results'
em_res_path = '%s/egomimic/%s/results/iter_%04d_%s.p' % (res_base_dir, args.egomimic_cfg, args.egomimic_iter, args.data)
sr_res_path = '%s/statereg/%s/results/iter_%04d_%s.p' % (res_base_dir, args.statereg_cfg, args.statereg_iter, args.data)
em_res, em_meta = pickle.load(open(em_res_path, 'rb')) if args.egomimic_cfg is not None else (None, None)
sr_res, sr_meta = pickle.load(open(sr_res_path, 'rb')) if args.statereg_cfg is not None else (None, None)
takes = list(em_res['traj_pred'].keys())

if args.mode == 'stats':
    pose_ctx = Pose2DContext(cfg)

    def eval_take(res, take):
        pose_dist = 0
        traj_pred = res['traj_pred'][take]
        traj_ub = meta['traj_ub'].get(take, traj_pred.shape[0])
        traj_pred = traj_pred[:traj_ub]
        tpv_offset = meta['tpv_offset'].get(take, cfg.fr_margin)
        flip = meta['tpv_flip'].get(take, False)
        fr_num = traj_pred.shape[0]
        valid_num = 0
        for fr in range(max(0, -tpv_offset), fr_num):
            gt_fr = fr + tpv_offset
            gt_file = '%s/tpv/poses/%s/%05d_keypoints.json' % (data_dir, take, gt_fr)
            gt_p = pose_ctx.load_gt_pose(gt_file)
            if not pose_ctx.check_gt(gt_p):
                # print('invalid frame: %s, %d' % (take, fr))
                continue
            valid_num += 1
            qpos = traj_pred[fr, :]
            p = pose_ctx.align_qpos(qpos, gt_p, flip=flip)
            dist = pose_ctx.get_pose_dist(p, gt_p)
            pose_dist += dist
            if args.stats_vis:
                img = cv2.imread('%s/tpv/s_frames/%s/%05d.jpg' % (data_dir, take, gt_fr))
                pose_ctx.draw_pose(img, p * 0.25, flip=flip)
                cv2.imshow('', img)
                cv2.waitKey(1)
        pose_dist /= valid_num
        vels = get_joint_vels(traj_pred, dt)
        accels = get_joint_accels(vels, dt)
        smoothness = get_mean_abs(accels)
        return pose_dist, smoothness

    def compute_metrics(res, algo):
        if res is None:
            return

        print('=' * 10 + ' %s ' % algo + '=' * 10)
        g_pose_dist = 0
        g_smoothness = 0
        for take in takes:
            pose_dist, smoothness = eval_take(res, take)
            g_pose_dist += pose_dist
            g_smoothness += smoothness
            print('%s - pose dist: %.4f, accels: %.4f' % (take, pose_dist, smoothness))
        g_pose_dist /= len(takes)
        g_smoothness /= len(takes)
        print('-' * 60)
        print('all - pose dist: %.4f, accels: %.4f' % (g_pose_dist, g_smoothness))
        print('-' * 60 + '\n')

    compute_metrics(em_res, 'ego mimic')
    compute_metrics(sr_res, 'state reg')

elif args.mode == 'vis':

    def key_callback(key, action, mods):
        global T, fr, paused, stop, reverse, algo_ind, take_ind, tpv_offset, mfr_int

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
            update_all()
        elif key == glfw.KEY_S:
            reverse = not reverse
        elif key == glfw.KEY_Q:
            tpv_offset -= 1
            update_images()
            print('tpv offset: %d' % tpv_offset)
        elif key == glfw.KEY_E:
            tpv_offset += 1
            update_images()
            print('tpv offset: %d' % tpv_offset)
        elif key == glfw.KEY_Z:
            fr = 0
            take_ind = (take_ind - 1) % len(takes)
            load_take()
            update_all()
        elif key == glfw.KEY_C:
            fr = 0
            take_ind = (take_ind + 1) % len(takes)
            load_take()
            update_all()
        elif key == glfw.KEY_V:
            env_vis.focus = not env_vis.focus
        elif glfw.KEY_1 <= key < glfw.KEY_1 + len(algos):
            algo_ind = key - glfw.KEY_1
            load_take(False)
            update_all()
        elif key == glfw.KEY_MINUS:
            mfr_int -= 1
            update_pose()
        elif key == glfw.KEY_EQUAL:
            mfr_int += 1
            update_pose()
        elif key == glfw.KEY_RIGHT:
            if fr < traj_pred.shape[0] - 1 and fr < len(traj_fpv) - 1:
                fr += 1
            update_all()
        elif key == glfw.KEY_LEFT:
            if fr > 0:
                fr -= 1
            update_all()
        elif key == glfw.KEY_SPACE:
            paused = not paused
        else:
            return False

        return True


    def update_pose():
        print('take_ind: %d, fr: %d, tpv fr: %d, mfr int: %d' % (take_ind, fr, fr + tpv_offset, mfr_int))
        if args.multi:
            nq = 59
            num_model = env_vis.model.nq // nq
            vec = np.array([0, -1])
            for i in range(num_model):
                fr_m = min(fr + i * mfr_int, traj_pred.shape[0] - 1)
                env_vis.data.qpos[i * nq: (i + 1) * nq] = traj_pred[fr_m, :]
                env_vis.data.qpos[i * nq + 3: i * nq + 7] = de_heading(traj_pred[fr_m, 3:7])
                env_vis.data.qpos[i * nq: i * nq + 2] = traj_pred[fr, :2] + vec * 0.8 * i
        else:
            env_vis.data.qpos[:] = traj_pred[fr, :]
        env_vis.sim_forward()


    def update_images():
        cv2.imshow('FPV', traj_fpv[fr])
        if args.tpv:
            cv2.imshow('TPV', traj_tpv[fr + tpv_offset])


    def update_all():
        update_pose()
        update_images()


    def load_take(load_images=True):
        global traj_pred, traj_fpv, traj_tpv, tpv_offset, h_arr, flip
        algo_res, algo = algos[algo_ind]
        if algo_res is None:
            return
        take = takes[take_ind]
        traj_pred = algo_res['traj_pred'][take]
        traj_ub = meta['traj_ub'].get(take, traj_pred.shape[0])
        traj_pred = traj_pred[:traj_ub]
        h_arr = []
        for fr in range(0, traj_pred.shape[0]):
            heading = get_heading(traj_pred[fr, 3:7])
            if heading > np.pi:
                heading -= 2 * np.pi
            elif heading < -np.pi:
                heading += 2 * np.pi
            h_arr.append(heading)
        flip = meta['tpv_flip'].get(take, False)

        if load_images:
            frame_folder = 'datasets/fpv_frames/%s' % take
            frame_files = glob.glob(os.path.join(frame_folder, '*.png'))
            frame_files.sort()
            traj_fpv = [cv2.imread(file) for file in frame_files[cfg.fr_margin:-cfg.fr_margin]]
            if args.tpv:
                frame_folder = 'datasets/tpv/s_frames/%s' % take
                frame_files = glob.glob(os.path.join(frame_folder, '*.jpg'))
                frame_files.sort()
                traj_tpv = [cv2.imread(file) for file in frame_files]
                tpv_offset = meta['tpv_offset'].get(take, cfg.fr_margin)
        print(len(traj_fpv), len(traj_tpv) if args.tpv else 0, traj_pred.shape[0])
        print('%s ---------- %s' % (algo, take))


    traj_pred = None
    traj_fpv = None
    traj_tpv = None
    h_arr = []
    flip = False
    vis_model_file = 'assets/mujoco_models/%s.xml' % (args.multi_vis_model if args.multi else args.vis_model)
    env_vis = HumanoidVisEnv(vis_model_file, 1)
    env_vis.set_custom_key_callback(key_callback)
    algos = [(em_res, 'ego mimic'), (sr_res, 'state reg')]
    algo_ind = 0
    take_ind = args.take_ind
    tpv_offset = 0
    mfr_int = 30
    load_take()

    """render or select part of the clip"""
    cv2.namedWindow('FPV')
    cv2.moveWindow('FPV', 150, 400)
    cv2.namedWindow('TPV')
    glfw.set_window_size(env_vis.viewer.window, 1000, 960)
    glfw.set_window_pos(env_vis.viewer.window, 500, 0)
    env_vis.viewer._hide_overlay = True
    T = 10
    fr = 0
    paused = False
    stop = False
    reverse = False

    update_all()
    t = 0
    while not stop:
        if t >= math.floor(T):
            if not reverse and fr < traj_pred.shape[0] - 1:
                fr += 1
                update_all()
            elif reverse and fr > 0:
                fr -= 1
                update_all()
            t = 0

        env_vis.viewer.cam.azimuth = math.degrees(sum(h_arr[fr - 30:fr]) / 30
                                                  if fr > 30 else sum(h_arr[:30]) / 30) + (0 if flip else 180)
        env_vis.render()
        if not paused:
            t += 1

