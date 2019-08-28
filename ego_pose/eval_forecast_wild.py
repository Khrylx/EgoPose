import argparse
import os
import sys
import pickle
import math
import time
import numpy as np
import yaml
import glob
sys.path.append(os.getcwd())

from ego_pose.utils.metrics import *
from ego_pose.utils.tools import *
from ego_pose.utils.pose2d import Pose2DContext
from ego_pose.utils.egoforecast_config import Config
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from envs.visual.humanoid_vis import HumanoidVisEnv


parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_1205_vis_ghost_v1')
parser.add_argument('--multi-vis-model', default='humanoid_1205_vis_blank_v1')
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--egoforecast-cfg', default='cross_01')
parser.add_argument('--egoforecast-iter', type=int, default=6000)
parser.add_argument('--horizon', type=int, default=30)
parser.add_argument('--data', default='wild_01')
parser.add_argument('--take-ind', type=int, default=0)
parser.add_argument('--mode', default='vis')
parser.add_argument('--tpv', action='store_true', default=True)
parser.add_argument('--stats-vis', action='store_true', default=False)
args = parser.parse_args()


cfg = Config(args.egoforecast_cfg, False)
dt = 1 / 30.0
data_dir = 'datasets'
meta = yaml.safe_load(open('%s/meta/meta_%s.yml' % (data_dir, args.data)))
pose_ctx = Pose2DContext(cfg)

res_base_dir = 'results'
ef_res_path = '%s/egoforecast/%s/results/iter_%04d_%s.p' % (res_base_dir, args.egoforecast_cfg, args.egoforecast_iter, args.data)
ef_res, ef_meta = pickle.load(open(ef_res_path, 'rb')) if args.egoforecast_cfg is not None else (None, None)
# some mocap data we captured have very noisy hands,
# we set hand angles to 0 during training and we do not use it in evaluation
remove_noisy_hands(ef_res)

if args.mode == 'stats':

    def get_kp_dist(traj, take, start_fr):
        pose_dist = 0
        traj_ub = meta['traj_ub'].get(take, None)
        tpv_offset = meta['tpv_offset'].get(take, cfg.fr_margin)
        flip = meta['tpv_flip'].get(take, False)
        fr_num = traj.shape[0]
        valid_num = 0
        for fr in range(fr_num):
            if traj_ub is not None and start_fr + fr >= traj_ub:
                break
            gt_fr = start_fr + fr + tpv_offset
            gt_file = '%s/tpv/poses/%s/%05d_keypoints.json' % (data_dir, take, gt_fr)
            gt_p = pose_ctx.load_gt_pose(gt_file)
            if not pose_ctx.check_gt(gt_p):
                # print('invalid frame: %s, %d' % (take, fr))
                continue
            valid_num += 1
            qpos = traj[fr, :]
            p = pose_ctx.align_qpos(qpos, gt_p, flip=flip)
            dist = pose_ctx.get_pose_dist(p, gt_p)
            pose_dist += dist
            if args.stats_vis:
                img = cv2.imread('%s/tpv/s_frames/%s/%05d.jpg' % (data_dir, take, gt_fr))
                pose_ctx.draw_pose(img, p * 0.25, flip=flip)
                cv2.imshow('', img)
                cv2.waitKey(1)
        pose_dist /= valid_num
        return pose_dist

    def compute_metrics(results, algo, horizon):
        if results is None:
            return

        print('=' * 10 + ' %s ' % algo + '=' * 10)
        g_pose_dist = 0
        g_smoothness = 0
        traj_pred = results['traj_pred']

        for take in traj_pred.keys():
            t_pose_dist = 0
            t_smoothness = 0
            for i in range(traj_pred[take].shape[0]):
                traj = traj_pred[take][i, cfg.fr_margin:cfg.fr_margin + horizon, :]
                kp_dist = get_kp_dist(traj, take, (i + 1) * cfg.fr_margin)
                # compute pred stats
                vels = get_joint_vels(traj, dt)
                accels = get_joint_accels(vels, dt)
                # compute metrics
                smoothness = get_mean_abs(accels)  # exclude root
                t_pose_dist += kp_dist
                t_smoothness += smoothness

            t_pose_dist /= traj_pred[take].shape[0]
            t_smoothness /= traj_pred[take].shape[0]

            print('%s - pose dist: %.4f, accels: %.4f' % (take, t_pose_dist, t_smoothness))

            g_pose_dist += t_pose_dist
            g_smoothness += t_smoothness

        g_pose_dist /= len(traj_pred)
        g_smoothness /= len(traj_pred)

        print('-' * 60)
        print('all - pose dist: %.4f, accels: %.4f' % (g_pose_dist, g_smoothness))
        print('-' * 60 + '\n')

    compute_metrics(ef_res, 'ego forecast', args.horizon)

elif args.mode == 'vis':
    """visualization"""

    def key_callback(key, action, mods):
        global T, fr, paused, stop, reverse, algo_ind, take_ind, traj_ind, ss_ind, mfr_int

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
        elif key == glfw.KEY_Z:
            fr = 0
            traj_ind = 0
            take_ind = (take_ind - 1) % len(takes)
            load_take()
            update_all()
        elif key == glfw.KEY_C:
            fr = 0
            traj_ind = 0
            take_ind = (take_ind + 1) % len(takes)
            load_take()
            update_all()
        elif key == glfw.KEY_Q:
            fr = 0
            traj_ind = (traj_ind - 1) % traj_pred.shape[0]
            update_pose()
        elif key == glfw.KEY_E:
            fr = 0
            traj_ind = (traj_ind + 1) % traj_pred.shape[0]
            update_pose()
        elif key == glfw.KEY_X:
            save_screen_shots(env_vis.viewer.window, 'out/%04d.png' % ss_ind)
            ss_ind += 1
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
            if fr < traj_pred.shape[1] - 1 and fr < len(traj_fpv) - 1:
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
        print('take_ind: %d, traj ind: %d, tpv fr: %d, mfr int: %d' % (take_ind, traj_ind, traj_ind * cfg.fr_margin + tpv_offset, mfr_int))
        if args.multi:
            fr_s = cfg.fr_margin
            nq = 59
            num_model = env_vis.model.nq // nq
            vec = np.array([0, -1])
            for i in range(num_model):
                fr_m = max(0, min(fr_s + (i - 3) * mfr_int, traj_pred.shape[1] - 1))
                env_vis.data.qpos[i * nq: (i + 1) * nq] = traj_pred[traj_ind, fr_m, :]
                env_vis.data.qpos[i * nq + 3: i * nq + 7] = de_heading(traj_pred[traj_ind, fr_m, 3:7])
                env_vis.data.qpos[i * nq: i * nq + 2] = traj_pred[traj_ind, fr_s, :2] + vec * 0.8 * i
        else:
            nq = env_vis.model.nq // 2
            if fr >= cfg.fr_margin:
                env_vis.data.qpos[:nq] = traj_pred[traj_ind, fr, :]
                env_vis.data.qpos[nq:] = traj_pred[traj_ind, cfg.fr_margin, :]
            else:
                env_vis.data.qpos[:nq] = traj_pred[traj_ind, fr, :]
                env_vis.data.qpos[nq:] = traj_pred[traj_ind, fr, :]
        env_vis.sim_forward()


    def update_images():
        cv2.imshow('FPV', traj_fpv[traj_ind * cfg.fr_margin + min(fr, cfg.fr_margin) + 10])
        if args.tpv:
            cv2.imshow('TPV', traj_tpv[traj_ind * cfg.fr_margin + min(fr, cfg.fr_margin) + tpv_offset])


    def update_all():
        update_pose()
        update_images()


    def load_take(load_images=True):
        global traj_pred, traj_fpv, traj_tpv, tpv_offset
        algo_res, algo = algos[algo_ind]
        if algo_res is None:
            return
        take = takes[take_ind]
        traj_pred = algo_res['traj_pred'][take]
        traj_ub = meta['traj_ub'].get(take, traj_pred.shape[0])
        traj_pred = traj_pred[:traj_ub]
        if load_images:
            frame_folder = 'datasets/fpv_frames/%s' % take
            frame_files = glob.glob(os.path.join(frame_folder, '*.png'))
            frame_files.sort()
            traj_fpv = [cv2.imread(file) for file in frame_files]
            if args.tpv:
                frame_folder = 'datasets/tpv/s_frames/%s' % take
                frame_files = glob.glob(os.path.join(frame_folder, '*.jpg'))
                frame_files.sort()
                traj_tpv = [cv2.imread(file) for file in frame_files]
                tpv_offset = meta['tpv_offset'][take]
        print(len(traj_fpv), len(traj_tpv) if args.tpv else 0, traj_pred.shape[0])
        print('%s ---------- %s' % (algo, take))

    traj_pred = None
    traj_fpv = None
    traj_tpv = None
    vis_model_file = 'assets/mujoco_models/%s.xml' % (args.multi_vis_model if args.multi else args.vis_model)
    env_vis = HumanoidVisEnv(vis_model_file, 1)
    env_vis.set_custom_key_callback(key_callback)
    takes = list(ef_res['traj_pred'].keys())
    algos = [(ef_res, 'ego forecast')]
    algo_ind = 0
    take_ind = args.take_ind
    traj_ind = 0
    tpv_offset = 0
    ss_ind = 0
    mfr_int = 10
    load_take()

    """render or select part of the clip"""
    cv2.namedWindow('FPV')
    cv2.moveWindow('FPV', 150, 400)
    cv2.namedWindow('TPV')
    glfw.set_window_size(env_vis.viewer.window, 1000, 960)
    glfw.set_window_pos(env_vis.viewer.window, 500, 0)
    env_vis.viewer._hide_overlay = True
    env_vis.viewer.cam.distance = 6
    T = 10
    fr = 0
    paused = False
    stop = False
    reverse = False

    update_all()
    t = 0
    while not stop:
        if t >= math.floor(T):
            if not reverse and fr < traj_pred.shape[1] - 1:
                fr += 1
                update_all()
            elif reverse and fr > 0:
                fr -= 1
                update_all()
            t = 0

        heading = get_heading(traj_pred[traj_ind, 0, 3:7])
        flip = meta['tpv_flip'].get(takes[take_ind], False)
        env_vis.viewer.cam.azimuth = math.degrees(heading) + (0 if flip else 180)
        env_vis.render()

        if not paused:
            t += 1

