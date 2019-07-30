import argparse
import os
import sys
import pickle
sys.path.append(os.getcwd())

from utils import *
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from ego_pose.utils.statereg_dataset import Dataset
from ego_pose.utils.egomimic_config import Config as EgoConfig

parser = argparse.ArgumentParser()
parser.add_argument('--meta-id', default=None)
parser.add_argument('--out-id', default=None)
args = parser.parse_args()

cfg_dict = {
    'meta_id': args.meta_id,
    'mujoco_model': 'humanoid_1205_v1',
    'vis_model': 'humanoid_1205_vis',
    'obs_coord': 'heading',
}
cfg = EgoConfig(None, create_dirs=False, cfg_dict=cfg_dict)
env = HumanoidEnv(cfg)
dataset = Dataset(args.meta_id, 'all', 0, 'iter', False, 0)


def get_expert(expert_qpos, lb, ub):
    expert = {'qpos': expert_qpos}
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', 'head_pos', 'obs', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel'}
    for key in feat_keys:
        expert[key] = []

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        # remove noisy hand data
        qpos[slice(*env.body_qposaddr['LeftHand'])] = 0.0
        qpos[slice(*env.body_qposaddr['RightHand'])] = 0.0
        env.data.qpos[:] = qpos
        env.sim.forward()
        rq_rmh = de_heading(qpos[3:7])
        obs = env.get_obs()
        ee_pos = env.get_ee_pos(env.cfg.obs_coord)
        ee_wpos = env.get_ee_pos(None)
        bquat = env.get_body_quat()
        com = env.get_com()
        head_pos = env.get_body_com('Head').copy()
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd(prev_qpos, qpos, env.dt)
            rlinv = qvel[:3].copy()
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7], env.cfg.obs_coord)
            rangv = qvel[3:6].copy()
            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        expert['obs'].append(obs)
        expert['ee_pos'].append(ee_pos)
        expert['ee_wpos'].append(ee_wpos)
        expert['bquat'].append(bquat)
        expert['com'].append(com)
        expert['head_pos'].append(head_pos)
        expert['rq_rmh'].append(rq_rmh)
    expert['qvel'].insert(0, expert['qvel'][0].copy())
    expert['rlinv'].insert(0, expert['rlinv'][0].copy())
    expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
    expert['rangv'].insert(0, expert['rangv'][0].copy())
    # get expert body quaternions
    for i in range(expert_qpos.shape[0]):
        if i > 0:
            bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
            expert['bangvel'].append(bangvel)
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    expert['qpos'] = expert['qpos'][lb:ub, :]
    for key in feat_keys:
        expert[key] = np.vstack(expert[key][lb:ub])
    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pos'][:, 2].min()
    return expert


num_sample = 0
expert_dict = {}
for i in range(len(dataset.takes)):
    take = dataset.takes[i]
    _, lb, ub = dataset.msync[take]
    expert_qpos = dataset.orig_trajs[i]
    expert = get_expert(expert_qpos, lb, ub)
    expert_dict[take] = expert
    num_sample += expert['len']
    print(take, expert['len'], expert['qvel'].min(), expert['qvel'].max(), expert['head_height_lb'])

print('meta: %s, total sample: %d, dataset length: %d' % (args.meta_id, num_sample, dataset.len))

path = 'datasets/features/expert_%s.p' % args.out_id
pickle.dump(expert_dict, open(path, 'wb'))


