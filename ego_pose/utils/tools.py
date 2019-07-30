import numpy as np
from utils.math import *


def normalize_traj(qpos_traj, qvel_traj):
    new_qpos_traj = []
    new_qvel_traj = []
    for qpos, qvel in zip(qpos_traj, qvel_traj):
        new_qpos = qpos.copy()
        new_qvel = qvel.copy()
        new_qvel[:3] = transform_vec(qvel[:3], qpos[3:7], 'heading')
        new_qpos[3:7] = de_heading(qpos[3:7])
        new_qpos_traj.append(new_qpos)
        new_qvel_traj.append(new_qvel)
    return np.vstack(new_qpos_traj), np.vstack(new_qvel_traj)


def sync_traj(qpos_traj, qvel_traj, ref_qpos):
    new_qpos_traj = []
    new_qvel_traj = []
    rel_heading = quaternion_multiply(get_heading_q(ref_qpos[3:7]), quaternion_inverse(get_heading_q(qpos_traj[0, 3:7])))
    ref_pos = ref_qpos[:3]
    start_pos = np.concatenate((qpos_traj[0, :2], ref_pos[[2]]))
    for qpos, qvel in zip(qpos_traj, qvel_traj):
        new_qpos = qpos.copy()
        new_qvel = qvel.copy()
        new_qpos[:2] = quat_mul_vec(rel_heading, qpos[:3] - start_pos)[:2] + ref_pos[:2]
        new_qpos[3:7] = quaternion_multiply(rel_heading, qpos[3:7])
        new_qvel[:3] = quat_mul_vec(rel_heading, qvel[:3])
        new_qpos_traj.append(new_qpos)
        new_qvel_traj.append(new_qvel)
    return np.vstack(new_qpos_traj), np.vstack(new_qvel_traj)


def remove_noisy_hands(results):
    for traj in results.values():
        for take in traj.keys():
            traj[take][..., 32:35] = 0
            traj[take][..., 42:45] = 0
    return
