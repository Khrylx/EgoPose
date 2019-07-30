from utils import *


def quat_space_reward_v3(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv = ws.get('w_p', 0.5), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_rp', 0.1), ws.get('w_rv', 0.1)
    k_p, k_v, k_e = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20)
    k_rh, k_rq, k_rl, k_ra = ws.get('k_rh', 300), ws.get('k_rq', 300), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # root position reward
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(-k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2))
    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(-k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * root_pose_reward + w_rv * root_vel_reward
    reward /= w_p + w_v + w_e + w_rp + w_rv
    if ws.get('decay', False):
        reward *= 1.0 - t / cfg.env_episode_len
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, root_pose_reward, root_vel_reward])


def constant_reward(env, state, action, info):
    reward = 1.0
    if info['end']:
        reward += env.end_reward
    return 1.0, np.zeros(1)


def pose_dist_reward(env, state, action, info):
    pose_dist = env.get_pose_dist()
    reward = 5.0 - 3.0 * pose_dist
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_dist])


reward_func = {'quat_v3': quat_space_reward_v3,
               'constant': constant_reward,
               'pose_dist': pose_dist_reward}

