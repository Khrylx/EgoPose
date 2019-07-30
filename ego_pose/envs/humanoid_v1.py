import numpy as np
from envs.common import mujoco_env
from gym import spaces
from utils import *
from utils.transformation import quaternion_from_euler, rotation_from_quaternion, quaternion_about_axis
from mujoco_py import functions as mjf
import pickle
import time
import cv2 as cv
from scipy.linalg import cho_solve, cho_factor


class HumanoidEnv(mujoco_env.MujocoEnv):

    def __init__(self, cfg):
        mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file, 15)
        self.cfg = cfg
        # visualization
        self.save_video = False
        self.video_res = (224, 224)
        self.video_dir = cfg.video_dir
        self.set_cam_first = set()
        self.subsample_rate = 1
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.set_spaces()
        self.set_model_params()
        # expert
        self.expert_ind = None
        self.expert_id = None
        self.expert_list = None     # name only
        self.expert_arr = None      # store actual experts
        self.expert = None
        self.cnn_feat = None
        # fixed sampling
        self.fix_expert_ind = None
        self.fix_start_ind = None
        self.fix_len = None
        self.fix_start_state = None
        self.fix_cnn_feat = None
        self.fix_head_lb = None

    def load_experts(self, expert_list, expert_feat_file, cnn_feat_file):
        self.expert_ind = 0
        self.expert_list = expert_list
        expert_dict = pickle.load(open(expert_feat_file, 'rb'))
        self.expert_arr = [expert_dict[x] for x in self.expert_list]
        self.set_expert(0)
        cnn_feat_dict, _ = pickle.load(open(cnn_feat_file, 'rb'))
        self.cnn_feat = [cnn_feat_dict[x] for x in self.expert_list]

    def set_model_params(self):
        if self.cfg.action_type == 'torque' and hasattr(self.cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cfg.j_stiff
            self.model.dof_damping[6:] = self.cfg.j_damp

    def set_spaces(self):
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(low=bounds[:, 0], high=bounds[:, 1], dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self):
        obs = self.get_full_obs()
        return obs

    def get_full_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full':
            obs.append(qvel)
        # phase
        if hasattr(self.cfg, 'obs_phase') and self.cfg.obs_phase:
            phase = min(self.cur_t / self.cfg.env_episode_len, 1.0)
            obs.append(np.array([phase]))
        obs = np.concatenate(obs)
        return obs

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_body_quat(self):
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.model.body_names[1:]:
            if body == 'Hips':
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self):
        return self.data.subtree_com[0, :].copy()

    def compute_desired_accel(self, qpos_err, qvel_err):
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(self.model.nv, self.model.nv)
        C = self.data.qfrc_bias.copy()
        k_p = np.zeros(nv)
        k_d = np.zeros(nv)
        k_p[6:] = self.cfg.jkp
        k_d[6:] = self.cfg.jkd
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d*dt), -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]))
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] - ctrl))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err)
        qvel_err += q_accel * dt
        torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]
        return torque

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            if self.save_video and i % self.subsample_rate == 0:
                img = self.render('image', *self.video_res)
                vind = self.cur_t * (n_frames // self.subsample_rate) + i // self.subsample_rate
                cv.imwrite('%s/%04d.png' % (self.video_dir, vind), img)

            ctrl = cfg.a_ref + action * cfg.a_scale
            if cfg.action_type == 'position':
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == 'torque':
                torque = ctrl
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque
            self.sim.step()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()
        # do simulation
        self.do_simulation(a, self.frame_skip)
        self.cur_t += 1
        self.bquat = self.get_body_quat()
        self.sync_expert()
        # get obs
        head_pos = self.get_body_com('Head')
        reward = 1.0
        if self.fix_head_lb is not None:
            fail = head_pos[2] < self.fix_head_lb
        else:
            fail = self.expert is not None and head_pos[2] < self.expert['head_height_lb'] - 0.1
        end = self.cur_t >= (cfg.env_episode_len if self.fix_len is None else self.fix_len)
        done = fail or end
        return self.get_obs(), reward, done, {'fail': fail, 'end': end}

    def reset_model(self):
        if self.fix_start_state is not None:
            init_pose = self.fix_start_state[:self.model.nq]
            init_vel = self.fix_start_state[self.model.nq:]
            self.set_state(init_pose, init_vel)
        elif self.expert_list is not None:
            cfg = self.cfg
            fr_margin = cfg.fr_margin
            # sample expert
            expert_ind = self.np_random.randint(len(self.expert_list)) if self.fix_expert_ind is None else self.fix_expert_ind
            self.set_expert(expert_ind)
            # sample start frame
            if self.fix_start_ind is None:
                ind = 0 if cfg.env_start_first else self.np_random.randint(fr_margin, self.expert['len'] - cfg.env_episode_len - fr_margin)
            else:
                ind = self.fix_start_ind
            self.start_ind = ind
            if hasattr(cfg, 'random_cur_t') and cfg.random_cur_t:
                self.cur_t = np.random.randint(cfg.env_episode_len)
                ind += self.cur_t
            init_pose = self.expert['qpos'][ind, :].copy()
            init_vel = self.expert['qvel'][ind, :].copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.sync_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.data.qpos[0:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def set_fix_sampling(self, expert_ind=None, start_ind=None, len=None, start_state=None, cnn_feat=None):
        self.fix_expert_ind = expert_ind
        self.fix_start_ind = start_ind
        self.fix_len = len
        self.fix_start_state = start_state
        self.fix_cnn_feat = cnn_feat

    def set_fix_head_lb(self, fix_head_lb=None):
        self.fix_head_lb = fix_head_lb

    def sync_expert(self):
        if self.expert is not None and self.cur_t % self.cfg.sync_exp_interval == 0:
            expert = self.expert
            ind = self.get_expert_index(self.cur_t)
            e_qpos = self.get_expert_attr('qpos', ind).copy()
            expert['rel_heading'] = quaternion_multiply(get_heading_q(self.data.qpos[3:7]),
                                                        quaternion_inverse(get_heading_q(e_qpos[3:7])))
            expert['start_pos'] = e_qpos[:3]
            expert['sim_pos'] = np.concatenate((self.data.qpos[:2], np.array([e_qpos[2]])))

    def set_expert(self, expert_ind):
        self.expert_ind = expert_ind
        self.expert_id = self.expert_list[expert_ind]
        self.expert = self.expert_arr[expert_ind]

    def get_expert_index(self, t):
        return self.start_ind + t

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]

    def get_pose_dist(self):
        ind = self.get_expert_index(self.cur_t)
        qpos_e = self.expert['qpos'][ind, :]
        qpos_g = self.data.qpos
        diff = qpos_e - qpos_g
        return np.linalg.norm(diff[2:])

    def get_pose_diff(self):
        ind = self.get_expert_index(self.cur_t)
        qpos_e = self.expert['qpos'][ind, :]
        qpos_g = self.data.qpos
        diff = qpos_e - qpos_g
        return np.abs(diff[2:])

    def get_episode_cnn_feat(self):
        fm = self.cfg.fr_margin
        num_fr = self.cfg.env_episode_len if self.fix_len is None else self.fix_len
        return self.cnn_feat[self.expert_ind][self.start_ind - fm: self.start_ind + num_fr + fm, :] \
            if self.fix_cnn_feat is None else self.fix_cnn_feat


