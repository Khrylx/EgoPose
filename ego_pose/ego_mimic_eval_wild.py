import argparse
import os
import sys
import pickle
import time
sys.path.append(os.getcwd())

from utils import *
from core.policy_gaussian import PolicyGaussian
from core.critic import Value
from models.mlp import MLP
from models.video_state_net import VideoStateNet
from models.video_reg_net import VideoRegNet
from envs.visual.humanoid_vis import HumanoidVisEnv
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from ego_pose.utils.egomimic_config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--test-ind', type=int, default=-1)
parser.add_argument('--test-feat', default=None)
parser.add_argument('--show-noise', action='store_true', default=False)
args = parser.parse_args()
cfg = Config(args.cfg, create_dirs=False)

dtype = torch.float64
torch.set_default_dtype(dtype)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval_wild.txt'))

"""test data"""
cnn_feat_file = '%s/features/cnn_feat_%s.p' % (cfg.data_dir, args.test_feat)
cnn_feat_dict, _ = pickle.load(open(cnn_feat_file, 'rb'))
takes = list(cnn_feat_dict.keys())

"""environment"""
env = HumanoidEnv(cfg)
env_vis = HumanoidVisEnv(cfg.mujoco_model_file, 10)
env.seed(cfg.seed)
cnn_feat_dim = cnn_feat_dict[takes[0]].shape[-1]
actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
body_qposaddr = get_body_qposaddr(env.model)

"""load policy net"""
policy_vs_net = VideoStateNet(cnn_feat_dim, cfg.policy_v_hdim, cfg.fr_margin, cfg.policy_v_net, cfg.policy_v_net_param, cfg.causal)
value_vs_net = VideoStateNet(cnn_feat_dim, cfg.value_v_hdim, cfg.fr_margin, cfg.value_v_net, cfg.value_v_net_param, cfg.causal)
policy_net = PolicyGaussian(MLP(state_dim + cfg.policy_v_hdim, cfg.policy_hsize, cfg.policy_htype), action_dim,
                            log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim + cfg.value_v_hdim, cfg.value_hsize, cfg.value_htype))
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
logger.info('loading policy net from checkpoint: %s' % cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
policy_vs_net.load_state_dict(model_cp['policy_vs_dict'])
value_net.load_state_dict(model_cp['value_dict'])
value_vs_net.load_state_dict(model_cp['value_vs_dict'])
running_state = model_cp['running_state']
value_stat = RunningStat(1)

"""load state net"""
cp_path = cfg.state_net_model
logger.info('loading state net from checkpoint: %s' % cp_path)
model_cp, meta = pickle.load(open(cp_path, "rb"))
state_net_mean, state_net_std, state_net_cfg = meta['mean'], meta['std'], meta['cfg']
state_net = VideoRegNet(state_net_mean.size, state_net_cfg.v_hdim, cnn_feat_dim, no_cnn=True,
                        cnn_type=state_net_cfg.cnn_type, mlp_dim=state_net_cfg.mlp_dim, v_net_type=state_net_cfg.v_net,
                        v_net_param=state_net_cfg.v_net_param, causal=state_net_cfg.causal)
state_net.load_state_dict(model_cp['state_net_dict'])
to_test(policy_vs_net, policy_net, value_vs_net, value_net, state_net)


def render():
    env_vis.data.qpos[:] = env.data.qpos
    env_vis.sim_forward()
    env_vis.render()


def reset_env_state(state, ref_qpos):
    qpos = ref_qpos.copy()
    qpos[2:] = state[:qpos.size - 2]
    qvel = state[qpos.size - 2:]
    align_human_state(qpos, qvel, ref_qpos)
    env.set_state(qpos, qvel)
    return env.get_obs()


def eval_take(take):
    logger.info('Testing on %s' % take)

    traj_pred = []
    vel_pred = []
    cnn_feat = cnn_feat_dict[take]
    data_len = cnn_feat.shape[0]
    test_len = data_len - 2 * cfg.fr_margin
    state = env.reset()
    cnn_feat = tensor(cnn_feat)
    policy_vs_net.initialize(cnn_feat)
    value_vs_net.initialize(cnn_feat)
    state_pred = state_net(cnn_feat.unsqueeze(1)).squeeze(1)[cfg.fr_margin: -cfg.fr_margin, :].numpy()
    state_pred = state_pred * state_net_std[None, :] + state_net_mean[None, :]

    state = reset_env_state(state_pred[0, :], env.data.qpos)
    if running_state is not None:
        state = running_state(state, update=False)

    for t in range(test_len):
        traj_pred.append(env.data.qpos.copy())
        vel_pred.append(env.data.qvel.copy())

        if args.render:
            render()

        """learner policy"""
        state_var = tensor(state, dtype=dtype).unsqueeze(0)
        policy_vs_out = policy_vs_net(state_var)
        value_vs_out = value_vs_net(state_var)
        value = value_net(value_vs_out).item()
        value_stat.push(np.array([value]))

        action = policy_net.select_action(policy_vs_out, mean_action=not args.show_noise)[0].numpy()
        next_state, reward, done, info = env.step(action)
        if running_state is not None:
            next_state = running_state(next_state, update=False)

        if value < 0.6 * value_stat.mean[0]:
            logger.info('reset state!')
            state = reset_env_state(state_pred[t+1, :], env.data.qpos)
            if running_state is not None:
                state = running_state(state, update=False)
        else:
            state = next_state

    return np.vstack(traj_pred), np.vstack(vel_pred)


# if test_ind is defined, then keeping visualizing this trajectory
if args.test_ind >= 0:
    for i in range(100):
        eval_take(takes[args.test_ind])
else:
    traj_pred = {}
    vel_pred = {}
    for take in takes:
        traj_pred[take], vel_pred[take] = eval_take(take)
    results = {'traj_pred': traj_pred, 'vel_pred': vel_pred}
    meta = {'algo': 'ego_mimic'}
    res_path = '%s/iter_%04d_%s.p' % (cfg.result_dir, args.iter, args.test_feat)
    pickle.dump((results, meta), open(res_path, 'wb'))
    logger.info('saved results to %s' % res_path)
