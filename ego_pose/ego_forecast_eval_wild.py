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
from models.video_forecast_net import VideoForecastNet
from models.video_reg_net import VideoRegNet
from envs.visual.humanoid_vis import HumanoidVisEnv
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from ego_pose.utils.egoforecast_config import Config
from ego_pose.utils.egomimic_config import Config as EgoMimicConfig
from ego_pose.utils.tools import *


parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_1205_vis_forecast_v1')
parser.add_argument('--cfg', default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--test-feat', default=None)
parser.add_argument('--test-ind', type=int, default=0)
parser.add_argument('--start-ind', type=int, default=None)
parser.add_argument('--show-noise', action='store_true', default=False)
parser.add_argument('--mode', default='save')

args = parser.parse_args()
cfg = Config(args.cfg, create_dirs=False)
if hasattr(cfg, 'random_cur_t'):
    cfg.random_cur_t = False

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
env_vis = HumanoidVisEnv('assets/mujoco_models/%s.xml' % args.vis_model, 10)
env.seed(cfg.seed)
cnn_feat_dim = cnn_feat_dict[takes[0]].shape[-1]
actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
body_qposaddr = get_body_qposaddr(env.model)

"""load policy net"""
policy_vs_net = VideoForecastNet(cnn_feat_dim, state_dim, cfg.policy_v_hdim, cfg.fr_margin, cfg.policy_v_net,
                                 cfg.policy_v_net_param, cfg.policy_s_hdim, cfg.policy_s_net)
value_vs_net = VideoForecastNet(cnn_feat_dim, state_dim, cfg.value_v_hdim, cfg.fr_margin, cfg.value_v_net,
                                cfg.value_v_net_param, cfg.value_s_hdim, cfg.value_s_net)
policy_net = PolicyGaussian(MLP(policy_vs_net.out_dim, cfg.policy_hsize, cfg.policy_htype), action_dim,
                            log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(value_vs_net.out_dim, cfg.value_hsize, cfg.value_htype))
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
logger.info('loading policy net from checkpoint: %s' % cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
policy_vs_net.load_state_dict(model_cp['policy_vs_dict'])
value_net.load_state_dict(model_cp['value_dict'])
value_vs_net.load_state_dict(model_cp['value_vs_dict'])
running_state = model_cp['running_state']
value_stat = RunningStat(1)
to_test(policy_vs_net, policy_net, value_vs_net, value_net)

"""load ego mimic results"""
em_cfg = EgoMimicConfig(cfg.ego_mimic_cfg)
em_res_path = '%s/iter_%04d_%s.p' % (em_cfg.result_dir, cfg.ego_mimic_iter, args.test_feat)
em_res, em_meta = pickle.load(open(em_res_path, 'rb'))


def render(qpos, epos, hide_expert=False):
    env_vis.data.qpos[:env.model.nq] = qpos
    env_vis.data.qpos[env.model.nq:] = epos
    env_vis.data.qpos[env.model.nq] += 1.0
    if hide_expert:
        env_vis.data.qpos[env.model.nq + 1] += 100.0
    env_vis.sim_forward()
    env_vis.render()


def eval_expert(test_ind, start_ind, test_len):

    take = takes[test_ind]
    traj_pred = []
    cnn_feat = tensor(cnn_feat_dict[take][start_ind - cfg.fr_margin: start_ind + test_len])
    policy_vs_net.initialize(cnn_feat)
    value_vs_net.initialize(cnn_feat)

    # use prediction from ego pose estimation
    em_offset = em_cfg.fr_margin
    assert start_ind >= cfg.fr_margin + em_offset
    state_pred = em_res['traj_pred'][take][start_ind - cfg.fr_margin - em_offset: start_ind + test_len - em_offset]
    vel_pred = em_res['vel_pred'][take][start_ind - cfg.fr_margin - em_offset: start_ind + test_len - em_offset]
    ind = cfg.fr_margin
    qpos = state_pred[ind].copy()
    qvel = vel_pred[ind].copy()
    env.set_state(qpos, qvel)
    state = env.get_obs()

    if running_state is not None:
        state = running_state(state, update=False)

    for t in range(-cfg.fr_margin, 0):
        qpos = state_pred[t + cfg.fr_margin]
        traj_pred.append(qpos.copy())
        if args.render:
            render(qpos, qpos, False)
    if args.render:
        env_vis.viewer._paused = True

    fail = False
    for t in range(test_len):
        qpos = env.data.qpos
        traj_pred.append(qpos.copy())

        if args.render:
            epos = state_pred[min(t + cfg.fr_margin, state_pred.shape[0])]
            render(qpos, epos)

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

        if info['fail']:
            fail = True

        state = next_state

    if fail:
        logger.info('fail - test_ind: %d, start_ind %d' % (test_ind, start_ind))

    return np.vstack(traj_pred)


if args.mode == 'save':
    test_len = cfg.env_episode_len
    traj_pred_dict = {}
    traj_orig_dict = {}
    for i, take in enumerate(takes):
        logger.info('Testing on expert trajectory %s' % take)
        take_len = cnn_feat_dict[take].shape[0]
        traj_pred_arr = []
        start_ind = cfg.fr_margin + em_cfg.fr_margin
        while start_ind + test_len <= take_len:
            # logger.info('%d %d %d' % (start_ind, start_ind + test_len, take_len))
            traj_pred = eval_expert(i, start_ind, test_len)
            traj_pred_arr.append(traj_pred)
            start_ind += cfg.fr_margin
        traj_pred_dict[take] = np.stack(traj_pred_arr, axis=0)
        logger.info('%s' % str(traj_pred_dict[take].shape))
    results = {'traj_pred': traj_pred_dict}
    meta = {'algo': 'ego_forecast'}
    res_path = '%s/iter_%04d_%s.p' % (cfg.result_dir, args.iter, args.test_feat)
    pickle.dump((results, meta), open(res_path, 'wb'))
    logger.info('saved results to %s' % res_path)

elif args.mode == 'vis':
    args.render = True

    def key_callback(key, action, mods):
        global T, fr, paused, start, reverse, test_ind

        if action != glfw.RELEASE:
            return False
        elif not start:
            if key == glfw.KEY_D:
                T *= 1.5
            elif key == glfw.KEY_F:
                T = max(1, T / 1.5)
            elif key == glfw.KEY_R:
                start = True
            elif key == glfw.KEY_Q:
                fr = cfg.fr_margin
                test_ind = (test_ind - 1) % len(env.expert_list)
            elif key == glfw.KEY_E:
                fr = cfg.fr_margin
                test_ind = (test_ind + 1) % len(env.expert_list)
            elif key == glfw.KEY_W:
                fr = cfg.fr_margin + em_cfg.fr_margin
            elif key == glfw.KEY_S:
                reverse = not reverse
            elif key == glfw.KEY_RIGHT:
                if fr < cnn_feat_dict[takes[test_ind]].shape[0] - em_cfg.fr_margin - 1 - cfg.env_episode_len:
                    fr += 1
            elif key == glfw.KEY_LEFT:
                if fr > cfg.fr_margin + em_cfg.fr_margin:
                    fr -= 1
            elif key == glfw.KEY_UP:
                if fr < cnn_feat_dict[takes[test_ind]].shape[0] - em_cfg.fr_margin - 5 - cfg.env_episode_len:
                    fr += 5
            elif key == glfw.KEY_DOWN:
                if fr >= cfg.fr_margin + em_cfg.fr_margin + 5:
                    fr -= 5
            elif key == glfw.KEY_SPACE:
                paused = not paused
            else:
                return False
            return True

        return False

    T = 1
    start = False
    paused = True
    reverse = False
    env_vis.set_custom_key_callback(key_callback)
    test_ind = args.test_ind
    fr = cfg.fr_margin + em_cfg.fr_margin if args.start_ind is None else args.start_ind
    qpos = em_res['traj_pred'][takes[test_ind]][fr - em_cfg.fr_margin]
    render(qpos, qpos, True)

    while True:
        t = 0
        paused = True
        while not start:
            if t >= math.floor(T):
                if not reverse and fr < cnn_feat_dict[takes[test_ind]].shape[0] - 1 - cfg.env_episode_len:
                    fr += 1
                elif reverse and fr > cfg.fr_margin:
                    fr -= 1
                t = 0

            print(fr - em_cfg.fr_margin)
            qpos = em_res['traj_pred'][takes[test_ind]][fr - em_cfg.fr_margin]
            render(qpos, qpos, True)
            if not paused:
                t += 1

        eval_expert(test_ind, start_ind=fr, test_len=cfg.env_episode_len)
        fr += 30
        start = False

