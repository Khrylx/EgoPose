import yaml
import os
import numpy as np
from utils import recreate_dirs


class Config:

    def __init__(self, cfg_id=None, create_dirs=False, cfg_dict=None):
        self.id = cfg_id
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_name = 'config/egomimic/%s.yml' % cfg_id
            if not os.path.exists(cfg_name):
                print("Config file doesn't exist: %s" % cfg_name)
                exit(0)
            cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'results'
        self.cfg_dir = '%s/egomimic/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.tb_dir)

        # data
        self.meta_id = cfg['meta_id']
        self.data_dir = 'datasets'
        self.meta = yaml.safe_load(open('%s/meta/%s.yml' % (self.data_dir, self.meta_id), 'r'))
        self.takes = {x: self.meta[x] for x in ['train', 'test']}
        self.expert_feat_file = '%s/features/expert_%s.p' % (self.data_dir, cfg['expert_feat']) if 'expert_feat' in cfg else None
        self.cnn_feat_file = '%s/features/cnn_feat_%s.p' % (self.data_dir, cfg['cnn_feat']) if 'cnn_feat' in cfg else None
        self.fr_margin = cfg.get('fr_margin', 10)

        # state net
        self.state_net_cfg = cfg.get('state_net_cfg', None)
        self.state_net_iter = cfg.get('state_net_iter', None)
        if self.state_net_cfg is not None:
            self.state_net_model = '%s/statereg/%s/models/iter_%04d_inf.p' % (self.base_dir, self.state_net_cfg, self.state_net_iter)

        # training config
        self.gamma = cfg.get('gamma', 0.95)
        self.tau = cfg.get('tau', 0.95)
        self.causal = cfg.get('causal', False)
        self.policy_htype = cfg.get('policy_htype', 'relu')
        self.policy_hsize = cfg.get('policy_hsize', [300, 200])
        self.policy_v_hdim = cfg.get('policy_v_hdim', 128)
        self.policy_v_net = cfg.get('policy_v_net', 'lstm')
        self.policy_v_net_param = cfg.get('policy_v_net_param', None)
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('policy_lr', 5e-5)
        self.policy_momentum = cfg.get('policy_momentum', 0.0)
        self.policy_weightdecay = cfg.get('policy_weightdecay', 0.0)
        self.value_htype = cfg.get('value_htype', 'relu')
        self.value_hsize = cfg.get('value_hsize', [300, 200])
        self.value_v_hdim = cfg.get('value_v_hdim', 128)
        self.value_v_net = cfg.get('value_v_net', 'lstm')
        self.value_v_net_param = cfg.get('value_v_net_param', None)
        self.value_optimizer = cfg.get('value_optimizer', 'Adam')
        self.value_lr = cfg.get('value_lr', 3e-4)
        self.value_momentum = cfg.get('value_momentum', 0.0)
        self.value_weightdecay = cfg.get('value_weightdecay', 0.0)
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.log_std = cfg.get('log_std', -2.3)
        self.fix_std = cfg.get('fix_std', False)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 10)
        self.min_batch_size = cfg.get('min_batch_size', 50000)
        self.max_iter_num = cfg.get('max_iter_num', 1000)
        self.seed = cfg.get('seed', 1)
        self.save_model_interval = cfg.get('save_model_interval', 100)
        self.reward_id = cfg.get('reward_id', 'quat')
        self.reward_weights = cfg.get('reward_weights', None)

        # adaptive parameters
        self.adp_iter_cp = np.array(cfg.get('adp_iter_cp', [0]))
        self.adp_noise_rate_cp = np.array(cfg.get('adp_noise_rate_cp', [1.0]))
        self.adp_noise_rate_cp = np.pad(self.adp_noise_rate_cp, (0, self.adp_iter_cp.size - self.adp_noise_rate_cp.size), 'edge')
        self.adp_log_std_cp = np.array(cfg.get('adp_log_std_cp', [self.log_std]))
        self.adp_log_std_cp = np.pad(self.adp_log_std_cp, (0, self.adp_iter_cp.size - self.adp_log_std_cp.size), 'edge')
        self.adp_policy_lr_cp = np.array(cfg.get('adp_policy_lr_cp', [self.policy_lr]))
        self.adp_policy_lr_cp = np.pad(self.adp_policy_lr_cp, (0, self.adp_iter_cp.size - self.adp_policy_lr_cp.size), 'edge')
        self.adp_noise_rate = None
        self.adp_log_std = None
        self.adp_policy_lr = None

        # env config
        self.mujoco_model_file = '%s/assets/mujoco_models/%s.xml' % (os.getcwd(), cfg['mujoco_model'])
        self.vis_model_file = '%s/assets/mujoco_models/%s.xml' % (os.getcwd(), cfg['vis_model'])
        self.env_start_first = cfg.get('env_start_first', False)
        self.env_init_noise = cfg.get('env_init_noise', 0.0)
        self.env_episode_len = cfg.get('env_episode_len', 200)
        self.obs_type = cfg.get('obs_type', 'full')
        self.obs_coord = cfg.get('obs_coord', 'heading')
        self.obs_heading = cfg.get('obs_heading', False)
        self.obs_vel = cfg.get('obs_vel', 'full')
        self.root_deheading = cfg.get('root_deheading', True)
        self.sync_exp_interval = cfg.get('sync_exp_interval', 100)
        self.action_type = cfg.get('action_type', 'position')

        # joint param
        if 'joint_params' in cfg:
            jparam = zip(*cfg['joint_params'])
            jparam = [np.array(p) for p in jparam]
            self.jkp, self.jkd, self.a_ref, self.a_scale, self.torque_lim = jparam[1:6]
            self.a_ref = np.deg2rad(self.a_ref)
            jkp_multiplier = cfg.get('jkp_multiplier', 1.0)
            jkd_multiplier = cfg.get('jkd_multiplier', jkp_multiplier)
            self.jkp *= jkp_multiplier
            self.jkd *= jkd_multiplier

        # body param
        if 'body_params' in cfg:
            bparam = zip(*cfg['body_params'])
            bparam = [np.array(p) for p in bparam]
            self.b_diffw = bparam[1]

    def update_adaptive_params(self, i_iter):
        cp = self.adp_iter_cp
        ind = np.where(i_iter >= cp)[0][-1]
        nind = ind + int(ind < len(cp) - 1)
        t = (i_iter - self.adp_iter_cp[ind]) / (cp[nind] - cp[ind]) if nind > ind else 0.0
        self.adp_noise_rate = self.adp_noise_rate_cp[ind] * (1-t) + self.adp_noise_rate_cp[nind] * t
        self.adp_log_std = self.adp_log_std_cp[ind] * (1-t) + self.adp_log_std_cp[nind] * t
        self.adp_policy_lr = self.adp_policy_lr_cp[ind] * (1-t) + self.adp_policy_lr_cp[nind] * t
