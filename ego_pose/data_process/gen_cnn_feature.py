import argparse
import os
import sys
import pickle
import math
import datetime
import numpy as np
sys.path.append(os.getcwd())

from utils import *
from models.video_reg_net import *
from ego_pose.utils.statereg_dataset import Dataset
from ego_pose.utils.statereg_config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--meta-id', default=None)
parser.add_argument('--out-id', default=None)
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--iter', type=int, default=100)
args = parser.parse_args()

cfg = Config(args.cfg, create_dirs=False)


"""setup"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
torch.set_grad_enabled(False)
logger = create_logger(os.path.join(cfg.log_dir, 'gen_cnn_feature.txt'))

cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
logger.info('loading model from checkpoint: %s' % cp_path)
model_cp, meta = pickle.load(open(cp_path, "rb"))

meta_id = cfg.meta_id if args.meta_id is None else args.meta_id
dataset = Dataset(meta_id, 'all', cfg.fr_num, 'iter', False, 0)
dataset.set_mean_std(meta['mean'], meta['std'])
state_net = VideoRegNet(dataset.mean.size, cfg.v_hdim, cfg.cnn_fdim, mlp_dim=cfg.mlp_dim, cnn_type=cfg.cnn_type,
                        v_net_type=cfg.v_net, v_net_param=cfg.v_net_param, causal=cfg.causal)
state_net.load_state_dict(model_cp['state_net_dict'])
state_net.to(device)
to_test(state_net)


num_sample = 0
take = dataset.takes[0]
cnn_features = {}
feature_arr = []
for of_np, _, _ in dataset:
    of = tensor(of_np, dtype=dtype, device=device)
    of = torch.cat((of, zeros(of.shape[:-1] + (1,), device=device)), dim=-1).permute(0, 3, 1, 2).unsqueeze(1).contiguous()
    feature = state_net.get_cnn_feature(of).cpu().numpy()
    feature_arr.append(feature)
    num_sample += feature.shape[0]
    # print(feature.shape, of_np.shape, next_take)
    if dataset.cur_ind >= len(dataset.takes) or dataset.takes[dataset.cur_tid] != take:
        cnn_features[take] = np.vstack(feature_arr)
        feature_arr = []
        take = dataset.takes[dataset.cur_tid]

logger.info('cfg: %s, iter: %d, total sample: %d, dataset length: %d' % (args.cfg, args.iter, num_sample, dataset.len))

meta = {'cfg': args.cfg, 'iter': args.iter, 'meta': meta_id, 'time': datetime.datetime.now()}
path = 'datasets/features/cnn_feat_%s.p' % args.out_id
pickle.dump((cnn_features, meta), open(path, 'wb'))
