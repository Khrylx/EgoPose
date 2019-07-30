import argparse
import os
import sys
import pickle
import time
import torch
import numpy as np
sys.path.append(os.getcwd())

from utils import *
from models.video_reg_net import *
from ego_pose.utils.statereg_dataset import Dataset
from ego_pose.utils.statereg_config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--mode', default='train')
parser.add_argument('--data', default=None)
parser.add_argument('--test-feat', default=None)
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
args = parser.parse_args()
if args.data is None:
    args.data = args.mode if args.mode in {'train', 'test'} else 'train'

cfg = Config(args.cfg, create_dirs=(args.iter == 0))

"""setup"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
tb_logger = Logger(cfg.tb_dir)
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

dataset = Dataset(cfg.meta_id, args.data, cfg.fr_num, cfg.iter_method, cfg.shuffle, 2*cfg.fr_margin, cfg.num_sample)

"""networks"""
# if only predicting pose, then the out dim will be n_pose + 6 (include root velocities)
state_dim = (dataset.traj_dim - 1) // 2 + 6 if cfg.pose_only else dataset.traj_dim
no_cnn = (args.mode == 'save_inf' or args.test_feat is not None)
state_net = VideoRegNet(state_dim, cfg.v_hdim, cfg.cnn_fdim, no_cnn=no_cnn, cnn_type=cfg.cnn_type,
                        mlp_dim=cfg.mlp_dim, v_net_type=cfg.v_net, v_net_param=cfg.v_net_param, causal=cfg.causal)
if args.iter > 0:
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp, meta = pickle.load(open(cp_path, "rb"))
    if args.data != 'train':
        dataset.set_mean_std(meta['mean'], meta['std'])
    state_net.load_state_dict(model_cp['state_net_dict'], strict=not no_cnn)
if args.mode != 'save_inf':
    state_net.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, state_net.parameters()), lr=cfg.lr)
fr_margin = cfg.fr_margin

if args.mode == 'train':
    to_train(state_net)
    for i_epoch in range(args.iter, cfg.num_epoch):
        t0 = time.time()
        epoch_num_sample = 0
        epoch_loss = 0
        for of_np, traj_np, _ in dataset:
            num = traj_np.shape[0] - 2 * fr_margin
            of = tensor(of_np, dtype=dtype, device=device)
            of = torch.cat((of, zeros(of.shape[:-1] + (1,), device=device)), dim=-1).permute(0, 3, 1, 2).unsqueeze(1).contiguous()
            state_gt = tensor(traj_np[fr_margin: -fr_margin, :state_dim], dtype=dtype, device=device)
            state_pred = state_net(of)[fr_margin: -fr_margin, :]
            # compute loss
            loss = (state_gt - state_pred).pow(2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            epoch_loss += loss.cpu().item() * num
            epoch_num_sample += num

            """clean up gpu memory"""
            del of
            torch.cuda.empty_cache()

        epoch_loss /= epoch_num_sample
        logger.info('epoch {:4d}    time {:.2f}     nsample {}   loss {:.4f} '
                    .format(i_epoch, time.time() - t0, epoch_num_sample, epoch_loss))
        tb_logger.scalar_summary('loss', epoch_loss, i_epoch)

        with to_cpu(state_net):
            if cfg.save_model_interval > 0 and (i_epoch + 1) % cfg.save_model_interval == 0:
                cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_epoch + 1)
                model_cp = {'state_net_dict': state_net.state_dict()}
                meta = {'mean': dataset.mean, 'std': dataset.std}
                pickle.dump((model_cp, meta), open(cp_path, 'wb'))

elif args.mode == 'test':

    def get_traj_from_state_pred(state_pred, init_pos, init_heading, dt):
        nv = (dataset.traj_dim + 1) // 2
        nq = nv + 1
        pos = init_pos.copy()
        heading = init_heading.copy()
        traj_pred = []
        for i in range(state_pred.shape[0]):
            qpos = np.concatenate((pos, state_pred[i, :nq-2]))
            qvel = state_pred[i, nq-2:]
            qpos[3:7] = quaternion_multiply(heading, qpos[3:7])
            linv = quat_mul_vec(heading, qvel[:3])
            angv = quat_mul_vec(qpos[3:7], qvel[3:6])
            pos += linv[:2] * dt
            new_q = quaternion_multiply(quat_from_expmap(angv * dt), qpos[3:7])
            heading = get_heading_q(new_q)
            traj_pred.append(qpos)
        traj_qpos = np.vstack(traj_pred)
        return traj_qpos

    to_test(state_net)
    dataset.iter_method = 'iter'
    dataset.shuffle = False
    torch.set_grad_enabled(False)
    epoch_num_sample = 0
    epoch_loss = 0
    state_pred_arr = []
    traj_orig_arr = []
    res_pred = {}
    res_orig = {}
    meta = {}
    if args.test_feat is None:
        take = dataset.takes[0]
        for of_np, traj_np, traj_orig_np in dataset:
            num = traj_np.shape[0] - 2 * fr_margin
            of = tensor(of_np, dtype=dtype, device=device)
            of = torch.cat((of, zeros(of.shape[:-1] + (1,), device=device)), dim=-1).permute(0, 3, 1, 2).unsqueeze(1).contiguous()
            state_gt = tensor(traj_np[fr_margin: -fr_margin, :state_dim], dtype=dtype, device=device)
            state_pred = state_net(of)[fr_margin: -fr_margin, :]
            # logging
            loss = (state_gt - state_pred).pow(2).sum(dim=1).mean()
            state_pred = state_pred.cpu().numpy() * dataset.std[None, :state_dim] + dataset.mean[None, :state_dim]
            traj_orig = traj_orig_np[fr_margin: -fr_margin, :]
            state_pred_arr.append(state_pred)
            traj_orig_arr.append(traj_orig)
            epoch_loss += loss.cpu().item() * num
            epoch_num_sample += num
            # save if datset moved on to next take
            if dataset.cur_ind >= len(dataset.takes) or dataset.takes[dataset.cur_tid] != take:
                state_pred = np.vstack(state_pred_arr)
                traj_orig = np.vstack(traj_orig_arr)
                init_pos = traj_orig[0, :2]
                init_heading = get_heading_q(traj_orig[0, 3:7])
                traj_pred = get_traj_from_state_pred(state_pred, init_pos, init_heading, dataset.dt)
                res_pred[take] = traj_pred
                res_orig[take] = np.vstack(traj_orig_arr)
                state_pred_arr, traj_orig_arr = [], []
                take = dataset.takes[dataset.cur_tid]
        epoch_loss /= epoch_num_sample
        results = {'traj_pred': res_pred, 'traj_orig': res_orig}
        meta['algo'] = 'state_reg'
        meta['num_sample'] = epoch_num_sample
        meta['epoch_loss'] = epoch_loss
        res_path = '%s/iter_%04d_%s.p' % (cfg.result_dir, args.iter, args.data)
    else:
        cnn_feat_file = '%s/features/cnn_feat_%s.p' % (dataset.base_folder, args.test_feat)
        cnn_feat_dict, _ = pickle.load(open(cnn_feat_file, 'rb'))
        for take, cnn_feat in cnn_feat_dict.items():
            cnn_feat = tensor(cnn_feat, dtype=dtype, device=device)
            state_pred = state_net(cnn_feat.unsqueeze(1)).squeeze(1)[cfg.fr_margin: -cfg.fr_margin, :].cpu().numpy()
            state_pred = state_pred * dataset.std[None, :state_dim] + dataset.mean[None, :state_dim]
            traj_pred = get_traj_from_state_pred(state_pred, np.zeros(2), np.array([1, 0, 0, 0]), dataset.dt)
            res_pred[take] = traj_pred
            epoch_num_sample += state_pred.shape[0]
        results = {'traj_pred': res_pred}
        meta['algo'] = 'state_reg'
        meta['num_sample'] = epoch_num_sample
        res_path = '%s/iter_%04d_%s.p' % (cfg.result_dir, args.iter, args.test_feat)
    pickle.dump((results, meta), open(res_path, 'wb'))
    logger.info('nsample {}   loss {:.4f} '.format(epoch_num_sample, epoch_loss))
    logger.info('saved results to %s' % res_path)

elif args.mode == 'save_inf':
    cp_path = '%s/iter_%04d_inf.p' % (cfg.model_dir, args.iter)
    model_cp = {'state_net_dict': state_net.state_dict()}
    meta = {'mean': dataset.mean, 'std': dataset.std, 'cfg': cfg}
    pickle.dump((model_cp, meta), open(cp_path, 'wb'))
