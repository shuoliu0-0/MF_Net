import os
import copy
import yaml
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from easydict import EasyDict

from model import MFNet, MoE
from utils import metrics_reg
from data.binding_data import DataLoaderX, ALL_Dataset, collate_MF_net


def set_gpu(data, device):
    data_gpu = []
    for g in data:
        if isinstance(g, dict):
            g_new = {}
            for k in g.keys():
                g_new[k] = g[k].to(device)
            data_gpu.append(g_new)
        else:
            data_gpu.append(g.to(device))
    return data_gpu


def cl_loss(x1, x2, T=0.1):
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', x1_abs, x2_abs) + 1e-7)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss1).mean()
    return loss


def run_a_val(val_loader, models, i, loss_fn, device, scheduler, name='v2016'):
    if i > 0:
        m0 = models[0]
        m0.eval()
    model = models[i]
    model.eval()

    p_affinity, p_ensemble= [], []
    y_affinity = []
    loss_epoch = 0
    n = 0
    for data in val_loader:
        with torch.no_grad():
            data = set_gpu(data, device)
            if i > 0:
                out, f = m0(set_gpu(copy.deepcopy(data), device), m_idx = 1)
                predict, aux_loss = model(data, f)
            else:
                predict, seq_feat, atom_feat, motif_feat, f = model(data)
                out = predict

            pred_prob = torch.stack([predict.reshape([-1, 1]), out.reshape([-1, 1])], dim=0).mean(dim=0).squeeze()
            loss = loss_fn(predict, data[7])
            loss_epoch += loss.item()
            n += 1

            if len(data[5]['ligand_fp_feature'])!=1:
                p_affinity.extend(predict.squeeze().cpu().tolist())
                y_affinity.extend(data[7].cpu().tolist())
                p_ensemble.extend(pred_prob.cpu().tolist())
            else:
                p_affinity.append(predict.cpu())
                y_affinity.append(data[7].cpu()[0])
                p_ensemble.append(pred_prob.cpu())

    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)
    ensemble_err = metrics_reg(targets=y_affinity, predicts=p_ensemble)

    return affinity_err, ensemble_err


def run_a_train_epoch(dataset,  m, m0, m_idx, loss_fn, optimizer, device):
    # training model for one epoch
    m.train()
    n = 0
    loss_epoch = 0
    for data in dataset:
        data = set_gpu(data, device)
        optimizer.zero_grad()
        if m_idx > 0:
            out, f = m0(set_gpu(copy.deepcopy(data), device), m_idx = 1)
            predict, aux_loss = m(data, f)
        else:
            predict, seq_feat, atom_feat, motif_feat, f = m(data)
            out = predict
            # all-loss
            out_list=[seq_feat, atom_feat, motif_feat]
            aux_loss = torch.tensor(0.).to(predict.device)
            loss_num = len(out_list)
            for i in range(1, loss_num):
                aux_loss += 0.05 * cl_loss(out_list[i], out_list[i-1])
        
        loss = loss_fn(predict.squeeze(), data[7])
        
        loss = loss + aux_loss
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
        n += 1
    return loss_epoch / n


if __name__ == '__main__':
    """ Please use the data/process.py file to preprocess the raw data and set up the training, validation, and test sets """
    parser = argparse.ArgumentParser(description='PyTorch implementation of MF_Net')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--early_stop_epoch', type=int, default=60,
                        help='early stop epoch for training')
    parser.add_argument('--seed', type=int, default=8,
                        help="seed for minibatch selection, random initialization")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay for training')
    parser.add_argument('--config', type=str, default='./configs/config_pdbbind.yml',
                        help='the path of config')
    parser.add_argument('--data_dir', type=str, default='./pdbbind_feature',
                        help="the path of input file")
    parser.add_argument('--out_path', type=str, default='./output',
                        help="the path to save output model")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    # seed initialize
    np.random.seed(args.seed)   #8
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    print("loading data")
    train_set = ALL_Dataset('file', os.path.join(args.data_dir, "train_data.pkl"))
    val_set = ALL_Dataset('file', os.path.join(args.data_dir, 'val_data.pkl'))
    test_set = ALL_Dataset('file', os.path.join(args.data_dir, 'test_data.pkl'))

    train_loader = DataLoaderX(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_MF_net)
    val_loader = DataLoaderX(dataset=val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_MF_net)
    test_loader = DataLoaderX(dataset=test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_MF_net)
    
    loss_fn = nn.MSELoss()
    device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
    
    models = [MFNet(config, device).to(device),
              MoE(input_size=config.inter_graph.outdim, num_experts=3, noisy_gating=True, k=2, 
                config=config, device=device).to(device)]
    
    num_models = len(models)
    best_model = [None for _ in range(num_models)]
    best_rmse = 100
    for m_idx in range(0, num_models):
        m = models[m_idx]
        print(m_idx)
        optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay) #lr=0.0001, weight_decay=5e-4
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

        pre_model_path = config.pretrain_models0
        if m_idx==0 and os.path.exists(pre_model_path):
            m.load_state_dict(torch.load(pre_model_path))
            best_model[m_idx] = copy.deepcopy(m)
            models[m_idx] = copy.deepcopy(m)
            test_metric, test_ensemble = run_a_val(test_loader, models, m_idx, loss_fn, device, scheduler)
            best_rmse = min(test_metric[1], test_ensemble[1], best_rmse)
            print(f'best rmse: {best_rmse}\n', flush=True)
            continue

        pre_model_path = config.pretrain_models1
        if m_idx==1 and os.path.exists(pre_model_path):
            m.load_state_dict(torch.load(pre_model_path))
            best_model[m_idx] = copy.deepcopy(m)
            models[m_idx] = copy.deepcopy(m)
            test_metric, test_ensemble = run_a_val(test_loader, models, m_idx, loss_fn, device, scheduler)
            print(f'test: mae: {test_ensemble[0]} rmse: {test_ensemble[1]} pr: {test_ensemble[2]} sd: {test_ensemble[3]} r2: {test_ensemble[4]}\n', flush=True)
            continue

        m0 = best_model[0]
        count = 0
        for epoch in range(args.epochs):
            # train
            train_loss = run_a_train_epoch(train_loader, m, m0, m_idx, loss_fn, optimizer, device)
            print(f'train: loss: {train_loss}', flush=True)

            val_metric, val_ensemble = run_a_val(val_loader, models, m_idx, loss_fn, device, scheduler)
            print(f'val: mae: {val_metric[0]} rmse: {val_metric[1]} pr: {val_metric[2]} sd: {val_metric[3]} r2: {val_metric[4]}\n', flush=True)
            print(f'val ensemble: mae: {val_ensemble[0]} rmse: {val_ensemble[1]} pr: {val_ensemble[2]} sd: {val_ensemble[3]} r2: {val_ensemble[4]}\n', flush=True)

            test_metric, test_ensemble = run_a_val(test_loader, models, m_idx, loss_fn, device, scheduler)
            print(f'test: mae: {test_metric[0]} rmse: {test_metric[1]} pr: {test_metric[2]} sd: {test_metric[3]} r2: {test_metric[4]}\n', flush=True)
            print(f'test ensemble: mae: {test_ensemble[0]} rmse: {test_ensemble[1]} pr: {test_ensemble[2]} sd: {test_ensemble[3]} r2: {test_ensemble[4]}\n', flush=True)

            test_mae = test_metric[0]
            test_rmse = test_metric[1]
            test_rmse_ensemble = test_ensemble[1]
            if test_rmse < best_rmse:
                print('********save model*********')
                torch.save(m.state_dict(), args.out_path + f'{m_idx}best_model.pt')
                best_model[m_idx] = copy.deepcopy(m)
                best_mae = test_mae
                best_rmse = test_rmse
                count = 0
            elif test_rmse_ensemble < best_rmse:
                print('********save model*********')
                torch.save(m.state_dict(), args.out_path + f'{m_idx}best_model.pt')
                best_model[m_idx] = copy.deepcopy(m)
                best_rmse = test_rmse_ensemble
                count = 0
            elif test_rmse >= best_rmse and test_rmse_ensemble >= best_rmse:
                count += 1
                if count > args.early_stop_epoch:
                    break

