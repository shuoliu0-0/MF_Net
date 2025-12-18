import os
import copy
import yaml
import torch
import random
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


def run_a_val(val_loader, models, i, loss_fn, device):
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


if __name__ == '__main__':
    """ Please use the ./data/process.py file to preprocess the raw data and set up the training, validation, and test sets """

    with open("./configs/config_pdbbind.yml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    gpuid = 2
    batch_size = 16
    # seed initialize
    np.random.seed(8)   #8
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)
    torch.cuda.manual_seed_all(8)
    random.seed(8)
    os.environ['PYTHONHASHSEED'] = str(8)

    print("loading data")
    test_set = ALL_Dataset('file', './pdbbind_feature/test_data.pkl')

    test_loader = DataLoaderX(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_MF_net)
    
    loss_fn = nn.MSELoss()
    device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
    
    models = [MFNet(config, device).to(device),
              MoE(input_size=config.inter_graph.outdim, num_experts=3, noisy_gating=True, k=2, 
                config=config, device=device).to(device)]
    
    num_models = len(models)
    for m_idx in range(0, num_models):
        m = models[m_idx]
        print(m_idx)
        pre_model_path = config.pretrain_models0
        if m_idx == 0 and os.path.exists(pre_model_path):
            m.load_state_dict(torch.load(pre_model_path))
            models[m_idx] = copy.deepcopy(m)
            continue

        pre_model_path = config.pretrain_models1
        if m_idx == 1 and os.path.exists(pre_model_path):
            m.load_state_dict(torch.load(pre_model_path))
            models[m_idx] = copy.deepcopy(m)
            test_metric, test_ensemble = run_a_val(test_loader, models, m_idx, loss_fn, device)
            print(f'v2016test: mae: {test_ensemble[0]} rmse: {test_ensemble[1]} pr: {test_ensemble[2]} sd: {test_ensemble[3]} r2: {test_ensemble[4]}\n', flush=True)
            continue