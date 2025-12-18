import dgl
import torch
import pickle
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ALL_Dataset(Dataset):
    def __init__(self, *args):
        if (args[0] == "file"):
            filepath = args[1]
            f = open(filepath, 'rb')
            self.G_list = pickle.load(f)
            self.len = len(self.G_list)
        elif (args[0] == 'list'):
            self.G_list = args[1]
            self.len = len(args[1])

    def __getitem__(self, index):
        G = self.G_list[index]
        return G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]

    def __len__(self):
        return self.len

    def k_fold(self, train_idx, val_idx):
        train_list = [self.G_list[i] for i in train_idx]
        val_list = [self.G_list[i] for i in val_idx]
        return train_list, val_list

    def merge(self, data):
        self.G_list += data
        return self.G_list

    def len(self):
        return self.len

    def get(self, idx):
        data = self.G_list[idx]
        return data


def get_batch_pocket(pocket_dict):
    pocket_dict_new = {'ligand_fp_feature': [],
                       'pocket_seq_feature': [], 
                       'pocket_feature': [],
                       'pocket_pos': [],}
    res_batch = []
    i = 0
    for pocket in pocket_dict:
        for k in list(pocket_dict_new.keys()):
            if k == 'pocket_feature':
                res_batch += [i] * len(pocket[k])
            elif k== 'ligand_fp_feature':
                pocket_dict_new[k].append(torch.tensor(pocket[k]))
                continue
            pocket_dict_new[k].append(pocket[k])
        i += 1
    
    for k in list(pocket_dict_new.keys()):
        pocket_dict_new[k] = torch.cat(pocket_dict_new[k], 0)
    
    pocket_dict_new['res_batch'] = torch.tensor(res_batch)
    return pocket_dict_new


def collate_MF_net(data_batch):
    ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, ligand_motif_graph, protein_res_dict, motif_inter_graph, score = map(list, zip(*data_batch))
    ligand_atom_graph = dgl.batch(ligand_atom_graph)
    protein_atom_graph = dgl.batch(protein_atom_graph)
    atom_complex_graph = dgl.batch(atom_complex_graph)
    atom_inter_graph = dgl.batch(atom_inter_graph)
    ligand_motif_graph = dgl.batch(ligand_motif_graph)
    motif_inter_graph = dgl.batch(motif_inter_graph)
    labels = torch.tensor(score)
    protein_res_dict = get_batch_pocket(protein_res_dict)
    return ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, ligand_motif_graph, protein_res_dict, motif_inter_graph, labels


