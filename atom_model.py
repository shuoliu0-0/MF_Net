import dgl
import torch
import torch.nn as nn
from torch.nn import Module

from schnet import SchNetEncoder
from transformer import CFTransformerEncoder


def compose_atom_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    num_graphs = len(batch_ligand)

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    num_ligand_node=0
    num_protein_node=0
    for i in range(num_graphs):
        batch_p = torch.tensor([i]*int(batch_protein[i])).to(batch_protein.device)
        batch_l = torch.tensor([i]*int(batch_ligand[i])).to(batch_ligand.device)

        batch_ctx += [batch_l, batch_p]
        h_ctx += [h_ligand[num_ligand_node:num_ligand_node+batch_ligand[i]], 
                  h_protein[num_protein_node:num_protein_node+batch_protein[i]]]
        pos_ctx += [pos_ligand[num_ligand_node:num_ligand_node+batch_ligand[i]], 
                    pos_protein[num_protein_node:num_protein_node+batch_protein[i]]]
        mask_protein += [
            torch.zeros([batch_l.size(0)], device=h_protein.device, dtype=torch.bool),
            torch.ones([batch_p.size(0)], device=h_protein.device, dtype=torch.bool),
        ]
        num_ligand_node+=batch_ligand[i]
        num_protein_node+=batch_protein[i]

    batch_ctx = torch.cat(batch_ctx, dim=0)
    h_ctx = torch.cat(h_ctx, dim=0)
    pos_ctx = torch.cat(pos_ctx, dim=0)
    mask_protein = torch.cat(mask_protein, dim=0)

    return h_ctx, pos_ctx, batch_ctx, mask_protein


def compose_motif_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    num_graphs = batch_protein.max().item() + 1

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    num_ligand_node=0
    for i in range(num_graphs):
        mask_p = (batch_protein == i)
        batch_p = batch_protein[mask_p]
        batch_l = torch.tensor([i]*int(batch_ligand[i])).to(batch_p.device)

        batch_ctx += [batch_l, batch_p]
        h_ctx += [h_ligand[num_ligand_node:num_ligand_node+batch_ligand[i]], h_protein[mask_p]]
        pos_ctx += [pos_ligand[num_ligand_node:num_ligand_node+batch_ligand[i]], pos_protein[mask_p]]
        mask_protein += [
            torch.zeros([batch_l.size(0)], device=h_protein.device, dtype=torch.bool),
            torch.ones([batch_p.size(0)], device=h_protein.device, dtype=torch.bool),
        ]
        num_ligand_node+=batch_ligand[i]

    batch_ctx = torch.cat(batch_ctx, dim=0)
    h_ctx = torch.cat(h_ctx, dim=0)
    pos_ctx = torch.cat(pos_ctx, dim=0)
    mask_protein = torch.cat(mask_protein, dim=0)

    return h_ctx, pos_ctx, batch_ctx, mask_protein


def get_encoder(config, name='schnet'):
    if name == 'schnet':
        return SchNetEncoder(
            hidden_channels = config.hidden_channels,
            num_filters = config.num_filters,
            num_interactions = config.num_interactions,
            edge_channels = config.edge_channels,
            cutoff = config.cutoff,
        )
    elif name == 'tf':
        return CFTransformerEncoder(
            hidden_channels = config.hidden_channels,
            edge_channels = config.edge_channels,
            key_channels = config.key_channels,
            num_heads = config.num_heads,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)


class DTIConvGraph(Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 )

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class EdgeWeightAndSum_V2(Module):
    """
    change the nn.Tanh() function to nn.Sigmoid()
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum_V2, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid(),
            # nn.ReLU(),
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            weights = g.edata['w']
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum, weights