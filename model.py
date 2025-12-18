import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module, Linear
from torch.nn import Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from torch.distributions.normal import Normal
from torch_geometric.nn import GINConv, global_add_pool

from MGraphDTA import get_batch, TargetRepresentation, GraphDenseNet, Conv1dReLU
from atom_model import compose_atom_context, compose_motif_context, get_encoder, DTIConvGraph, EdgeWeightAndSum_V2


class SeqEmbedding(Module):
    def __init__(self,
                 config, device=None):
        super(SeqEmbedding, self).__init__()
        
        outdim = int(config.inter_graph.outdim / 2)
        self.protein_encoder = TargetRepresentation(config.pocket.block_num, config.pocket.embedding_size,
                                                    out_dim=outdim)
        self.ligand_encoder = GraphDenseNet(num_input_features=config.ligand.node_feat_size,
                                            out_dim=outdim, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])
        
        self.mlp_fp = nn.Sequential(
            nn.Linear(config.ligand.fp_feat_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, outdim)
        )
        
        self.mlp = MLP([outdim*3, outdim*2, config.inter_graph.outdim], dropout=0.1)

    def forward(self, ligand_atom_graph, seq_feats, fp_feat=None):
        protein_x = self.protein_encoder(seq_feats.view(-1, 125, 33).float())
        ligand_x = self.ligand_encoder(ligand_atom_graph)
        
        if fp_feat is None:
            feats = torch.cat([protein_x, ligand_x], dim=-1)
        else:
            fp_x = self.mlp_fp(fp_feat.float())
            feats = torch.cat([protein_x, ligand_x, fp_x], dim=-1)
            feats = self.mlp(feats)
        return feats


class AtomEmbedding(Module):
    def __init__(self,
                 config,
                 degree_dict=None):
        super(AtomEmbedding, self).__init__()
        node_feat_size = config.ligand.node_feat_size
        hidden_channels = config.encoder.hidden_channels

        self.ligand_emb = Linear(node_feat_size, hidden_channels)
        self.pocket_emb = Linear(config.pocket.atom_feature_dim, hidden_channels)
        
        self.complex_emb = get_encoder(config.encoder)

    def forward(self, ligand_graph, pocket_graph):
        ligand_feats = ligand_graph.ndata.pop('h')
        ligand_pos = ligand_graph.ndata['pos']
        pocket_feats = pocket_graph.ndata.pop('h')
        pocket_pos = pocket_graph.ndata['pos']

        ligand_feats = self.ligand_emb(ligand_feats.float())
        pocket_feats = self.pocket_emb(pocket_feats.float())

        feats, pos_ctx, batch_ctx, mask_protein = compose_atom_context(h_protein=pocket_feats,
                                                                        h_ligand=ligand_feats,
                                                                        pos_protein=pocket_pos,
                                                                        pos_ligand=ligand_pos,
                                                                        batch_protein=pocket_graph.batch_num_nodes(),
                                                                        batch_ligand=ligand_graph.batch_num_nodes())
        feats = self.complex_emb(node_attr=feats.float(), pos=pos_ctx.float(), batch=batch_ctx)
        
        return feats


class DTIGraph(Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIGraph, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)
        # read out
        self.readout = EdgeWeightAndSum_V2(out_dim)

    def forward(self, atom_feats, atom_inter_graph):
        bond_feats = atom_inter_graph.edata['e']
        new_feats = self.grah_conv(atom_inter_graph, atom_feats, bond_feats)
        new_feats = self.dropout(new_feats)
        new_feats = self.bn_layer(new_feats)
        new_feats, weights = self.readout(atom_inter_graph, new_feats)
        return new_feats


class MotifEmbedding(Module):
    def __init__(self,
                 config,
                 degree_dict=None):
        super(MotifEmbedding, self).__init__()
        node_feat_size = config.ligand.motif_feat_size
        hidden_channels = config.encoder.hidden_channels

        self.ligand_motif_emb = Linear(node_feat_size, hidden_channels)
        self.pocket_res_emb = Linear(config.pocket.res_feature_dim, hidden_channels)
        
        self.res_emb = get_encoder(config.encoder, name='tf')

    def forward(self, ligand_motif_graph, pocket_res_graph):
        ligand_motif_feats = ligand_motif_graph.ndata.pop('h')
        ligand_motif_pos = ligand_motif_graph.ndata['pos']

        ligand_motif_feats = self.ligand_motif_emb(ligand_motif_feats.float())
        pocket_res_feats = self.pocket_res_emb(pocket_res_graph['pocket_feature'].float())

        motif_feats, pos_ctx, batch_ctx, mask_protein = compose_motif_context(h_protein=pocket_res_feats,
                                                                               h_ligand=ligand_motif_feats,
                                                                               pos_protein=pocket_res_graph['pocket_pos'],
                                                                               pos_ligand=ligand_motif_pos,
                                                                               batch_protein=pocket_res_graph['res_batch'],
                                                                               batch_ligand=ligand_motif_graph.batch_num_nodes())
        motif_feats = self.res_emb(node_attr=motif_feats.float(), pos=pos_ctx.float(), batch=batch_ctx)
        
        return motif_feats


class GINNet(torch.nn.Module):
    def __init__(self, x_dim, e_dim, dim):
        super(GINNet, self).__init__()
        self.dim = dim

        x_nn1 = Sequential(Linear(x_dim, dim), ReLU(), Linear(dim, dim))
        self.x_conv1 = GINConv(x_nn1)
        self.x_bn1 = torch.nn.BatchNorm1d(dim)
        x_nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.x_conv2 = GINConv(x_nn2)
        self.x_bn2 = torch.nn.BatchNorm1d(dim)

        self.e_linear = Sequential(Linear(e_dim, 22),
                                 ReLU(),
                                 Linear(22, 40),
                                 ReLU(),
                                 Linear(40, dim))

        self.linear = Sequential(Linear(dim*3, 512),
                                 ReLU(),
                                 Linear(512, 512),
                                 ReLU(),
                                 Linear(512, 256),
                                 ReLU(),
                                 Linear(256, dim))

    def forward(self, x, data):
        edge_index = torch.stack(data.edges(), dim=0)
        edge_attr = data.edata.pop('e')
        edge_batch = data.batch_num_edges()
        edge_batch = get_batch(edge_batch)

        x = F.relu(self.x_conv1(x, edge_index))
        x = self.x_bn1(x)
        x = F.relu(self.x_conv2(x, edge_index))
        x = self.x_bn2(x)

        e = self.e_linear(edge_attr)
        x = x[edge_index].transpose(0, 1).reshape(-1, self.dim * 2)
        xe = torch.cat((x, e), 1)
        xe = self.linear(xe)
        out = global_add_pool(xe, edge_batch)

        return out


class MFNet(Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.seq_emb = SeqEmbedding(config, device)

        self.atom_emb = AtomEmbedding(config)
        self.atom_inter_gnn = DTIGraph(config.encoder.hidden_channels + 10, config.inter_graph.outdim,
                                        config.inter_graph.dropout)
        
        self.motif_emb = MotifEmbedding(config)
        # self.motif_inter_gnn = DTIGraph(config.encoder.hidden_channels + 1, config.inter_graph.outdim,
        #                                 config.inter_graph.dropout)
        self.motif_inter_gnn = GINNet(x_dim=config.encoder.hidden_channels, e_dim=1, dim=config.inter_graph.outdim)
        
        # Conv1dReLU
        self.feature_fusion = Conv1dReLU(config.inter_graph.outdim, config.inter_graph.outdim, kernel_size=3, padding=0)

        self.out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.seq_out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.atom_out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.motif_out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    
    def forward(self, data, m_idx = 0):
        ligand_atom_graph = data[0]
        protein_atom_graph = data[1]
        atom_inter_graph = data[3]
        ligand_motif_graph = data[4]
        protein_res_dict = data[5]
        motif_inter_graph = data[6]

        seq_feat = self.seq_emb(ligand_atom_graph, protein_res_dict['pocket_seq_feature'], 
                                protein_res_dict['ligand_fp_feature'])
        
        atom_feat = self.atom_emb(ligand_atom_graph, protein_atom_graph)
        atom_feat = self.atom_inter_gnn(atom_feat, atom_inter_graph)

        motif_feat = self.motif_emb(ligand_motif_graph, protein_res_dict)
        motif_feat = self.motif_inter_gnn(motif_feat, motif_inter_graph)

        feats = torch.stack([seq_feat, atom_feat, motif_feat], dim=1)
        feats = feats.permute(0, 2, 1)
        feats = self.feature_fusion(feats)  # Conv1dReLU
        y_hat = self.out(feats.squeeze())

        if m_idx == 0:
            seq_feat = self.seq_out(seq_feat)
            atom_feat = self.atom_out(atom_feat)
            motif_feat = self.motif_out(motif_feat)
            return torch.squeeze(y_hat), seq_feat, atom_feat, motif_feat, feats.squeeze()
        else:
            return torch.squeeze(y_hat), feats.squeeze()


class MOE_MFNet(Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.seq_emb = SeqEmbedding(config, device)

        self.atom_emb = AtomEmbedding(config)
        self.atom_inter_gnn = DTIGraph(config.encoder.hidden_channels + 10, config.inter_graph.outdim,
                                        config.inter_graph.dropout)
        
        self.motif_emb = MotifEmbedding(config)
        # self.motif_inter_gnn = DTIGraph(config.encoder.hidden_channels + 1, config.inter_graph.outdim,
        #                                 config.inter_graph.dropout)
        self.motif_inter_gnn = GINNet(x_dim=config.encoder.hidden_channels, e_dim=1, dim=config.inter_graph.outdim)
        
        # Conv1dReLU
        self.feature_fusion = Conv1dReLU(config.inter_graph.outdim, config.inter_graph.outdim, kernel_size=3, padding=0)

        self.out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.seq_out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.atom_out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.motif_out = nn.Sequential(
            nn.Linear(config.inter_graph.outdim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    
    def forward(self, data, m_idx = 0):
        ligand_atom_graph = data[0]
        protein_atom_graph = data[1]
        atom_inter_graph = data[3]
        ligand_motif_graph = data[4]
        protein_res_dict = data[5]
        motif_inter_graph = data[6]

        seq_feat = self.seq_emb(ligand_atom_graph, protein_res_dict['pocket_seq_feature'], 
                                protein_res_dict['ligand_fp_feature'])
        
        atom_feat = self.atom_emb(ligand_atom_graph, protein_atom_graph)
        atom_feat = self.atom_inter_gnn(atom_feat, atom_inter_graph)

        motif_feat = self.motif_emb(ligand_motif_graph, protein_res_dict)
        motif_feat = self.motif_inter_gnn(motif_feat, motif_inter_graph)

        feats = torch.stack([seq_feat, atom_feat, motif_feat], dim=1)
        feats = feats.permute(0, 2, 1)
        feats = self.feature_fusion(feats)  # Conv1dReLU
        y_hat = self.out(feats.squeeze())

        if m_idx == 0:
            seq_feat = self.seq_out(seq_feat)
            atom_feat = self.atom_out(atom_feat)
            motif_feat = self.motif_out(motif_feat)
            return torch.squeeze(y_hat), seq_feat, atom_feat, motif_feat, feats.squeeze()
        else:
            return torch.squeeze(y_hat), feats.squeeze()


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        if isinstance(inp, list):
            inps, result = [], []
            for index in self._batch_index:
                inps.append(inp[index])
            i = 0
            for index in self._part_sizes:
                result.append(inps[i:i + index])
                i += index
            return result
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index]
        # inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, num_experts, noisy_gating, k,
                 config, device, pretrain_models=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        # self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        model = MFNet(config, device).to(device)
        # model = MOE_MFNet(config, device).to(device)
        if pretrain_models != None:
            model.load_state_dict(torch.load(pretrain_models))
        self.experts = nn.ModuleList([model for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, data, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)

        expert_outputs = [self.experts[i](copy.deepcopy(data))[0] for i in range(self.num_experts)]
        out_experts = []
        for i, out in enumerate(expert_outputs):
            out_experts.append(dispatcher.dispatch(out)[i].unsqueeze(1))
        y = dispatcher.combine(out_experts)
        return y, loss

