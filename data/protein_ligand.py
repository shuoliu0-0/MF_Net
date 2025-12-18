import dgl
import torch
import numpy as np
from rdkit import Chem

from utils import one_of_k_encoding_unk, get_fp_feature, maccskeys_emb, pharm_property_types_feats
from get_motif import tree_decomp
from pocket_feature import PDBProtein, cal_seq_feats


# allowable node and edge features
ATOM_FEATURES = {
    'atomic_num': [6, 7, 8, 9, 15, 16, 17, 35, 53],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], }


def get_mol(lig_file, pocket_file):
    mol = Chem.MolFromMol2File(lig_file)
    # mol = next(iter(Chem.SDMolSupplier(filepath, removeHs=True)))
    if mol is None:
        mol = Chem.MolFromMol2File(lig_file.split('.')[0].strip('_opt')+'.mol2')
        if mol is None:
            suppl = Chem.SDMolSupplier(lig_file.split('.')[0].strip('_opt')+'.sdf')
            mols = [mol for mol in suppl if mol]
            if len(mols)>0:
                mol = mols[0]
    pocket = Chem.MolFromPDBFile(pocket_file, sanitize=True)
    if mol is None or pocket is None:
        return None, None
    return mol, pocket


def get_protein_graph(protein, protein_positions, pidx):
    pidx2tidx = {pidx[i]: i for i in range(len(pidx))}

    innerdis = protein_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]
    innerdis = np.sqrt((innerdis * innerdis).sum(-1))

    atom_features_list = []
    atom_pos_list = []
    for i in pidx:
        atom = protein.GetAtomWithIdx(int(i))
        atom_feature = []
        atom_feature += one_of_k_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
        atom_feature += one_of_k_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
        atom_feature += one_of_k_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
        atom_feature += one_of_k_encoding_unk(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag'])
        atom_feature += one_of_k_encoding_unk(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
        atom_feature += one_of_k_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
        atom_feature += [1 if atom.GetIsAromatic() else 0]
        atom_feature += [0]
        # atom_feature = list(get_atom_features(atom, is_protein=True)) + protein_positions[i].tolist()
        atom_features_list.append(atom_feature)
        atom_pos_list.append(protein_positions[i].tolist())
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    pos = torch.tensor(np.array(atom_pos_list), dtype=torch.float)

    edges_list = []
    edge_features_list = []
    for bond in protein.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
            continue
        if atom1.GetIdx() in pidx and atom2.GetIdx() in pidx:
            is_Aromatic = int(bond.GetIsAromatic())
            is_inring = int(bond.IsInRing())
            bt = bond.GetBondType()
            fbond = [bt == Chem.rdchem.BondType.SINGLE,
                    bt == Chem.rdchem.BondType.DOUBLE,
                    bt == Chem.rdchem.BondType.TRIPLE,
                    bt == Chem.rdchem.BondType.AROMATIC, 0]
            edge_features_list.append(fbond + [is_Aromatic, is_inring, 1, 0]+ [innerdis[atom1.GetIdx(), atom2.GetIdx()]/10])
            edge_features_list.append(fbond + [is_Aromatic, is_inring, 1, 0]+ [innerdis[atom1.GetIdx(), atom2.GetIdx()]/10])
            edges_list.append((pidx2tidx[atom1.GetIdx()], pidx2tidx[atom2.GetIdx()]))
            edges_list.append((pidx2tidx[atom2.GetIdx()], pidx2tidx[atom1.GetIdx()]))

    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)
    
    pocket_atom_graph = dgl.DGLGraph()
    pocket_atom_graph.add_nodes(len(pos))
    pocket_atom_graph.add_edges(edge_index[0], edge_index[1])
    pocket_atom_graph.ndata['pos'] = pos
    pocket_atom_graph.ndata['h'] = x
    pocket_atom_graph.edata['e'] = edge_attr
    return pocket_atom_graph


def get_ligand_graph(ligand, ligand_positions):
    ligand_atom_features_list = []
    ligand_atom_pos_list = []
    for i in range(len(ligand.GetAtoms())):
        atom = ligand.GetAtomWithIdx(int(i))
        atom_feature = []
        atom_feature += one_of_k_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
        atom_feature += one_of_k_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
        atom_feature += one_of_k_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
        atom_feature += one_of_k_encoding_unk(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag'])
        atom_feature += one_of_k_encoding_unk(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
        atom_feature += one_of_k_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
        atom_feature += [1 if atom.GetIsAromatic() else 0]
        atom_feature += [1]
        ligand_atom_features_list.append(atom_feature)
        ligand_atom_pos_list.append(ligand_positions[i].tolist())
    ligand_x = torch.tensor(np.array(ligand_atom_features_list), dtype=torch.float)
    ligand_pos = torch.tensor(np.array(ligand_atom_pos_list), dtype=torch.float)
    
    innerdis = ligand_positions[:, np.newaxis, :] - ligand_positions[np.newaxis, :, :]
    innerdis = np.sqrt((innerdis * innerdis).sum(-1))
    if len(ligand.GetBonds()) > 0:  # mol has bonds
        ligand_edges_list = []
        ligand_edge_features_list = []
        for bond in ligand.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            is_Aromatic = int(bond.GetIsAromatic())
            is_inring = int(bond.IsInRing())
            bt = bond.GetBondType()
            fbond = [bt == Chem.rdchem.BondType.SINGLE,
                    bt == Chem.rdchem.BondType.DOUBLE,
                    bt == Chem.rdchem.BondType.TRIPLE,
                    bt == Chem.rdchem.BondType.AROMATIC, 0]
            ligand_edge_features_list.append(fbond + [is_Aromatic, is_inring, 0, 1] + [innerdis[i, j]/10])
            ligand_edge_features_list.append(fbond + [is_Aromatic, is_inring, 0, 1] + [innerdis[i, j]/10])
            ligand_edges_list.append((i, j))
            ligand_edges_list.append((j, i))
    ligand_edge_index = torch.tensor(np.array(ligand_edges_list).T, dtype=torch.long)
    ligand_edge_attr = torch.tensor(np.array(ligand_edge_features_list),
                             dtype=torch.float)
    
    ligand_atom_graph = dgl.DGLGraph()
    ligand_atom_graph.add_nodes(len(ligand_pos))
    ligand_atom_graph.add_edges(ligand_edge_index[0], ligand_edge_index[1])
    ligand_atom_graph.ndata['pos'] = ligand_pos
    ligand_atom_graph.ndata['h'] = ligand_x
    ligand_atom_graph.edata['e'] = ligand_edge_attr
    return ligand_atom_graph


def get_atom_complex_graph(ligand_graph, protein_graph):
    g = dgl.DGLGraph()
    # add nodes
    num_atoms_ligand = len(ligand_graph.ndata['h'])  # number of ligand atoms
    num_atoms_protein = len(protein_graph.ndata['h'])
    num_atoms = num_atoms_ligand + num_atoms_protein
    g.add_nodes(num_atoms)
    g.ndata['h'] = torch.cat([ligand_graph.ndata['h'], protein_graph.ndata['h']])
    g.ndata['pos'] = torch.cat([ligand_graph.ndata['pos'], protein_graph.ndata['pos']])

    # add edges
    g.add_edges(ligand_graph.edges()[0], ligand_graph.edges()[1])
    g.add_edges(protein_graph.edges()[0]+num_atoms_ligand, protein_graph.edges()[1]+num_atoms_ligand)
    g.edata['e'] = torch.cat([ligand_graph.edata['e'], protein_graph.edata['e']], dim=0)
    return g


def get_atom_inter_graph(ligand_graph, protein_graph, inter_edge_index, inter_edge_attr):
    g = dgl.DGLGraph()
    # add nodes
    num_atoms_ligand = len(ligand_graph.ndata['h'])  # number of ligand atoms
    num_atoms_protein = len(protein_graph.ndata['h'])
    num_atoms = num_atoms_ligand + num_atoms_protein
    g.add_nodes(num_atoms)
    g.ndata['h'] = torch.cat([ligand_graph.ndata['h'], protein_graph.ndata['h']])
    g.ndata['pos'] = torch.cat([ligand_graph.ndata['pos'], protein_graph.ndata['pos']])
    
    # add edges
    g.add_edges(ligand_graph.edges()[0], ligand_graph.edges()[1])
    g.add_edges(inter_edge_index[0], inter_edge_index[1])
    g.edata['e'] = torch.cat([ligand_graph.edata['e'], inter_edge_attr], dim=0)
    return g


def get_atom_graphs(protein, ligand, threshhold=5):
    ligand_conf = ligand.GetConformer()
    ligand_positions = ligand_conf.GetPositions()
    protein_conf = protein.GetConformer()
    protein_positions = protein_conf.GetPositions()

    dis = ligand_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]
    dis = np.sqrt((dis * dis).sum(-1))
    idx = np.where(dis < threshhold)
    idx = [[i, j] for i, j in zip(idx[0], idx[1])]

    pidx = [i[1] for i in idx]
    pidx = sorted(list(set(pidx)))
    pidx = [i for i in pidx if protein.GetAtomWithIdx(int(i)).GetPDBResidueInfo().GetResidueName() != 'HOH']

    # get atom graph
    protein_atom_graph = get_protein_graph(protein, protein_positions, pidx)
    ligand_atom_graph = get_ligand_graph(ligand, ligand_positions)
    
    # get atom complex graph
    atom_complex_graph = get_atom_complex_graph(ligand_atom_graph, protein_atom_graph)

    # get atom inter graph
    edges_list = []
    edge_features_list = []
    nligand_atoms = len(ligand_atom_graph.ndata['h'])
    pidx2tidx = {pidx[i]: i + nligand_atoms for i in range(len(pidx))}
    for b in idx:
        try:
            edges_list.append((b[0], pidx2tidx[b[1]]))
            edge_feature = [0, 0, 0, 0, 1, 0, 0, 1, 1] + [dis[b[0], b[1]]/10]
            edge_features_list.append(edge_feature)
        except:
            continue
    inter_edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    inter_edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)
    atom_inter_graph = get_atom_inter_graph(ligand_atom_graph, protein_atom_graph, inter_edge_index, inter_edge_attr)
    
    return ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph


def get_ligand_motif_graph(ligand, ligand_atom_graph):
    atom_pos = ligand.GetConformer().GetPositions().astype(np.float32)
    breaks_bonds, cliques = tree_decomp(ligand)
    motif_atom_feature = []
    num_motif = len(cliques)
    motif_pos = []
    if num_motif > 0:
        for motif in cliques:
            pos = atom_pos[motif]
            motif_pos.append(np.mean(pos, axis=0))
            motif_smi = Chem.MolFragmentToSmiles(ligand, motif)
            motif_mol = Chem.MolFromSmiles(motif_smi)
            try:
                emb_0 = maccskeys_emb(motif_mol)
                emb_1 = pharm_property_types_feats(motif_mol)
            except Exception:
                print('error error error')
                emb_0 = [0 for i in range(167)]
                emb_1 = [0 for i in range(27)]
            motif_atom_feature.append(emb_0 + emb_1)
    
    bond_index = torch.stack(ligand_atom_graph.edges(), dim=1)
    bond_feature = ligand_atom_graph.edata['e']
    motif_edge_index, motif_edge_attr = [], []
    for idx, (i, j) in enumerate(bond_index):
        if f'{i}_{j}' in list(breaks_bonds.keys()):
            motif_edge_index.append([breaks_bonds[f'{i}_{j}'][0],
                                     breaks_bonds[f'{i}_{j}'][1]])
            motif_edge_attr.append(bond_feature[idx])
            motif_edge_index.append([breaks_bonds[f'{i}_{j}'][1],
                                     breaks_bonds[f'{i}_{j}'][0]])
            motif_edge_attr.append(bond_feature[idx])
    
    ligand_motif_graph = dgl.DGLGraph()
    ligand_motif_graph.add_nodes(len(motif_pos))
    ligand_motif_graph.ndata['pos'] = torch.tensor(np.array(motif_pos), dtype=torch.float)
    ligand_motif_graph.ndata['h'] = torch.tensor(np.array(motif_atom_feature), dtype=torch.float)
    motif_edge_index = torch.tensor(np.array(motif_edge_index).T)
    if len(motif_edge_index) != 0:
        ligand_motif_graph.add_edges(motif_edge_index[0], motif_edge_index[1])
    ligand_motif_graph.edata['e'] = torch.tensor(np.array(motif_edge_attr), dtype=torch.float)
    return ligand_motif_graph


def get_pocket_dict(ligand_graph, protein_dict, threshhold):
    ligand_positions = ligand_graph.ndata['pos']
    protein_positions = protein_dict['pos_CA']
    dis = ligand_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]
    dis = np.sqrt((dis * dis).sum(-1))
    idx = np.where(dis < threshhold)
    idx = [[i, j] for i, j in zip(idx[0], idx[1])]

    pidx = [i[1] for i in idx]
    pidx = sorted(list(set(pidx)))

    pocket_res = ''
    pocket_pos = []
    for i in pidx:
        pocket_res += protein_dict['amino_acid'][i]
        pocket_pos.append(protein_positions[i])
    pocket_feature = cal_seq_feats(pocket_res)
    seq_feature = cal_seq_feats(protein_dict['amino_acid'])

    target_len = 125
    pocket_seq_feature = np.zeros((target_len, 33))
    if seq_feature.shape[0] < target_len:
        pocket_seq_feature[:seq_feature.shape[0]] = seq_feature
    else:
        pocket_seq_feature = seq_feature[:target_len]
    
    pocket_dict = {'pocket_seq_feature': torch.tensor(pocket_seq_feature),
                   'pocket_feature': torch.tensor(pocket_feature),
                   'pocket_pos': torch.tensor(np.array(pocket_pos), dtype=torch.float),}
    return pocket_dict


def get_motif_inter_graph(lig_motif_graph, pocket_coord, dis_threshold):
    lig_coord = lig_motif_graph.ndata['pos']
    ## IGN
    # add interaction edges, only consider the euclidean distance within dis_threshold
    num_atoms = len(lig_coord) + len(pocket_coord)
    g = dgl.DGLGraph()
    g.add_nodes(num_atoms)
    # dis_matrix = distance_matrix(lig_coord, pocket_coord)
    # node_idx = np.where(dis_matrix < dis_threshold)

    dis = lig_coord[:, np.newaxis, :] - pocket_coord[np.newaxis, :, :]
    dis = np.sqrt((dis * dis).sum(-1))
    node_idx = np.where(dis < dis_threshold)

    src_ls3 = np.concatenate([node_idx[0]])
    dst_ls3 = np.concatenate([node_idx[1] + len(lig_coord)])
    g.add_edges(src_ls3, dst_ls3)

    # 'd', distance between ligand atoms and pocket atoms
    inter_dis = np.concatenate([dis[node_idx[0], node_idx[1]]])
    g_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)
    g.edata['e'] = g_d * 0.1
    return g


def get_motif_graphs(protein_path, ligand, ligand_atom_graph, threshhold=8):
    # Fingerprint feature
    ligand_fp_feature = np.array([get_fp_feature(ligand)])
    
    ligand_motif_graph = get_ligand_motif_graph(ligand, ligand_atom_graph)

    protein = PDBProtein(protein_path)
    protein_dict = protein.to_dict_residue()

    pocket_dict = get_pocket_dict(ligand_motif_graph, protein_dict, threshhold)

    motif_inter_graph = get_motif_inter_graph(ligand_motif_graph,
                                              pocket_dict['pocket_pos'], 
                                              dis_threshold=threshhold,)

    pocket_dict['ligand_fp_feature'] = ligand_fp_feature
    return ligand_motif_graph, pocket_dict, motif_inter_graph