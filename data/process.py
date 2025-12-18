import os
import pickle
import random
from tqdm import tqdm
from rdkit import Chem
from protein_ligand import get_mol, get_atom_graphs, get_motif_graphs


def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res


def GetPDBList(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = []
    for line in lines:
        if "//" in line:
            temp = line.split()
            res.append(temp[0])
    random.shuffle(res)
    return res


def process_raw_data(dataset_path, test_data_list):
    index_refined_path = os.path.join(dataset_path, 'index/INDEX_refined_data.2016')
    res = GetPDBDict(Path=index_refined_path)
    protein_list = GetPDBList(Path=index_refined_path)
    G_train_list, G_val_list, G_test_list = [], [], []
    val_i = 0
    for item in tqdm(protein_list):
        score = res[item]
        if item in test_data_list:
            lig_file_name = dataset_path + 'CASF-2016/coreset/' + item + '/' + item + '_ligand_opt.mol2'
            pocket_file_name = dataset_path + 'CASF-2016/coreset/' + item + '/' + item + '_pocket.pdb'
            protein_path = dataset_path + 'CASF-2016/coreset/' + item + '/'
            lig_mol, pocket_mol = get_mol(lig_file_name, pocket_file_name)
            if lig_mol is None or pocket_mol is None:
                continue
            
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            G_test_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
        else:
            lig_file_name = dataset_path + 'refined-set/' + item + '/' + item + '_ligand.mol2'
            pocket_file_name = dataset_path + 'refined-set/' + item + '/' + item + '_pocket.pdb'
            protein_path = dataset_path + 'refined-set/' + item + '/'
            if not os.path.exists(protein_path):
                print('path not exists')
                print(item+' file path not exist')
                continue
            lig_mol, pocket_mol = get_mol(lig_file_name, pocket_file_name)
            if lig_mol is None or pocket_mol is None:
                continue
            
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            
            if val_i < 1000:
                G_val_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
            else:
                G_train_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                    ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
            val_i += 1
    
    with open(os.path.join(dataset_path, 'dock2/val_data.pkl'), 'wb') as f:
        pickle.dump(G_val_list, f)
    f.close()
    with open(os.path.join(dataset_path, 'dock2/test_data.pkl'), 'wb') as f:
        pickle.dump(G_test_list, f)
    f.close()

    index_general_path = os.path.join(dataset_path, 'index/INDEX_general_PL_data.2016')
    res = GetPDBDict(Path=index_general_path)
    protein_list = GetPDBList(Path=index_general_path)
    for item in tqdm(protein_list):
        score = res[item]
        lig_file_name = dataset_path + 'general-set-except-refined/' + item + '/' + item + '_ligand.mol2'
        pocket_file_name = dataset_path + 'general-set-except-refined/' + item + '/' + item + '_pocket.pdb'
        protein_path = dataset_path + 'general-set-except-refined/' + item + '/'
        if not os.path.exists(protein_path):
            print('path not exists')
            print(item)
            continue
        lig_mol, pocket_mol = get_mol(lig_file_name, pocket_file_name)
        if lig_mol is None or pocket_mol is None:
            continue
        
        ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
        ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
        G_train_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                            ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
    
    print('train sample num: ', len(G_train_list))
    print('val sample num: ', len(G_val_list))
    print('test sample num: ', len(G_test_list))
    with open(os.path.join(dataset_path, 'dock2/train_data.pkl'), 'wb') as f:
        pickle.dump(G_train_list, f)
    f.close()


def process_dude_data(dataset_path):
    G_train_list = []
    data_list = os.listdir(dataset_path)
    for item in data_list:
        print('start:')
        print(item)
        data_path = os.path.join(dataset_path, item)
        pocket_file_name = os.path.join(data_path, 'receptor.pdb')
        pocket_mol = Chem.MolFromPDBFile(pocket_file_name, sanitize=True)
        try:
            protein_conf = pocket_mol.GetConformer()
            protein_positions = protein_conf.GetPositions()
        except:
            print('error error error')
            print(item)
            continue
        lig_file_name = os.path.join(data_path, 'actives_final.sdf')
        score = 1
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_train_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
        lig_file_name = os.path.join(data_path, 'decoys_final.sdf')
        score = 0
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_train_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
            
    print('train sample num: ', len(G_train_list))
    with open('./dataset/dude/dock2/dude_train_data.pkl', 'wb') as f:
        pickle.dump(G_train_list, f)
    f.close()


def process_test_data(dataset_path):
    G_test_list = []
    data_list = os.listdir(dataset_path)
    for item in data_list:
        data_path = os.path.join(dataset_path, item)
        pocket_file_name = os.path.join(data_path, f'{item}_prot', f'{item}_p.pdb')
        pocket_mol = Chem.MolFromPDBFile(pocket_file_name, sanitize=True)
        try:
            protein_conf = pocket_mol.GetConformer()
            protein_positions = protein_conf.GetPositions()
        except:
            print('error error error')
            print(item)
            continue
        lig_file_name = os.path.join(data_path, f'{item}_actives.sdf')
        score = 1
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_test_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
        lig_file_name = os.path.join(data_path, f'{item}_decoys.sdf')
        score = 0
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_test_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
            
    print('test sample num: ', len(G_test_list))
    with open('./dataset/dude/dock2/dekois_test_data.pkl', 'wb') as f:
        pickle.dump(G_test_list, f)
    f.close()


def process_litpcba_data(dataset_path):
    G_test_list = []
    G_train_list = []
    data_list = ['FEN1_5fv7', 'ALDH1_4x4l', 'GBA_2v3e', 'KAT2A_5h84', 'MAPK1_2ojg', 'PKM2_3gr4', 'VDR_3a2j']
    for item in data_list:
        data_path = os.path.join(dataset_path, item)
        pocket_file_name = os.path.join(data_path, f'{item}_prot', f'{item}_p.pdb')
        pocket_mol = Chem.MolFromPDBFile(pocket_file_name, sanitize=True)
        try:
            protein_conf = pocket_mol.GetConformer()
            protein_positions = protein_conf.GetPositions()
        except:
            print('error error error')
            print(item)
            continue
        # test
        lig_file_name = os.path.join(data_path, f'{item}_SP_active_V.sdf')
        score = 1
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_test_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
        lig_file_name = os.path.join(data_path, f'{item}_SP_inactive_V.sdf')
        score = 0
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_test_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
        # train
        lig_file_name = os.path.join(data_path, f'{item}_SP_active_T.sdf')
        score = 1
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_train_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
        lig_file_name = os.path.join(data_path, f'{item}_SP_inactive_T.sdf')
        score = 0
        mols = []
        if os.path.exists(lig_file_name):
            suppl = Chem.SDMolSupplier(lig_file_name)
            mols = [mol for mol in suppl if mol]
        print(len(mols))
        for lig_mol in mols:
            ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph = get_atom_graphs(pocket_mol, lig_mol, threshhold=5)
            if ligand_atom_graph is None:
                continue
            ligand_motif_graph, protein_res_dict, motif_inter_graph = get_motif_graphs(pocket_file_name, lig_mol, ligand_atom_graph, threshhold=8)
            if ligand_motif_graph is None:
                continue
            G_train_list.append([ligand_atom_graph, protein_atom_graph, atom_complex_graph, atom_inter_graph, 
                                ligand_motif_graph, protein_res_dict, motif_inter_graph, score])
            
        print('test sample num: ', item, len(G_test_list))
        with open(os.path.join(dataset_path, 'dock2', f'{item}_test.pkl'), 'wb') as f:
            pickle.dump(G_test_list, f)
        f.close()
        print('train sample num: ', item, len(G_train_list))
        with open(os.path.join(dataset_path, 'dock2', f'{item}_train.pkl'), 'wb') as f:
            pickle.dump(G_train_list, f)
        f.close()


if __name__ == '__main__':
    # process pdbbind v2016
    raw_data_path = '/home/cuda/nfs3/lius/dataset/pdbbind/v2016/'
    test_data_list = os.listdir(os.path.join(raw_data_path, 'CASF-2016/coreset'))
    process_raw_data(raw_data_path, test_data_list)
    
    # train at DUD-E, test at DEKOIS_2.0x
    # raw_data_path = '/home/cuda/nfs3/lius/dataset/dude/dataset/'
    # process_dude_data(raw_data_path)
    # raw_data_path = '/home/cuda/nfs3/lius/dataset/dude/DEKOIS_2.0x/'
    # process_dekois_data(raw_data_path)
    
    #process LIT-PCBA
    # raw_data_path = '/home/cuda/nfs3/lius/dataset/LIT-PCBA/'
    # process_litpcba_data(raw_data_path)
