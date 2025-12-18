from rdkit.Chem import BRICS


def get_cliques_link(breaks, cliques):
    clique_id_of_node = {}
    for idx, clique in enumerate(cliques):
        for c in clique:
            clique_id_of_node[c] = idx
    breaks_bond = {}
    for bond in breaks:
        if bond[0] in clique_id_of_node.keys() and bond[1] in clique_id_of_node.keys():
            breaks_bond[f'{bond[0]}_{bond[1]}'] = [clique_id_of_node[bond[0]], clique_id_of_node[bond[1]]]
    return breaks_bond


def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0], [0]]

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])
            breaks.append([bond[0][0], bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            elif mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)
            bond = mol.GetBondBetweenAtoms(c[0], c[1])
            if mol.GetAtomWithIdx(c[1]).IsInRing() and mol.GetAtomWithIdx(c[0]).IsInRing() and not bond.IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                cliques.append([c[1]])
                breaks.append(c)

    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]

    breaks = get_cliques_link(breaks, cliques)
    return breaks, cliques