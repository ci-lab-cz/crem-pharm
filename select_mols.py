from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from itertools import combinations
from read_input import read_input
from operator import itemgetter
import numpy as np


def fused_ring_atoms(m):
    # count rings considering fused and spiro cycles as a single ring system
    # print(rings('C1CC23CCC2CC13'))  # 1
    # print(rings('O=C(N1CCNCC1)c1ccc(=O)oc1'))  # 2
    # print(rings('O=C(C1CCC(=O)C23CCCC2CCC13)N1CCNC2CCCC12'))  # 2
    # print(rings('O=C(C1CCC(=O)C23CCCC2CCC13)N1CCNC2C1CCC21CCCC1'))  # 2
    # print(rings('C1CC2(C1)CC1(C2)C2CCC22CCC12'))  # 1
    # print(rings('CC12CCC(C1)C1CCC21'))  # 1
    # print(rings('CC12CCC3(CCC3C1)C2'))  # 1
    # print(rings('CC'))  # 0
    # print(rings('C1CC2CCCC(C1)CCCC2'))  # 1
    q = m.GetRingInfo()
    rings = [set(r) for r in q.AtomRings()]
    go_next = True
    while go_next:
        go_next = False
        for i, j in combinations(range(len(rings)), 2):
            if rings[i] & rings[j]:
                q = rings[i] | rings[j]
                del rings[j], rings[i]
                rings.append(q)
                go_next = True
                break
    return rings


# def check_substr_mols(small, large):
#     small = Chem.RemoveHs(small)
#     large = Chem.RemoveHs(large)
#     large_ring_ids = fused_ring_atoms(large)
#     small_nrings = rdMolDescriptors.CalcNumRings(small)
#     if small_nrings == 0:
#         return large.HasSubstructMatch(small)
#     else:
#         for ids in large.GetSubstructMatches(small):
#             for r_ids in large_ring_ids:
#                 if set(ids).intersection(r_ids) == set(r_ids):
#                     return True
#         return False


input_fname = 'test_project/screen.pkl'
mols = [(mol, mol_name, mol.GetNumHeavyAtoms()) for mol, mol_name in read_input(input_fname)]

# smis = ['C1=CC=NC=C1', 'C1=CC=NC=C1C', 'CC1=CC=NC2=C1C=CO2', 'CC1=CC=NC2=C1C=C(F)O2', 'CC(=O)CC1=CC=NC2=C1C=C(F)O2', 'CC(=O)C']
# mols = [(Chem.MolFromSmiles(smi), smi, Chem.MolFromSmiles(smi).GetNumHeavyAtoms()) for smi in smis]

# mols = sorted(mols, key=itemgetter(2))
# hacs = np.array([item[2] for item in mols])
# deleted = np.zeros(hacs.shape)
#
# for i, (mol, mol_name, hac) in enumerate(mols):
#     print(i, mol_name, sum(deleted))
#     for j in np.where(np.logical_and(hacs <= hac, deleted == 0))[0]:
#         if i != j and check_substr_mols(mols[j][0], mol):
#             deleted[i] = 1
#
# mols = [mols[i] for i in np.where(deleted == 0)[0]]
#
# for m, name, hac in mols:
#     print(name, Chem.MolToSmiles(Chem.RemoveHs(m)), m.GetNumConformers())



# from rdkit.Chem.rdmolops import PatternFingerprint
#
# print(len(mols))
#
# p = {mol_id: PatternFingerprint(mol) for mol, mol_id, hac in mols}
#
# print(p)


from functools import partial

def check_substr_mols(large, large_ring_ids, small):
    """

    :param large: larger mol
    :param large_ring_ids: list of sets with ids of ring systems of a larger mol
    :param small: smaller mol
    :return:
    """
    small_nrings = rdMolDescriptors.CalcNumRings(small)
    if small_nrings == 0:
        return large.HasSubstructMatch(small)
    else:
        for ids in large.GetSubstructMatches(small):
            ids = set(ids)
            for r_ids in large_ring_ids:
                if r_ids.issubset(ids):
                    return True
        return False

def select_mols(mols, ncpu=1):
    """
    Remove those molecules which are superstructure of another one. Thus, if CO and CCO matched a pharmacophore
    the latter is superfluous and can be removed. It is expected that if needed the former will be able to grow
    to CCO and further.
    :param mols:
    :param ncpu:
    :return:
    """

    pool = None

    mols = [(Chem.RemoveHs(mol), mol.GetNumHeavyAtoms()) for mol in mols]
    mols = [(mol, hac, fused_ring_atoms(mol)) for mol, hac in mols]   # mol, hac, list of sets with ids of ring systems required for substructure check
    mols = sorted(mols, key=itemgetter(1))
    hacs = np.array([item[1] for item in mols])
    deleted = np.zeros(hacs.shape)

    for i, (mol, hac, ring_ids) in enumerate(mols):
        if not deleted[i]:
            mol_ids = np.where(np.logical_and(hacs >= hac, deleted == 0))[0]
            mol_ids = np.delete(mol_ids, np.argwhere(mol_ids == i))
            if pool is None:
                remove_ids = [j for j in mol_ids if check_substr_mols(mols[j][0], mols[j][2], mol)]
            else:
                mask = list(pool.starmap(partial(check_substr_mols, small=mol),
                                         ((mols[mol_id][0], mols[mol_id][2]) for mol_id in mol_ids)))
                remove_ids = mol_ids[mask]

            deleted[remove_ids] = 1

    return [mols[i][0] for i in np.where(deleted == 0)[0]]

mols2 = select_mols([m[0] for m in mols])

print(len(mols2))

import pickle
with open('111.pkl', 'wb') as f:
    pickle.dump(mols2, f)
