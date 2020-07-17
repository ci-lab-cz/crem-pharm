from rdkit import Chem
from rdkit.Chem import AllChem
from psearch.database import DB
from pmapper.pharmacophore import Pharmacophore as P
import pickle


def keep_confs(mol, ids):

    all_ids = set(c.GetId() for c in mol.GetConformers())
    remove_ids = all_ids - set(ids)
    for cid in set(remove_ids):
        mol.RemoveConformer(cid)
    # conformers are reindexed staring with 0 step 1
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)
    return mol


p2 = P()
p2.load_from_xyz('test_project/1.xyz')
m2 = p2.get_mol()
m2_nfeatures = 3

db = DB('test_project/7_stereo_ids_rms1.dat')

w = Chem.SDWriter('test_project/screen.sdf')
wpkl = open('test_project/screen.pkl', 'wb')

mol_names = db.get_mol_names()

# print(mol_names)
#
# print(len(mol_names))

for j, mol_name in enumerate(mol_names):

    print(j, mol_name)

    pharm = db.get_pharm(mol_name)[0]

    rmsd_dict = dict()
    for i, coords in enumerate(pharm):
        p1 = P()
        p1.load_from_feature_coords(coords)
        m1 = p1.get_mol()
        min_rmsd = float('inf')
        min_ids = None
        for ids1 in m1.GetSubstructMatches(m2):
            a = AllChem.AlignMol(Chem.Mol(m1), m2, atomMap=tuple(zip(ids1, range(m2_nfeatures))))
            if a < min_rmsd:
                min_rmsd = a
                min_ids = ids1
        # print(min_rmsd)
        if min_rmsd <= 0.2:
            rmsd_dict[i] = AllChem.GetAlignmentTransform(Chem.Mol(m1), m2, atomMap=tuple(zip(min_ids, range(m2_nfeatures))))

    if rmsd_dict:
        # print(rmsd_dict)
        m = db.get_mol(mol_name)[0]
        m.SetProp('_Name', mol_name)
        for k, (rms, matrix) in rmsd_dict.items():
            AllChem.TransformMol(m, matrix, k, keepConfs=True)
            m.SetProp('rms', str(rms))
            w.write(m, k)
        m = keep_confs(m, rmsd_dict.keys())
        pickle.dump((m, mol_name), wpkl, -1)

