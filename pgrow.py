#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
import json
import pickle
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
from itertools import combinations, product
from operator import itemgetter
import numpy as np
import pandas as pd
import sqlite3
import time
from math import cos, sin, pi

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
# from rdkit.Chem.AllChem import AlignMol, EmbedMultipleConfs
# from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Geometry.rdGeometry import Point3D
from crem.crem import grow_mol
from pmapper.utils import load_multi_conf_mol
from pmapper.pharmacophore import Pharmacophore as P
from psearch.database import DB
from read_input import read_input
from openbabel import openbabel as ob
from openbabel import pybel
from dask.distributed import Client
from dask import bag


class PharmModel2(P):

    def __init__(self, bin_step=1, cached=False):
        super().__init__(bin_step, cached)
        self.clusters = defaultdict(set)
        self.exclvol = None

    def get_num_features(self):
        return len(self._get_ids())

    def get_xyz(self, ids):
        return np.array([xyz for label, xyz in self.get_feature_coords(ids)])

    def set_clusters(self, clustering_threshold, init_ids=None):
        """

        :param clustering_threshold:
        :param ids: ids of features selected as a starting pharmacophore
        :return:
        """
        ids = tuple(set(self._get_ids()) - set(init_ids))
        coords = self.get_xyz(ids)
        c = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_threshold).fit(coords)
        for i, j in enumerate(c.labels_):
            self.clusters[j].add(ids[i])

    def get_subpharmacophore(self, ids):
        coords = self.get_feature_coords(ids)
        p = P()
        p.load_from_feature_coords(coords)
        return p

    def select_nearest_cluster(self, ids):

        def min_distance(ids1, ids2):
            xyz1 = self.get_xyz(ids1)
            xyz2 = self.get_xyz(ids2)
            return np.min(cdist(xyz1, xyz2))

        ids = set(ids)
        selected_ids = tuple()
        min_dist = float('inf')
        for k, v in self.clusters.items():
            if not v & set(ids):
                dist = min_distance(ids, v)
                if dist < min_dist:
                    min_dist = dist
                    selected_ids = tuple(v)
        return selected_ids

    def get_feature_coords_pd(self, ids=None):
        ids = self._get_ids(ids)
        data = [(i, label, x, y, z) for i, (label, (x, y, z)) in zip(ids, self.get_feature_coords(ids))]
        coords = pd.DataFrame(data, columns=['id', 'label', 'x', 'y', 'z'])
        return coords

    def load_from_xyz(self, fname):
        self.exclvol = []
        with open(fname) as f:
            feature_coords = []
            f.readline()
            line = f.readline().strip()
            if line:
                opts = dict(item.split('=') for item in line.split(';'))
                if 'bin_step' in opts:
                    self.update(bin_step=float(opts['bin_step']))
            for line in f:
                label, *coords = line.strip().split()
                coords = tuple(map(float, coords))
                if label != 'e':
                    feature_coords.append((label, coords))
                else:
                    self.exclvol.append(coords)
            self.load_from_feature_coords(tuple(feature_coords))
        if self.exclvol:
            self.exclvol = np.array(self.exclvol)
        else:
            self.exclvol = None


def gen_stereo(mol):
    stereo_opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    for b in mol.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOANY:
            b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    isomers = tuple(EnumerateStereoisomers(mol, options=stereo_opts))
    return isomers


def ConstrainedEmbedMultipleConfs(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342, **kwargs):

    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    cids = list(cids)
    if len(cids) == 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        for cid in cids:
            ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            for i, idxI in enumerate(match):
                for j in range(i + 1, len(match)):
                    idxJ = match[j]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.)
            ff.Initialize()
            n = 4
            more = ff.Minimize()
            while more and n:
                more = ff.Minimize()
                n -= 1
            # rotate the embedded conformation onto the core:
            rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        for cid in cids:
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
            ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            conf = core.GetConformer()
            for i in range(core.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
            ff.Initialize()
            n = 4
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            while more and n:
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                n -= 1
            # realign
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol


def __embed_multiple_confs_constrained_ob(mol, template_mol, nconf, seed=42, **kwargs):
    # generate constrained conformers using OpenBabel, if fails use RDKit
    # **kwargs arguments are for RDKit generator

    mol = AllChem.ConstrainedEmbed(Chem.AddHs(mol), template_mol, randomseed=seed)
    ids = set(range(mol.GetNumAtoms())) - set(mol.GetSubstructMatch(template_mol))
    ids = [i + 1 for i in ids]   # ids of atoms to rotate
    mol_rdkit = mol

    mol = pybel.readstring('mol', Chem.MolToMolBlock(mol)).OBMol   # convert mol from RDKit to OB

    pff = ob.OBForceField_FindType("mmff94")
    if not pff.Setup(mol):  # if OB FF setup fails use RDKit conformer generation (slower)
        return ConstrainedEmbedMultipleConfs(Chem.AddHs(mol_rdkit), Chem.RemoveHs(template_mol), nconf, randomseed=seed, **kwargs)

    constraints = ob.OBFFConstraints()
    for atom in ob.OBMolAtomIter(mol):
        atom_id = atom.GetIndex() + 1
        if atom_id not in ids:
            constraints.AddAtomConstraint(atom_id)
    pff.SetConstraints(constraints)

    pff.DiverseConfGen(0.5, 1000, 50, False)   # rmsd, nconf_tries, energy, verbose

    pff.GetConformers(mol)
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('mol')

    output_strings = []
    for conf_num in range(max(0, mol.NumConformers() - nconf), mol.NumConformers()):   # save last nconf conformers (it seems the last one is the original conformer)
        mol.SetConformer(conf_num)
        output_strings.append(obconversion.WriteString(mol))

    out_mol = Chem.MolFromMolBlock(output_strings[0])
    for a in output_strings[1:]:
        out_mol.AddConformer(Chem.MolFromMolBlock(a).GetConformer(0), assignId=True)
    return out_mol


def __gen_confs(mol, template_mol=None, nconf=10, seed=42, alg='rdkit', **kwargs):
    # alg - 'rdkit' or 'ob'
    try:
        if template_mol:
            if alg == 'rdkit':
                mol = ConstrainedEmbedMultipleConfs(Chem.AddHs(mol), Chem.RemoveHs(template_mol), nconf, randomseed=seed, **kwargs)
            elif alg == 'ob':
                mol = __embed_multiple_confs_constrained_ob(mol, template_mol, nconf, seed, **kwargs)
        else:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=nconf*4, randomSeed=seed)
    except ValueError:
        return None
    return mol


def get_grow_points2(mol_xyz, pharm_xyz, tol=2):
    dist = np.min(cdist(mol_xyz, pharm_xyz), axis=1)
    ids = np.flatnonzero(dist <= (np.min(dist) + tol))
    return ids.tolist()


def get_grow_atom_ids(mol, pharm_xyz, tol=2):

    # search for the closest growing points among heavy atoms with attached hydrogens only

    ids = set()
    atoms_with_h = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1 and a.GetTotalNumHs() > 0:
            atoms_with_h.append(a.GetIdx())
    for c in mol.GetConformers():
        ids.update(get_grow_points2(c.GetPositions()[atoms_with_h], pharm_xyz, tol))
    res = list(sorted(atoms_with_h[i] for i in ids))
    return res


def remove_confs_rms(mol, rms=0.25):

    remove_ids = []
    mol_tmp = Chem.RemoveHs(mol)   # calc rms for heavy atoms only
    match_ids = mol_tmp.GetSubstructMatches(mol_tmp, uniquify=False, useChirality=True)

    # determine best rms taking into account symmetry of a molecule
    rms_list = []
    cids = [c.GetId() for c in mol_tmp.GetConformers()]
    for i, j in combinations(cids, 2):
        best_rms = float('inf')
        for ids in match_ids:
            rms = np.sqrt(np.mean(np.sum((mol_tmp.GetConformer(i).GetPositions() - mol_tmp.GetConformer(j).GetPositions()[ids, ]) ** 2, axis=1)))
            if rms < best_rms:
                best_rms = rms
        rms_list.append((i, j, best_rms))

    while any(item[2] < rms for item in rms_list):
        for item in rms_list:
            if item[2] < rms:
                remove_ids.append(item[1])
                rms_list = [i for i in rms_list if i[0] != item[1] and i[1] != item[1]]
                break

    for cid in set(remove_ids):
        mol.RemoveConformer(cid)

    return mol


def reassing_conf_ids(mol):
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)


def remove_conf(mol, cids):
    for cid in set(cids):
        mol.RemoveConformer(cid)
    reassing_conf_ids(mol)
    return mol


def keep_confs(mol, ids):

    all_ids = set(c.GetId() for c in mol.GetConformers())
    remove_ids = all_ids - set(ids)
    for cid in set(remove_ids):
        mol.RemoveConformer(cid)
    reassing_conf_ids(mol)
    return mol


def is_collinear(p, epsilon=0.01):
    a = np.array([xyz for label, xyz in p.get_feature_coords()])
    a = np.around(a, 3)
    a = np.unique(a, axis=0)
    if a.shape[0] < 2:
        raise ValueError('Too few pharmacophore features were supplied. At least two features with distinct '
                         'coordinates are required.')
    if a.shape[0] == 2:
        return True
    d = cdist(a, a)
    id1, id2 = np.where(d == np.max(d))[0]   # most distant features
    max_dist = d[id1, id2]
    for i in set(range(d.shape[0])) - {id1, id2}:
        if max_dist - d[i, id1] - d[i, id2] > epsilon:
            return False
    return True


def get_rotate_matrix(p1, p2, theta):

    # https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas
    diff = [j - i for i, j in zip(p1, p2)]
    squaredSum = sum([i ** 2 for i in diff])
    u2, v2, w2 = [i ** 2 / squaredSum for i in diff]
    u = u2 ** 0.5 * (1 if diff[0] >= 0 else -1)   # to keep the sign
    v = v2 ** 0.5 * (1 if diff[1] >= 0 else -1)
    w = w2 ** 0.5 * (1 if diff[2] >= 0 else -1)

    c = cos(pi * theta / 180)
    s = sin(pi * theta / 180)

    x, y, z = p1

    m = [[u2 + (v2 + w2) * c,       u * v * (1 - c) - w * s,   u * w * (1 - c) + v * s,   (x * (v2 + w2) - u * (y * v + z * w)) * (1 - c) + (y * w - z * v) * s],
         [u * v * (1 - c) + w * s,  v2 + (u2 + w2) * c,        v * w * (1 - c) - u * s,   (y * (u2 + w2) - v * (x * u + z * w)) * (1 - c) + (z * u - x * w) * s],
         [u * w * (1 - c) - v * s,  v * w * (1 - c) + u * s,   w2 + (u2 + v2) * c,        (z * (u2 + v2) - w * (x * u + y * v)) * (1 - c) + (x * v - y * u) * s],
         [0, 0, 0, 1]]

    return np.array(m)


def get_rotate_matrix_from_collinear_pharm(p, theta):
    a = np.array([xyz for label, xyz in p.get_feature_coords()])
    a = np.around(a, 3)
    a = np.unique(a, axis=0)
    d = cdist(a, a)
    id1, id2 = np.where(d == np.max(d))[0]   # most distant features
    return get_rotate_matrix(a[id1].tolist(), a[id2].tolist(), theta)


def screen(mol_name, pharm_list, query_mol, query_nfeatures, rmsd):

    # return one map, this can be a problem for fragments with different orientations and the same rmsd
    # [NH2]c1nc([NH2])ncc1 - of two NH2 and n between were matched

    rmsd_dict = dict()
    for i, coords in enumerate(pharm_list):
        p1 = P()
        p1.load_from_feature_coords(coords)
        m1 = p1.get_mol()
        min_rmsd = float('inf')
        min_ids = None
        for ids1 in m1.GetSubstructMatches(query_mol):
            a = AllChem.AlignMol(Chem.Mol(m1), query_mol, atomMap=tuple(zip(ids1, range(query_nfeatures))))
            if a < min_rmsd:
                min_rmsd = a
                min_ids = ids1
        if min_rmsd <= rmsd:
            rmsd_dict[i] = AllChem.GetAlignmentTransform(Chem.Mol(m1), query_mol,
                                                         atomMap=tuple(zip(min_ids, range(query_nfeatures))))
    return mol_name, rmsd_dict


def screen_mp(items):
    return screen(*items)


def supply_screen(db, query_mol, rmsd):
    n = query_mol.GetNumAtoms()
    mol_names = db.get_mol_names()
    for mol_name in mol_names:
        pharm_dict = db.get_pharm(mol_name)
        try:
            pharm_list = pharm_dict[0]   # because there is only one stereoisomer for each entry
        except KeyError:
            sys.stderr.write(f'{mol_name} does not have pharm_dict[0]\n')
            continue
        yield mol_name, pharm_list, query_mol, n, rmsd


def screen_pmapper(query_pharm, db_fname, output_sdf, rmsd, ncpu):

    db = DB(db_fname)
    query_mol = query_pharm.get_mol()

    pool = Pool(ncpu)
    d = []
    for mol_name, rmsd_dict in pool.imap_unordered(screen_mp, supply_screen(db, query_mol, rmsd)):
        if rmsd_dict:
            d.append((mol_name, rmsd_dict))

    if not d:
        return False

    # for additional sampling of collinear pharmacophores
    theta = 10
    if is_collinear(query_pharm):
        rotate_mat = get_rotate_matrix_from_collinear_pharm(query_pharm, theta)
    else:
        rotate_mat = None

    w = Chem.SDWriter(output_sdf)
    for mol_name, rmsd_dict in d:
        m = db.get_mol(mol_name)[0]
        m.SetProp('_Name', mol_name)
        for k, (rms, matrix) in rmsd_dict.items():
            AllChem.TransformMol(m, matrix, k, keepConfs=True)
            m.SetProp('rms', str(rms))
            w.write(m, k)
            if rotate_mat is not None:  # rotate molecule if pharmacophore features are collinear
                for _ in range(360 // theta - 1):
                    AllChem.TransformMol(m, rotate_mat, k, keepConfs=True)
                    w.write(m, k)
    w.close()

    return True


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


def check_substr_mols(small, large):
    small = Chem.RemoveHs(small)
    large = Chem.RemoveHs(large)
    large_ring_ids = fused_ring_atoms(large)
    small_nrings = rdMolDescriptors.CalcNumRings(small)
    if small_nrings == 0:
        return large.HasSubstructMatch(small)
    else:
        for ids in large.GetSubstructMatches(small):
            for r_ids in large_ring_ids:
                if set(ids).intersection(r_ids) == set(r_ids):
                    return True
        return False


def select_mols(mols):
    """
    Remove those molecules which are superstructure of another one. Thus, if CO and CCO matched a pharmacophore
    the latter is superfluous and can be removed. It is expected that if needed the former will be able to grow
    to CCO and further.
    :param mols:
    :return:
    """
    mols = [(Chem.RemoveHs(mol), mol.GetNumHeavyAtoms()) for mol in mols]
    mols = sorted(mols, key=itemgetter(1))
    hacs = np.array([item[1] for item in mols])
    deleted = np.zeros(hacs.shape)

    for i, (mol, hac) in enumerate(mols):
        for j in np.where(np.logical_and(hacs <= hac, deleted == 0))[0]:
            if i != j and check_substr_mols(mols[j][0], mol):
                deleted[i] = 1

    return [mols[i][0] for i in np.where(deleted == 0)[0]]


def save_mols(mols, sdf_fname):
    w = Chem.SDWriter(sdf_fname)
    try:
        with open(sdf_fname.rsplit('.', 1)[0] + '.pkl', 'wb') as wpkl:
            for mol in mols:
                pickle.dump((mol, mol.GetProp('_Name')), wpkl, -1)
                for c in mol.GetConformers():
                    w.write(mol, c.GetId())
    finally:
        w.close()


def create_db(db_fname):
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS mols")
        cur.execute("CREATE TABLE mols("
                    "id INTEGER NOT NULL, "
                    "conf_id INTEGER NOT NULL, "
                    "mol_block TEXT NOT NULL, "
                    "matched_ids TEXT NOT NULL, "
                    "visited_ids TEXT NOT NULL, "
                    "matched_ids_count INTEGER NOT NULL, "
                    "visited_ids_count INTEGER NOT NULL, "
                    "parent_mol_id INTEGER, "
                    "parent_conf_id INTEGER, "
                    "nmols INTEGER NOT NULL, "
                    "used INTEGER NOT NULL, "
                    "time TEXT NOT NULL)")
        conn.commit()


def merge_confs(mols_dict):
    # mols_dict - dict {parent_conf_id: [mol1, mol2, ...], ...}
    # molecules with identical smiles are combined into one with multiple conformers
    smiles = dict()  # {smi: Mol, ...}
    for parent_conf_id, mols in mols_dict.items():
        for mol in mols:
            visited_ids = mol.GetProp('visited_ids')
            for c in mol.GetConformers():
                # c.SetProp('parent_conf_id', str(parent_conf_id))
                c.SetProp('visited_ids', visited_ids)
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if smi not in smiles:
                smiles[smi] = Chem.Mol(mol)
            else:
                # atoms of additional conformers are renumbered according to already stored structure
                ids = smiles[smi].GetSubstructMatch(mol, useChirality=True)
                for c in mol.GetConformers():
                    pos = c.GetPositions()
                    for query_id, atom_id in enumerate(ids):
                        x, y, z = pos[query_id,]
                        c.SetAtomPosition(atom_id, Point3D(x, y, z))
                    smiles[smi].AddConformer(c, assignId=True)
    return smiles.values()


def save_res(mols, db_fname, parent_mol_id=None):

    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        cur.execute('SELECT MAX(id) FROM mols')
        mol_id = cur.fetchone()[0]
        if mol_id is None:
            mol_id = 0
        for mol in mols:
            mol_id += 1
            mol.SetProp('_Name', str(mol_id))
            for conf in mol.GetConformers():
                mol_block = Chem.MolToMolBlock(mol, confId=conf.GetId())
                visited_ids = conf.GetProp('visited_ids')
                visited_ids_count = visited_ids.count(',') + 1
                matched_ids = conf.GetProp('matched_ids')
                matched_ids_count = matched_ids.count(',') + 1
                parent_conf_id = conf.GetProp('parent_conf_id')
                if parent_conf_id == 'None':
                    parent_conf_id = None
                sql = 'INSERT INTO mols VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)'
                cur.execute(sql, (mol_id, conf.GetId(), mol_block, matched_ids, visited_ids, matched_ids_count,
                                  visited_ids_count, parent_mol_id, parent_conf_id, 0, 0))
        conn.commit()


def update_db(db_fname, mol_id, nmols):
    with sqlite3.connect(db_fname) as conn:
        while mol_id is not None:
            cur = conn.cursor()
            cur.execute('SELECT nmols FROM mols WHERE id = %i' % mol_id)
            n = cur.fetchone()[0] + nmols
            cur.execute('UPDATE mols SET nmols = %i WHERE id = %i' % (n, mol_id))
            conn.commit()
            cur.execute('SELECT parent_mol_id FROM mols WHERE id = %i' % mol_id)
            mol_id = cur.fetchone()[0]


def choose_mol_to_grow(db_fname, max_features, search_deep=True):

    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()

        if search_deep:
            cur.execute(f"""SELECT id, conf_id, mol_block, matched_ids, visited_ids 
                            FROM mols 
                            WHERE id = (
                              SELECT id
                              FROM mols
                              WHERE visited_ids_count < {max_features} AND used = 0
                              ORDER BY
                                visited_ids_count - matched_ids_count,
                                matched_ids_count DESC,
                                rowid DESC
                              LIMIT 1
                            )""")
        else:
            cur.execute(f"""SELECT id, conf_id, mol_block, matched_ids, visited_ids 
                            FROM mols 
                            WHERE id = (
                              SELECT c.id FROM (
                                SELECT DISTINCT 
                                  a.id, 
                                  a.matched_ids_count, 
                                  a.visited_ids_count, 
                                  ifnull(b.nmols, a.nmols) AS parent_nmols
                                FROM (SELECT * FROM mols WHERE visited_ids_count < {max_features} AND used = 0) AS a
                                LEFT JOIN mols b ON b.id = a.parent_mol_id
                                ORDER BY
                                  a.visited_ids_count,
                                  a.visited_ids_count - a.matched_ids_count,
                                  parent_nmols
                                LIMIT 1
                              ) AS c
                            )""")

        res = cur.fetchall()
        if not res:
            return None

        mol = Chem.MolFromMolBlock(res[0][2])
        mol.SetProp('_Name', str(res[0][0]))
        mol.SetProp('visited_ids', res[0][4])
        mol.GetConformer().SetId(res[0][1])
        mol.GetConformer().SetProp('matched_ids', res[0][3])
        for mol_id, conf_id, mol_block, matched_ids, visited_ids in res[1:]:
            m = Chem.MolFromMolBlock(mol_block)
            m.GetConformer().SetId(conf_id)
            m.GetConformer().SetProp('matched_ids', matched_ids)
            mol.AddConformer(m.GetConformer(), assignId=False)

        cur.execute('UPDATE mols SET used = 1 WHERE id = %i' % res[0][0])
        conn.commit()

        return mol


def remove_confs_exclvol(mol, exclvol_xyz, threshold):
    # if threshold < 0 (default -1) means ignore excl volumes
    if threshold >= 0 and exclvol_xyz is not None:
        cids = []
        ids = [atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()]
        for c in mol.GetConformers():
            d = cdist(c.GetPositions()[ids], exclvol_xyz)
            if (d < threshold).any():
                cids.append(c.GetId())
        if len(cids) == mol.GetNumConformers():
            return None
        for i in cids:
            mol.RemoveConformer(i)
    return mol


def get_pharm_xyz(pharm, ids=None):
    ids = pharm._get_ids(ids)
    coords = pharm.get_feature_coords(ids)
    df = pd.DataFrame([(i, label, *c) for i, (label, c) in zip(ids, coords)], columns=['id', 'label', 'x', 'y', 'z'])
    return df


def remove_confs_match(mol, pharm, matched_ids, new_ids, dist):

    remove_cids = []
    cids = [c.GetId() for c in mol.GetConformers()]
    plist = load_multi_conf_mol(mol)
    matched_xyz = get_pharm_xyz(pharm, matched_ids)
    new_xyz = get_pharm_xyz(pharm, new_ids)

    for cid, p in zip(cids, plist):

        conf_xyz = get_pharm_xyz(p)

        # match new pharmacophore features
        mask = np.array([i != j for i, j in product(conf_xyz['label'], new_xyz['label'])]).\
            reshape(conf_xyz.shape[0], new_xyz.shape[0])
        d = cdist(conf_xyz[['x', 'y', 'z']], new_xyz[['x', 'y', 'z']])
        d[mask] = dist + 1
        d = np.min(d, axis=0) <= dist
        new_matched_ids = new_xyz[d]['id'].tolist()
        if not new_matched_ids:
            remove_cids.append(cid)
            continue
        else:
            mol.GetConformer(cid).SetProp('matched_ids', ','.join(map(str, matched_ids + new_matched_ids)))
            mol.GetConformer(cid).SetIntProp('matched_ids_count', len(matched_ids) + len(new_matched_ids))

        # match previously matched pharmacophore features
        mask = np.array([i != j for i, j in product(conf_xyz['label'], matched_xyz['label'])]).\
            reshape(conf_xyz.shape[0], matched_xyz.shape[0])
        d = cdist(conf_xyz[['x', 'y', 'z']], matched_xyz[['x', 'y', 'z']])
        d[mask] = dist + 1
        if not (np.min(d, axis=0) <= dist).all():
            remove_cids.append(cid)
            continue

    if len(remove_cids) == mol.GetNumConformers():
        return None
    else:
        for cid in remove_cids:
            mol.RemoveConformer(cid)
        return mol


def get_confs(mol, template_conf_id, template_mol, nconfs, conf_alg, pharm, new_pids, dist, evol, seed):

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    mol = Chem.AddHs(mol)

    # start = time.process_time()
    try:
        mol = __gen_confs(mol, Chem.RemoveHs(template_mol), nconf=nconfs, alg=conf_alg, seed=seed,
                          coreConfId=template_conf_id, ignoreSmoothingFailures=False)
    except Exception as e:
        sys.stderr.write(f'the following error was occurred for in __gen_confs ' + str(e) + '\n')
        return template_conf_id, None

    if not mol:
        return template_conf_id, None

    # print(f'__gen_conf: {mol.GetNumConformers()} confs, {time.process_time() - start}')
    # start = time.process_time()

    mol = remove_confs_rms(mol)

    # print(f'remove_confs_rms: {mol.GetNumConformers()} confs, {time.process_time() - start}')
    # start = time.process_time()

    mol = remove_confs_exclvol(mol, pharm.exclvol, evol)
    if not mol:
        return template_conf_id, None

    # print(f'remove_confs_exclvol: {mol.GetNumConformers()} confs, {time.process_time() - start}')
    # start = time.process_time()

    mol = remove_confs_match(mol,
                             pharm=pharm,
                             matched_ids=list(map(int, template_mol.GetConformer(template_conf_id).GetProp('matched_ids').split(','))),
                             new_ids=new_pids,
                             dist=dist)

    # print(f'remove_confs_match: {mol.GetNumConformers() if mol else None} confs, {time.process_time() - start}')

    if not mol:
        return template_conf_id, None

    mol.SetProp('visited_ids', template_mol.GetProp('visited_ids') + ',' + ','.join(map(str, new_pids)))
    for conf in mol.GetConformers():
        conf.SetProp('parent_conf_id', str(template_conf_id))
    return template_conf_id, mol


def get_features(p, pids, add_features):
    if add_features:
        c = Counter(label for label, xyz in p.get_feature_coords(pids))
        output = dict()
        for k, v in c.items():
            if k == 'a':
                output['nAr'] = (v, 10)
            else:
                output[f'n{k}'] = (v, 10)
        return output
    else:
        return dict()


def filter_by_hashes(row_ids, cur, radius, db_hashes, hashes):
    """

    :param row_ids: selected row_ids which should be filtered (necessary parameter)
    :param cur: SQLite cursor object of CReM DB (necessary parameter)
    :param radius: radius in CReM DB (necessary parameter)
    :param db_hashes: file name of DB of 3D pharmacophore hashes
    :param hashes: list of 3D pharmacophore hashes to keep fragments
    :return:
    """

    if not row_ids:
        return []
    batch_size = 32000
    row_ids = list(row_ids)
    smis = defaultdict(list)
    for start in range(0, len(row_ids), batch_size):
        batch = row_ids[start:start + batch_size]
        sql = f"SELECT rowid, core_smi FROM radius{radius} WHERE rowid IN ({','.join('?' * len(batch))})"
        for i, smi in cur.execute(sql, batch).fetchall():
            smis[smi].append(i)

    con = sqlite3.connect(db_hashes)
    sql = f"SELECT DISTINCT(frags.smi) FROM frags, hashes WHERE frags.id == hashes.id AND hashes.hash IN ({','.join('?' * len(hashes))})"
    res = [item[0] for item in con.execute(sql, list(hashes)).fetchall()]

    output_row_ids = []
    for smi in res:
        output_row_ids.extend(smis[smi])
    return output_row_ids


def enumerate_hashes(att_positions, feature_positions, bin_step, max_features=5):
    output_hashes = set()
    for pos in att_positions:
        p = P(cached=True, bin_step=bin_step)
        p.load_from_feature_coords(feature_positions + [('T', tuple(pos))])
        # feature_ids = p.get_feature_ids()
        # att_id = feature_ids['T'][0]
        # other_ids = []
        # for k, v in feature_ids.items():
        #     if k != 'T':
        #         other_ids.extend(v)
        # max_features = min(max_features, len(other_ids))
        # hashes = []
        # for n in range(1, max_features + 1):
        #     for comb in combinations(other_ids, n):
        #         hashes.append(p.get_signature_md5(ids=[att_id] + list(comb)))
        # output_hashes.update(hashes)
        output_hashes.add(p.get_signature_md5())
    return output_hashes


def enumerate_hashes_directed(mol, att_ids, feature_positions, bin_step, directed):

    output_hashes = set()

    for conf in mol.GetConformers():
        for att_id in att_ids:
            t_pos = conf.GetAtomPosition(att_id)

            if directed:
                for end_atom in mol.GetAtomWithIdx(att_id).GetNeighbors():
                    if end_atom.GetAtomicNum() == 1:
                        end_atom_id = end_atom.GetIdx()
                        h_pos = conf.GetAtomPosition(end_atom_id)
                        q_pos = t_pos + (h_pos - t_pos) / np.linalg.norm(t_pos - h_pos) * bin_step
                        p = P(cached=True, bin_step=bin_step)
                        p.load_from_feature_coords(feature_positions + [('T', tuple(t_pos)), ('Q', tuple(q_pos))])
                        output_hashes.add(p.get_signature_md5())

            else:
                p = P(cached=True, bin_step=bin_step)
                p.load_from_feature_coords(feature_positions + [('T', tuple(t_pos))])
                output_hashes.add(p.get_signature_md5())

    return output_hashes


def main():
    parser = argparse.ArgumentParser(description='Grow structures to fit query pharmacophore.')
    parser.add_argument('-q', '--query', metavar='FILENAME', required=True,
                        help='pharmacophore model.')
    parser.add_argument('--ids', metavar='INTEGER', required=True, nargs='+', type=int,
                        help='ids of pharmacophore features used for initial screening. 0-index based.')
    parser.add_argument('-o', '--output', metavar='DIRNAME', required=True,
                        help='path to directory where intermediate and final results will be stored. '
                             'If output db file (res.db) exists in the directory the computation will be continued '
                             '(skip screening of initial fragment DB).')
    parser.add_argument('-t', '--clustering_threshold', metavar='NUMERIC', required=False, type=float, default=3,
                        help='threshold to determine clusters. Default: 3.')
    parser.add_argument('-f', '--fragments', metavar='FILENAME', required=True,
                        help='file with initial fragments - DB in pmapper format.')
    parser.add_argument('-d', '--db', metavar='FILENAME', required=True,
                        help='database with interchangeable fragments.')
    parser.add_argument('-r', '--radius', metavar='INTEGER', type=int, choices=[1, 2, 3, 4, 5], default=3,
                        help='radius of a context of attached fragments.')
    parser.add_argument('--max_replacements', metavar='INTEGER', type=int, default=None,
                        help='maximum number of fragments considered for growing. By default all fragments are '
                             'considered, that may cause combinatorial explosion in some cases.')
    parser.add_argument('-x', '--additional_features', action='store_true', default=False,
                        help='indicate if the fragment database contains pharmacophore features to be used for '
                             'fragment selection.')
    parser.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=20,
                        help='number of conformers generated per structure. Default: 20.')
    parser.add_argument('--conf_gen', metavar='STRING', required=False, type=str, default='rdkit',
                        help='can take "rdkit" or "ob" values to choose conformer generator. Default: rdkit.')
    parser.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=-1,
                        help='seed for random number generator to get reproducible output. Default: -1.')
    parser.add_argument('--dist', metavar='NUMERIC', required=False, type=float, default=1,
                        help='maximum distance to discard conformers in fast filtering. Default: 1.')
    parser.add_argument('-e', '--exclusion_volume', metavar='NUMERIC', required=False, type=float, default=-1,
                        help='radius of exclusion volumes (distance to heavy atoms). By default exclusion volumes are '
                             'disabled even if they are present in a query pharmacophore. To enable them set '
                             'a positive numeric value.')
    parser.add_argument('--hash_db', metavar='FILENAME', required=False, default=None,
                        help='database with 3D pharmacophore hashes for additional filtering of fragments for growing.')
    parser.add_argument('--hash_db_bin_step', metavar='NUMERIC', required=False, default=1.5, type=float,
                        help='bin step used to create 3D pharmacophore hashes.')
    parser.add_argument('--hash_db_directed', required=False, default=False, action='store_true',
                        help='if set direction of attachment points will be considered during calculation of '
                             '3D pharmacophore hashes.')
    parser.add_argument('-u', '--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    parser.add_argument('-w', '--num_workers', metavar='INTEGER', required=False, type=int, default=100,
                        help='the number of workers to be spawn by the dask cluster. For efficiency it seems that '
                             'it should be equal to the total number of cores requested by the cluster. Default: 100.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use. Default: 1.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    if args.hostfile is not None:
        dask_client = Client(open(args.hostfile).readline().strip() + ':8786')
    pool = Pool(args.ncpu)

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'setup.json'), 'wt') as f:
        f.write(json.dumps(vars(args), indent=4))

    shutil.copyfile(args.query, os.path.join(args.output, os.path.basename(args.query)))

    args.ids = tuple(sorted(set(args.ids)))

    p = PharmModel2()
    p.load_from_xyz(args.query)
    p.set_clusters(args.clustering_threshold, args.ids)

    print(p.clusters)

    res_db_fname = os.path.join(args.output, 'res.db')

    if os.path.isfile(res_db_fname):  # restart if db existed
        mol = choose_mol_to_grow(res_db_fname, p.get_num_features())

    else:
        create_db(res_db_fname)

        conf_fname = os.path.join(args.output, f'iter0.sdf')

        new_pids = tuple(args.ids)

        print(f"===== Initial screening =====")
        start = time.perf_counter()

        flag = screen_pmapper(query_pharm=p.get_subpharmacophore(new_pids), db_fname=args.fragments,
                              output_sdf=conf_fname, rmsd=0.2, ncpu=args.ncpu)
        if not flag:
            exit('No matches between starting fragments and the chosen subpharmacophore.')

        print(f'{round(time.perf_counter() - start, 4)}')

        mols = select_mols([mol for mol, mol_name in read_input(conf_fname, sdf_confs=True)])
        mols = [remove_confs_exclvol(mol, p.exclvol, args.exclusion_volume) for mol in mols]
        mols = [m for m in mols if m]
        ids = ','.join(map(str, new_pids))
        for mol in mols:
            mol.SetProp('visited_ids', ids)
            for conf in mol.GetConformers():
                conf.SetProp('matched_ids', ids)
        mols = merge_confs({None: mols})   # return list of mols
        mols = [remove_confs_rms(m) for m in mols]
        save_res(mols, res_db_fname)

        mol = choose_mol_to_grow(res_db_fname, p.get_num_features())

    while mol:

        print(f"===== {mol.GetProp('_Name')} =====")
        start = time.perf_counter()

        new_pids = p.select_nearest_cluster(tuple(map(int, mol.GetProp('visited_ids').split(','))))
        atom_ids = get_grow_atom_ids(mol, p.get_xyz(new_pids))
        kwargs = get_features(p, new_pids, args.additional_features)

        # create additional constrains for selection of fragments which will be attached during growing
        __max_features = 5   # max number of enumerated feature combinations in 3D pharm hash db
        use_hash_db = args.hash_db is not None and len(new_pids) <= __max_features
        hashes = []
        if use_hash_db:
            feature_positions = p.get_feature_coords(new_pids)
            hashes = enumerate_hashes_directed(mol=mol, att_ids=atom_ids, feature_positions=feature_positions,
                                               bin_step=args.hash_db_bin_step, directed=args.hash_db_directed)

        new_mols = list(grow_mol(mol, args.db, radius=args.radius, min_atoms=1, max_atoms=12,
                                 max_replacements=args.max_replacements, replace_ids=atom_ids, return_mol=True,
                                 ncores=args.ncpu,
                                 filter_func=partial(filter_by_hashes, db_hashes=args.hash_db, hashes=hashes) if use_hash_db else None,
                                 **kwargs))

        print(f'mol grow: {len(new_mols)} mols, {round(time.perf_counter() - start, 4)}')
        sys.stdout.flush()
        start2 = time.perf_counter()

        new_isomers = []
        for isomers in pool.imap_unordered(gen_stereo, (m[1] for m in new_mols)):
            new_isomers.extend(isomers)

        print(f'stereo enumeration: {len(new_isomers)} isomers, {round(time.perf_counter() - start2, 4)}')
        sys.stdout.flush()
        start2 = time.perf_counter()

        new_mols = defaultdict(list)   # {parent_conf_id_1: [mol1, mol2, ...], ... }
        inputs = [(new_isomer, conf.GetId()) for new_isomer, conf in product(new_isomers, mol.GetConformers())]

        if args.hostfile is not None:
            b = bag.from_sequence(inputs, npartitions=args.num_workers * 2)
            for conf_id, m in b.starmap(get_confs, template_mol=mol,
                                                   nconfs=args.nconf,
                                                   conf_alg=args.conf_gen,
                                                   pharm=p,
                                                   new_pids=new_pids,
                                                   dist=args.dist,
                                                   evol=args.exclusion_volume,
                                                   seed=args.seed).compute():
                if m:
                    new_mols[conf_id].append(m)
        else:
            for conf_id, m in pool.starmap(partial(get_confs, template_mol=mol,
                                                              nconfs=args.nconf,
                                                              conf_alg=args.conf_gen,
                                                              pharm=p,
                                                              new_pids=new_pids,
                                                              dist=args.dist,
                                                              evol=args.exclusion_volume,
                                                              seed=args.seed), inputs):
                if m:
                    new_mols[conf_id].append(m)

        print(f'conf generation: {sum(len(v) for v in new_mols.values())} molecules, {round(time.perf_counter() - start2, 4)}')
        start2 = time.perf_counter()

        # keep only conformers with maximum number of matched features
        max_count = 0
        for v in new_mols.values():
            for m in v:
                for c in m.GetConformers():
                    if c.GetIntProp('matched_ids_count') > max_count:
                        max_count = c.GetIntProp('matched_ids_count')

        for v in new_mols.values():
            for i in reversed(range(len(v))):
                cids = []
                for c in v[i].GetConformers():
                    if c.GetIntProp('matched_ids_count') < max_count:
                        cids.append(c.GetId())
                if len(cids) == v[i].GetNumConformers():
                    del v[i]
                else:
                    for cid in cids:
                        v[i].RemoveConformer(cid)

        for conf_id in new_mols:
            new_mols[conf_id] = select_mols(new_mols[conf_id])

        print(f'conf filtering and mol selection: {sum(len(v) for v in new_mols.values())} compounds, {round(time.perf_counter() - start2, 4)}')

        new_mols = merge_confs(new_mols)   # return list of mols
        new_mols = [remove_confs_rms(m) for m in new_mols]

        parent_mol_id = int(mol.GetProp('_Name'))
        save_res(new_mols, res_db_fname, parent_mol_id=parent_mol_id)

        update_db(res_db_fname, parent_mol_id, len(new_isomers))

        print('saved mols:', len(new_mols))

        search_deep = True
        if mol.GetProp('visited_ids').count(',') + 1 == p.get_num_features() or not new_mols:
            search_deep = False
        print('search deep: ', search_deep)
        mol = choose_mol_to_grow(res_db_fname, p.get_num_features(), search_deep=search_deep)

        print(f'overall time {round(time.perf_counter() - start, 4)}')

        sys.stdout.flush()


if __name__ == '__main__':
    main()
