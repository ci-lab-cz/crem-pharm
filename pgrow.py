#!/usr/bin/env python3

import argparse
import os
import sys
import pickle
import operator
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
from itertools import combinations, product
from operator import itemgetter
import numpy as np
from numpy.linalg import norm
import pandas as pd
import sqlite3
import time

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.AllChem import AlignMol, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from crem.crem import grow_mol
from pmapper.utils import load_multi_conf_mol
from pmapper.pharmacophore import Pharmacophore as P
from psearch.database import DB
from read_input import read_input


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


def gen_stereo(mol):
    stereo_opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    for b in mol.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOANY:
            b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    isomers = tuple(EnumerateStereoisomers(mol, options=stereo_opts))
    return isomers


def ConstrainedEmbedMultipleConfs(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342,
                                  getForceField=UFFGetMoleculeForceField, **kwargs):

    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    cids = EmbedMultipleConfs(mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    cids = list(cids)
    if len(cids) == 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        for cid in cids:
            ff = getForceField(mol, confId=cid)
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
            rms = AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        for cid in cids:
            rms = AlignMol(mol, core, prbCid=cid, atomMap=algMap)
            ff = getForceField(mol, confId=cid)
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
            rms = AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol


def __gen_confs(mol, template_mol=None, nconf=10, seed=42, **kwargs):
    try:
        if template_mol:
            mol = ConstrainedEmbedMultipleConfs(Chem.AddHs(mol), Chem.RemoveHs(template_mol), nconf, randomseed=seed, **kwargs)
        else:
            AllChem.EmbedMultipleConfs(Chem.AddHs(mol), numConfs=nconf, maxAttempts=nconf*4, randomSeed=seed)
    except ValueError:
        return None
    return mol


def sdf_not_empty(fname):
    try:
        for m in Chem.SDMolSupplier(fname):
            if m:
                return True
        return False
    except OSError:
        return False


def get_grow_points2(mol_xyz, pharm_xyz, tol=2):
    dist = np.min(cdist(mol_xyz, pharm_xyz), axis=1)
    ids = np.flatnonzero(dist <= (np.min(dist) + tol))
    return ids.tolist()


def get_grow_atom_ids(mol, pharm_xyz, tol=2):
    ids = set()
    for c in mol.GetConformers():
        ids.update(get_grow_points2(c.GetPositions(), pharm_xyz, tol))
    return list(sorted(ids))


def remove_confs_rms(mol, rms=0.5):

    remove_ids = []
    cids = [c.GetId() for c in mol.GetConformers()]

    rms_list = [(i1, i2, norm(mol.GetConformer(i1).GetPositions() - mol.GetConformer(i2).GetPositions()))
                for i1, i2 in combinations(cids, 2)]
    while any(item[2] < rms for item in rms_list):
        for item in rms_list:
            if item[2] < rms:
                remove_ids.append(item[1])
                rms_list = [i for i in rms_list if i[0] != item[1] and i[1] != item[1]]
                break

    for cid in set(remove_ids):
        mol.RemoveConformer(cid)

    return mol


def remove_conf(mol, cids):
    for cid in set(cids):
        mol.RemoveConformer(cid)
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)
    return mol


def keep_confs(mol, ids):

    all_ids = set(c.GetId() for c in mol.GetConformers())
    remove_ids = all_ids - set(ids)
    for cid in set(remove_ids):
        mol.RemoveConformer(cid)
    # conformers are reindexed staring with 0 step 1
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)
    return mol


def screen_pmapper(query_pharm, db_fname, output_sdf, rmsd=0.2):

    matches = False

    query_mol = query_pharm.get_mol()
    query_nfeatures = query_mol.GetNumAtoms()

    db = DB(db_fname)

    w = Chem.SDWriter(output_sdf)
    wpkl = open(output_sdf.rsplit('.', 1)[0] + '.pkl', 'wb')

    mol_names = db.get_mol_names()

    for j, mol_name in enumerate(mol_names):

        pharm = db.get_pharm(mol_name)[0]

        rmsd_dict = dict()
        for i, coords in enumerate(pharm):
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
                rmsd_dict[i] = AllChem.GetAlignmentTransform(Chem.Mol(m1), query_mol, atomMap=tuple(zip(min_ids, range(query_nfeatures))))

        if rmsd_dict:
            matches = True
            m = db.get_mol(mol_name)[0]
            m.SetProp('_Name', mol_name)
            for k, (rms, matrix) in rmsd_dict.items():
                AllChem.TransformMol(m, matrix, k, keepConfs=True)
                m.SetProp('rms', str(rms))
                w.write(m, k)
            m = keep_confs(m, rmsd_dict.keys())
            pickle.dump((m, mol_name), wpkl, -1)

    return matches


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
    mols = [(mol, mol.GetNumHeavyAtoms()) for mol in mols]
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
        cur.execute("DROP TABLE IF EXISTS res")
        cur.execute("CREATE TABLE res("
                    "mol_id INTEGER NOT NULL, "
                    "conf_id INTEGER NOT NULL, "
                    "mol_block TEXT NOT NULL, "
                    "matched_ids TEXT NOT NULL, "
                    "visited_ids TEXT NOT NULL, "
                    "matched_ids_count INTEGER NOT NULL, "
                    "visited_ids_count INTEGER NOT NULL, "
                    "parent_mol_id INTEGER, "
                    "parent_conf_id INTEGER, "
                    "used INTEGER NOT NULL)")
        conn.commit()


def save_res(mols, db_fname, parent_mol_id=None, parent_conf_id=None):
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        cur.execute('SELECT MAX(mol_id) FROM res')
        mol_id = cur.fetchone()[0]
        if mol_id is None:
            mol_id = 0
        for mol in mols:
            mol_id += 1
            visited_ids = mol.GetProp('visited_ids')
            visited_ids_count = visited_ids.count(',') + 1
            for conf_id, conf in enumerate(mol.GetConformers()):
                mol_block = Chem.MolToMolBlock(mol, confId=conf.GetId())
                matched_ids = conf.GetProp('matched_ids')
                matched_ids_count = matched_ids.count(',') + 1
                sql = 'INSERT INTO res VALUES (?,?,?,?,?,?,?,?,?,?)'
                cur.execute(sql, (mol_id, conf_id, mol_block, matched_ids, visited_ids, matched_ids_count,
                                  visited_ids_count, parent_mol_id, parent_conf_id, 0))
        conn.commit()


def choose_mol_to_grow(db_fname, max_features):
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        cur.execute("""SELECT mol_id, conf_id, mol_block, matched_ids, visited_ids 
                       FROM res 
                       WHERE mol_id = (
                         SELECT mol_id
                         FROM res
                         WHERE visited_ids_count < %s AND used = 0
                         ORDER BY
                           visited_ids_count - matched_ids_count,
                           matched_ids_count DESC
                         LIMIT 1
                       )""" % max_features)
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

        cur.execute('UPDATE res SET used = 1 WHERE mol_id = %i' % res[0][0])
        conn.commit()

        return mol


def remove_confs_exclvol(mol, exclvol_xyz, threshold=2):
    if exclvol_xyz:
        cids = []
        for c in mol.GetConformers():
            d = cdist(c.GetPositions(), exclvol_xyz)
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
        d = np.min(d <= dist, axis=0)
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
        if not np.min(d <= dist, axis=0).all():
            remove_cids.append(cid)
            continue

    if len(remove_cids) == mol.GetNumConformers():
        return None
    else:
        for cid in remove_cids:
            mol.RemoveConformer(cid)
        return mol


def get_confs(mol, template_mol, template_conf_id, nconfs, pharm, new_pids, dist, seed):

    # start = time.process_time()
    try:
        mol = __gen_confs(Chem.AddHs(mol), Chem.RemoveHs(template_mol), nconf=nconfs, seed=seed, coreConfId=template_conf_id)
    except Exception as e:
        sys.stderr.write(f'the following error was occurred for in __gen_confs ' + str(e) + '\n')
        return None

    if not mol:
        return None

    # print(f'__gen_conf: {mol.GetNumConformers()} confs, {time.process_time() - start}')
    # start = time.process_time()

    mol = remove_confs_rms(mol, rms=dist)

    # print(f'remove_confs_rms: {mol.GetNumConformers()} confs, {time.process_time() - start}')
    # start = time.process_time()

    mol = remove_confs_exclvol(mol, pharm.exclvol)
    if not mol:
        return None

    # print(f'remove_confs_exclvol: {mol.GetNumConformers()} confs, {time.process_time() - start}')
    # start = time.process_time()

    mol = remove_confs_match(mol,
                             pharm=pharm,
                             matched_ids=tuple(map(int, template_mol.GetConformer(template_conf_id).GetProp('matched_ids').split(','))),
                             new_ids=new_pids,
                             dist=dist)

    # print(f'remove_confs_match: {mol.GetNumConformers() if mol else None} confs, {time.process_time() - start}')

    if not mol:
        return None

    mol.SetProp('visited_ids', template_mol.GetProp('visited_ids') + ',' + ','.join(map(str, new_pids)))
    return mol


def main():
    parser = argparse.ArgumentParser(description='Grow structures to fit query pharmacophore.')
    parser.add_argument('-q', '--query', metavar='FILENAME', required=True,
                        help='pharmacophore model.')
    parser.add_argument('--ids', metavar='INTEGER', required=True, nargs='+', type=int,
                        help='ids of pharmacophore features used for initial screening.')
    parser.add_argument('-o', '--output', metavar='DIRNAME', required=True,
                        help='path to directory where intermediate and final results will be stored.')
    parser.add_argument('-t', '--clustering_threshold', metavar='NUMERIC', required=False, type=float, default=3,
                        help='threshold to determine clusters. Default: 3.')
    parser.add_argument('-f', '--fragments', metavar='FILENAME', required=True,
                        help='file with initial fragments. If extension is SMI - conformers will be generated. '
                             'If SMI file contains fragment names they should be tab-separated. If extension is SDF - '
                             'the file should contain conformers having identical names for the same compounds. '
                             'This can also be pharmit database with precomputed conformers.')
    parser.add_argument('-d', '--db', metavar='FILENAME', required=True,
                        help='database with interchangeable fragments.')
    parser.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=20,
                        help='number of conformers generated per structure. Default: 20.')
    parser.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=-1,
                        help='seed for random number generator to get reproducible output. Default: -1.')
    parser.add_argument('--dist', metavar='NUMERIC', required=False, type=float, default=1,
                        help='maximum distance to discard conformers in fast filtering. Default: 1.')
    parser.add_argument('-p', '--pharmit', metavar='pharmit executable', required=False, type=str,
                        default='/home/pavel/pharmit/pharmit/src/build/pharmit',
                        help='path to pharmit executable.')
    parser.add_argument('-a', '--pharmit_spec', metavar='FILENAME', required=False, type=str, default=None,
                        help='path to file with pharmacophore specifications in pharmit format.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use. Default: 1.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    pool = Pool(args.ncpu)

    args.ids = tuple(sorted(set(args.ids)))

    res_db_fname = os.path.join(args.output, 'res.db')
    create_db(res_db_fname)

    p = PharmModel2()
    p.load_from_xyz(args.query)
    p.set_clusters(args.clustering_threshold, args.ids)

    print(p.clusters)

    conf_fname = os.path.join(args.output, f'iter0.sdf')
    conf_fname_pkl = os.path.splitext(conf_fname)[0] + '.pkl'

    new_pids = tuple(args.ids)

    flag = screen_pmapper(query_pharm=p.get_subpharmacophore(new_pids), db_fname=args.fragments, output_sdf=conf_fname)
    if not flag:
        exit('No matches between starting fragments and the chosen subpharmacophore.')

    mols = select_mols([mol for mol, mol_name in read_input(conf_fname_pkl)])
    mols = [remove_confs_exclvol(mol, p.exclvol) for mol in mols]
    mols = [m for m in mols if m]
    ids = ','.join(map(str, new_pids))
    for mol in mols:
        mol.SetProp('visited_ids', ids)
        for conf in mol.GetConformers():
            conf.SetProp('matched_ids', ids)
    save_res(mols, res_db_fname)

    mol = choose_mol_to_grow(res_db_fname, p.get_num_features())

    while mol:

        print(f"===== {mol.GetProp('_Name')} =====")
        start = time.perf_counter()

        new_pids = p.select_nearest_cluster(tuple(map(int, mol.GetProp('visited_ids').split(','))))
        atom_ids = get_grow_atom_ids(mol, p.get_xyz(new_pids))
        new_mols = list(grow_mol(mol, args.db, radius=2, min_atoms=1, max_atoms=12,
                                 max_replacements=None, replace_ids=atom_ids, return_mol=True,
                                 ncores=1))

        print(f'mol grow: {len(new_mols)} mols, {time.perf_counter() - start}')
        start2 = time.perf_counter()

        new_isomers = []
        for isomers in pool.imap_unordered(gen_stereo, (m[1] for m in new_mols)):
            new_isomers.extend(isomers)

        print(f'stereo enumeration: {len(new_isomers)} isomers, {time.perf_counter() - start2}')
        start2 = time.perf_counter()

        new_mols = defaultdict(list)
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            for new_mol in pool.imap_unordered(partial(get_confs,
                                                       template_mol=mol,
                                                       template_conf_id=conf_id,
                                                       nconfs=args.nconf,
                                                       pharm=p,
                                                       new_pids=new_pids,
                                                       dist=args.dist,
                                                       seed=args.seed),
                                               new_isomers):
                if new_mol:
                    new_mols[conf_id].append(new_mol)

        print(f'conf generation: {sum(len(v) for v in new_mols.values())} confs, {time.perf_counter() - start2}')
        start2 = time.perf_counter()

        max_count = 0
        for v in new_mols.values():
            for m in v:
                for c in m:
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

        print(f'conf filtering and selection: {sum(len(v) for v in new_mols.values())} confs, {time.perf_counter() - start2}')

        parent_mol_id = mol.GetProp('_Name')
        for conf_id, mols in new_mols.items():
            save_res(mol, res_db_fname, parent_mol_id=parent_mol_id, parent_conf_id=conf_id)

        mol = choose_mol_to_grow(res_db_fname, p.get_num_features())

        print('selected mols:', sum(len(v) for v in new_mols.values()))
        print(f'overall time {time.perf_counter() - start}')


if __name__ == '__main__':
    main()
