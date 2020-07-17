#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import shutil
import pickle
import random
import operator
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
from itertools import combinations
from operator import itemgetter
import numpy as np
from numpy.linalg import norm
import pandas as pd
import shelve

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.AllChem import AlignMol, EmbedMolecule, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from crem.crem import grow_mol, grow_mol2
# from openbabel import openbabel as ob
# from openbabel import pybel
from pmapper.utils import load_multi_conf_mol
from pmapper.pharmacophore import Pharmacophore as P
from psearch.database import DB
from read_input import read_input


class PharmModel:

    __replace_names = {'Aromatic': 'a', 'HydrogenDonor': 'D', 'HydrogenAcceptor': 'A',
                       'Hydrophobic': 'H', 'PositiveIon': 'P', 'NegativeIon': 'N'}

    def __init__(self, fname):
        p = json.load(open(fname))['points']
        self.p = [i for i in p if i['enabled']]
        self.clusters = defaultdict(set)

    def __get_ids(self, ids=None):
        if ids:
            return tuple(sorted(set(ids)))
        else:
            return tuple(range(len(self.p)))

    def get_feature_coords(self, ids=None):
        ids = self.__get_ids(ids)
        coords = [(i, PharmModel.__replace_names[self.p[i]['name']], self.p[i]['x'], self.p[i]['y'], self.p[i]['z']) for i in ids]
        coords = pd.DataFrame(coords, columns=['id', 'label', 'x', 'y', 'z'])
        # coords.sort_values(by='label', axis=1, inplace=True)
        # coords.set_index(keys=['label'], drop=False, inplace=True)
        return coords

    def xyz(self, ids=None, cluster=None):
        """

        :param ids: list of feature ids or None (all features)
        :param cluster: number of a cluster, if not None it has higher precedence than ids
        :return: numpy array of shape (N, 3)
        """
        if cluster:
            ids = self.clusters[cluster]
        else:
            ids = self.__get_ids(ids)
        m = np.array([(self.p[i]['x'], self.p[i]['y'], self.p[i]['z']) for i in ids])
        return m

    def set_clusters(self, clustering_threshold, ids=None):
        ids = tuple(set(self.__get_ids()) - set(ids))
        c = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_threshold).fit(self.xyz(ids))
        for i, j in enumerate(c.labels_):
            self.clusters[j].add(ids[i])

    def select_nearest_cluster(self, ids):

        def min_distance(ids1, ids2):
            xyz1 = self.xyz(ids1)
            xyz2 = self.xyz(ids2)
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

    def write_model(self, fname, ids=None):
        ids = self.__get_ids(ids)
        p = [self.p[i] for i in ids]
        p = {'points': p}
        json.dump(p, open(fname, 'wt'))

    def write_model_xyz(self, fname, ids=None):
        ids = self.__get_ids(ids)
        with open(fname, 'wt') as f:
            f.write('\n\n')
            for i in ids:
                f.write(' '.join(map(str, (self.p[i]['name'], self.p[i]['x'], self.p[i]['y'], self.p[i]['z']))) + '\n')


class PharmModel2(P):

    def __init__(self, bin_step=1, cached=False):
        super().__init__(bin_step, cached)
        self.clusters = defaultdict(set)

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



def gen_stereo(mol):
    stereo_opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    for b in mol.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOANY:
            b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    isomers = tuple(EnumerateStereoisomers(mol, options=stereo_opts))
    return isomers


def load_frags(fname):
    res = []
    with open(fname) as f:
        for i, line in enumerate(f):
            items = line.strip().split()
            if items:
                mol = Chem.MolFromSmiles(items[0])
                if mol:
                    if len(items) == 1:
                        name = f'mol-{i}'
                    else:
                        name = items[1]
                    mol.SetProp('_Name', name)
                    res.append(mol)
    return res


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


def __embed_multiple_confs_constrained_rdkit(mol, template_mol, nconf, seed=42):

    new_confs = []
    template_mol = Chem.RemoveHs(template_mol)
    random.seed(seed)
    seeds = random.sample(range(100000), nconf)
    for i in seeds:
        q = Chem.Mol(mol)
        new_confs.append(AllChem.ConstrainedEmbed(q, template_mol, randomseed=i))
    out_mol = new_confs[0]
    for a in new_confs[1:]:
        out_mol.AddConformer(a.GetConformer(0), assignId=True)
    return out_mol


# def __embed_multiple_confs_constrained(mol, template_mol, nconf, seed=42):
#     mol_rdkit = AllChem.ConstrainedEmbed(Chem.AddHs(mol), template_mol, randomseed=seed)
#     mol_rdkit = Chem.RemoveHs(mol_rdkit)
#     ids = [i + 1 for i in mol_rdkit.GetSubstructMatch(template_mol)]   # ids of atoms to fix
#
#     mol = pybel.readstring('mol', Chem.MolToMolBlock(mol_rdkit)).OBMol   # convert mol from RDKit to OB
#
#     pff = ob.OBForceField_FindType("mmff94")
#     if not pff.Setup(mol):  # if OB FF setup fails use RDKit conformer generation (slower)
#         return __embed_multiple_confs_constrained_rdkit(mol_rdkit, template_mol, nconf, seed)
#
#     constraints = ob.OBFFConstraints()
#     for atom in ob.OBMolAtomIter(mol):
#         atom_id = atom.GetIndex() + 1
#         if atom_id in ids:
#             constraints.AddAtomConstraint(atom_id)
#     pff.SetConstraints(constraints)
#
#     pff.DiverseConfGen(0.5, 1000, 50, False)   # rmsd, nconf_tries, energy, verbose
#
#     pff.GetConformers(mol)
#     obconversion = ob.OBConversion()
#     obconversion.SetOutFormat('mol')
#
#     output_strings = []
#     for conf_num in range(max(0, mol.NumConformers() - nconf), mol.NumConformers()):   # save last nconf conformers (it seems the last one is the original conformer)
#         mol.SetConformer(conf_num)
#         output_strings.append(obconversion.WriteString(mol))
#
#     out_mol = Chem.MolFromMolBlock(output_strings[0])
#     for a in output_strings[1:]:
#         out_mol.AddConformer(Chem.MolFromMolBlock(a).GetConformer(0), assignId=True)
#     return out_mol


def __gen_confs(mol, template_mol=None, nconf=10, seed=42, **kwargs):
    try:
        if template_mol:
            mol = ConstrainedEmbedMultipleConfs(Chem.AddHs(mol), Chem.RemoveHs(template_mol), nconf, randomseed=seed, **kwargs)
        else:
            AllChem.EmbedMultipleConfs(Chem.AddHs(mol), numConfs=nconf, maxAttempts=nconf*4, randomSeed=seed)
    except ValueError:
        return None
    return mol


def append_generated_confs_file(mols, fname, template_mol=None, nconf=10, seed=42, pool=None):
    """
    Generate conformers and append them to sdf file
    :param mols: list of mols
    :param fname: sdf file name to save conformers
    :param template_mol: RDKit Mol if constrained embedding should be done
    :param nconf: number of generated conformers
    :param seed: random seed to embed conformers
    :param pool: multiprocessing pool to use multiple CPUs or None to use a single CPU
    :return:
    """
    with open(fname, 'a') as f:
        w = Chem.SDWriter(f)
        if pool:
            for mol in pool.imap_unordered(partial(__gen_confs, template_mol=template_mol, nconf=nconf, seed=seed), mols):
                if mol:
                    for c in mol.GetConformers():
                        w.write(mol, c.GetId())
        else:
            for mol in mols:
                mol = __gen_confs(mol, template_mol=template_mol, nconf=nconf, seed=seed)
                if mol:
                    for c in mol.GetConformers():
                        w.write(mol, c.GetId())
        w.close()


def create_db(fname, dbname, pharmit_path, pharmit_spec, ncpu=1):
    a = [pharmit_path, 'dbcreate', '-dbdir', dbname, '-in', fname, '-nthreads', str(ncpu)]
    if pharmit_spec is not None:
        a.extend(['-pharmaspec', pharmit_spec])
    subprocess.run(a)


def search_db(dbname, query_fname, out_fname, pharmit_path, ncpu=1):
    subprocess.run([pharmit_path, 'dbsearch', '-dbdir', dbname, '-in', query_fname, '-out', out_fname,
                    '-reduceconfs', '1', '-sort-rmsd', '-nthreads', str(ncpu)])


def sdf_not_empty(fname):
    try:
        for m in Chem.SDMolSupplier(fname):
            if m:
                return True
        return False
    except OSError:
        return False


def select_compounds(fname):
    mols = []
    for m in Chem.SDMolSupplier(fname):
        if m:
            mols.append((m.GetNumHeavyAtoms(), m))
    min_hac = min((item[0] for item in mols))
    mols = [item[1] for item in mols if item[0] <= min_hac + 2]
    return mols


def get_grow_points(mol, pharm_xyz, tol=2):
    mol_xyz = mol.GetConformer().GetPositions()
    dist = np.min(cdist(mol_xyz, pharm_xyz), axis=1)
    ids = np.flatnonzero(dist <= (np.min(dist) + tol))
    return ids.tolist()


def get_grow_points2(mol_xyz, pharm_xyz, tol=2):
    dist = np.min(cdist(mol_xyz, pharm_xyz), axis=1)
    ids = np.flatnonzero(dist <= (np.min(dist) + tol))
    return ids.tolist()


def screen(dbdir, pharmacophore, pids, new_pids, iteration, output_dir, ncpu, pharmit_path, pd_search):
    # screen db against pharmacophore and if no matches screen its subpharmacopohres till the minimum size
    flag = False
    for i in reversed(range(1, len(new_pids) + 1)):
        for j, new_pids_subset in enumerate(combinations(new_pids, i)):
            cur_pids = tuple(sorted(set(pids + new_pids_subset)))
            p_fname = os.path.join(output_dir, f'iter{iteration}-p{len(cur_pids)}-model{j}.json')
            pharmacophore.write_model(p_fname, ids=cur_pids)
            p_fname_xyz = os.path.join(output_dir, f'iter{iteration}-p{len(cur_pids)}-model{j}.xyz')
            pharmacophore.write_model_xyz(p_fname_xyz, ids=cur_pids)
            found_poses_fname = os.path.splitext(p_fname)[0] + '_found.sdf'
            search_db(dbdir, p_fname, found_poses_fname, pharmit_path=pharmit_path, ncpu=ncpu)
            if sdf_not_empty(found_poses_fname):
                pd_search.loc[pd_search.shape[0]] = [iteration, cur_pids, found_poses_fname, None]
                flag = True
            else:
                pd_search.loc[pd_search.shape[0]] = [iteration, cur_pids, None, None]
        if flag:
            break
    return flag


def select(iteration, pd_search):
    N = 10
    for rowid in np.where(pd_search['iteration'] == iteration):
        in_fname = pd_search.loc[rowid, 'found_sdf'].values[0]
        if in_fname is None:
            continue
        out_fname = os.path.splitext(in_fname)[0] + '_selected.sdf'
        pd_search.loc[rowid, 'selected_sdf'] = out_fname
        mols = [(mol.GetNumHeavyAtoms(), mol) for mol in Chem.SDMolSupplier(in_fname)]
        min_hac = min(i for i, j in mols)
        w = Chem.SDWriter(out_fname)
        for i, (hac, mol) in enumerate(sorted(mols, key=operator.itemgetter(0))):
            if i < N or mol.GetNumHeavyAtoms() <= min_hac:
                w.write(mol)
        w.close()


def remove_confs_rms(mol, rms=0.5):

    remove_ids = []
    cids = [c.getId() for c in mol.GetConformers()]

    rms_list = [(i1, i2, norm(mol.GetConformer(i1).GetPositions() - mol.GetConformer(i2).GetPositions()))
                for i1, i2 in combinations(cids, 2)]
    while any(item[2] < rms for item in rms_list):
        for item in rms_list:
            if item[2] < rms:
                remove_ids.append(item[1])
                rms_list = [i for i in rms_list if i[0] != item[1] and i[1] != item[1]]
                break

    if remove_ids:
        for cid in set(remove_ids):
            mol.RemoveConformer(cid)
        # conformers are reindexed staring with 0 step 1
        for i, conf in enumerate(mol.GetConformers()):
            conf.SetId(i)

    return mol


def get_confs(mol, template_mol, pharm_coords, dist, nconf, seed, **kwargs):
    """

    :param mol: RDKit Mol
    :param template_mol: template RDKit MOl
    :param pharm_coords: pandas df with first column feature labels and three columns of coordinates
    :param dist: maximum allowed distance to pharmacophore feature, conformers where all distances to features greater
                 than this value will be discarded
    :param nconf: number of conformers
    :return: mol with suitable conformers or None if no conformer matches
    """

    dists = []
    labels = pharm_coords['label'].unique().tolist()
    mol = __gen_confs(mol, template_mol, nconf=nconf, seed=seed, **kwargs)
    mol = remove_confs_rms(mol, rms=dist)
    if mol:
        plist = load_multi_conf_mol(mol)
        for conf_id, p in enumerate(plist):
            conf_coords = p.get_feature_coords()
            conf_coords = pd.DataFrame([(label, *xyz) for label, xyz in conf_coords], columns=['label', 'x', 'y', 'z'])
            res = []
            for lb in labels:
                if lb in conf_coords.label.values:
                    res.append(np.min(cdist(conf_coords.loc[conf_coords.label == lb, ['x', 'y', 'z']],
                                            pharm_coords.loc[pharm_coords.label == lb, ['x', 'y', 'z']]),
                                      axis=0)[0])
            dists.append(res)
        dists = np.array(dists)
        # remove conformers which can not fit
        remove_cids = np.where((dists > dist).all(axis=1))[0]
        if remove_cids.size:
            if remove_cids.shape[0] == mol.GetNumConformers():
                mol = None
            else:
                print(dists)
                remove_cids = list(map(int, remove_cids))
                mol = remove_conf(mol, remove_cids)
    return mol


def remove_conf(mol, cids):
    for cid in set(cids):
        mol.RemoveConformer(cid)
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)
    return mol


def screen_mols_simple(mols, fname, template_mol, pool, nconf, seed):
    with open(fname, 'a') as f:
        w = Chem.SDWriter(f)
        if pool:
            for mol in pool.imap_unordered(partial(__gen_confs, template_mol=template_mol, nconf=nconf, seed=seed), mols):
                if mol:
                    for c in mol.GetConformers():
                        w.write(mol, c.GetId())
        else:
            for mol in mols:
                mol = __gen_confs(mol, template_mol=template_mol, nconf=nconf, seed=seed)
                if mol:
                    for c in mol.GetConformers():
                        w.write(mol, c.GetId())
        w.close()


def screen_simple(db, pharmacophore, pids, new_pids, iteration, output_dir, dist, pd_search):
    # screen db against pharmacophore and if no matches screen its subpharmacopohres till the minimum size
    flag = False
    for i in reversed(range(1, len(new_pids) + 1)):
        for j, new_pids_subset in enumerate(combinations(new_pids, i)):
            cur_pids = tuple(sorted(set(pids + new_pids_subset)))
            p_fname = os.path.join(output_dir, f'iter{iteration}-p{len(cur_pids)}-model{j}.json')
            pharmacophore.write_model(p_fname, ids=cur_pids)
            found_poses_fname = os.path.splitext(p_fname)[0] + '_found.sdf'
            with open(found_poses_fname, 'a') as f:
                w = Chem.SDWriter(f)
                for mol_name in db.get_mol_names():
                    # determine conf_id of a conformer which has all distances below threshold and
                    # minimum sum of these distances
                    dists = db.get_dists(mol_name)[list(new_pids_subset)]
                    cids = np.where((dists <= dist).all(axis=1))[0]
                    if cids.size:
                        sum_dists = np.sum(dists, axis=1)
                        cid = cids[np.argmin(sum_dists[cids])]
                        mol = db.get_mol(mol_name)
                        w.write(mol, int(cid))
                w.close()
            if sdf_not_empty(found_poses_fname):
                pd_search.loc[pd_search.shape[0]] = [iteration, cur_pids, found_poses_fname, None]
                flag = True
            else:
                pd_search.loc[pd_search.shape[0]] = [iteration, cur_pids, None, None]
        if flag:
            break
    return flag



def keep_confs(mol, ids):

    all_ids = set(c.GetId() for c in mol.GetConformers())
    remove_ids = all_ids - set(ids)
    for cid in set(remove_ids):
        mol.RemoveConformer(cid)
    # conformers are reindexed staring with 0 step 1
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)
    return mol


def screen_pmapper(query_pharm, db_fname, output_sdf):

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
            if min_rmsd <= 0.2:
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


def select_mols(input_pkl):

    mols = [(mol, mol_name, mol.GetNumHeavyAtoms()) for mol, mol_name in read_input(input_pkl)]
    mols = sorted(mols, key=itemgetter(2))
    hacs = np.array([item[2] for item in mols])
    deleted = np.zeros(hacs.shape)

    for i, (mol, mol_name, hac) in enumerate(mols):
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

    # if args.ncpu > 1:
    pool = Pool(args.ncpu)
    # else:
    #     pool = None

    args.ids = tuple(sorted(set(args.ids)))

    # tree_search = defaultdict(dict)  # {iteration_number: {pids: sdf_fname, pids: sdf_fname, ...}, ...}
    pd_search = pd.DataFrame(columns=['iteration', 'feature_ids', 'found_sdf', 'selected_sdf'])

    p = PharmModel2()
    p.load_from_xyz(args.query)
    p.set_clusters(args.clustering_threshold, args.ids)

    print(p.clusters)

    iteration = 0

    conf_fname = os.path.join(args.output, f'iter{iteration}.sdf')
    conf_fname_pkl = os.path.splitext(conf_fname)[0] + '.pkl'

    new_pids = tuple(args.ids)
    pd_search.loc[0] = [-1, tuple(), None, None]

    flag = screen_pmapper(query_pharm=p.get_subpharmacophore(new_pids), db_fname=args.fragments, output_sdf=conf_fname)
    if flag:
        pd_search.loc[pd_search.shape[0]] = [iteration, new_pids, conf_fname, None]
    else:
        exit('No matches between starting fragments and the chosen subpharmacophore.')

    mols = select_mols(conf_fname_pkl)
    conf_selected_fname = os.path.splitext(conf_fname)[0] + '_selected.sdf'
    save_mols(mols, conf_selected_fname)
    pd_search.loc[pd_search['iteration'] == iteration, 'selected_sdf'] = conf_selected_fname

    iteration = 1

    stereo_opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)

    while True:

        # new_pids will be the same for different lines from the same iteration because only a cluster with
        # non intersected feature ids can be selected
        new_pids = p.select_nearest_cluster(pd_search.loc[pd_search['iteration'] == iteration - 1].iloc[0]['feature_ids'])

        # if no new features then exit
        if not new_pids:
            break

        new_xyz = p.get_xyz(new_pids)
        new_pharm_coords = p.get_feature_coords_pd(new_pids)

        print(new_pids)
        print(new_pharm_coords)

        # all generated compounds are stored in a single file, however their parent compounds could match different
        # subsets of features. This was done due to fast screening, so we will not lose much time, but greatly
        # simplify manipulation with data
        conf_fname = os.path.join(args.output, f'iter{iteration}.sdf')

        for i, rowid in enumerate(np.where(pd_search['iteration'] == iteration - 1)):

            fname = pd_search.iloc[rowid]['selected_sdf'].values[0]
            if fname is None:
                continue

            for j, (parent_mol, parent_mol_name) in enumerate(read_input(os.path.splitext(fname)[0] + '.pkl')):
                for conf in parent_mol.GetConformers():
                    replace_ids = get_grow_points2(conf.GetPositions(), new_xyz)
                    new_mols = list(grow_mol(parent_mol, args.db, radius=2, min_atoms=1, max_atoms=12,
                                             max_replacements=None, replace_ids=replace_ids, return_mol=True,
                                             ncores=args.ncpu))
                    new_isomers = []
                    for isomers in pool.imap_unordered(gen_stereo, (m[1] for m in new_mols)):
                        new_isomers.extend(isomers)
                    for k, new_mol in enumerate(new_isomers):
                        new_mol.SetProp('_Name', f'mol-{iteration}-{str(j).zfill(5)}-{str(k).zfill(6)}')

                    # add parent compounds with the same conformation, because it may accidentally match
                    # a larger subset of features. But since we do not change its conformation this would be
                    # unlikely
                    new_mols = new_isomers + [parent_mol]

                    # add to file molecule and distances to new pharmacophore features
                    with open(conf_fname, 'at') as f:
                        w = Chem.SDWriter(f)
                        if pool:
                            generator = pool.imap_unordered(partial(get_confs,
                                                                    template_mol=parent_mol,
                                                                    nconf=args.nconf,
                                                                    seed=args.seed,
                                                                    dist=args.dist,
                                                                    pharm_coords=new_pharm_coords,
                                                                    coreConfId=conf.GetId()), new_mols)
                        else:
                            generator = (get_confs(mol, parent_mol, pharm_coords=new_pharm_coords,
                                                   nconf=args.nconf, dist=args.dist, seed=args.seed,
                                                   coreConfId=conf.GetId())
                                         for mol in new_mols)
                        for mol in generator:
                            if mol:
                                for i in range(mol.GetNumConformers()):
                                    w.write(mol, i)
                        w.close()

        print(f'{conf_fname} was created')

        db_dname = os.path.join(args.output, f'iter{iteration}_db')
        create_db(conf_fname, db_dname, pharmit_path=args.pharmit, pharmit_spec=args.pharmit_spec, ncpu=args.ncpu)

        flags = []
        for rowid in np.where(pd_search['iteration'] == iteration - 1):
            print(pd_search.iloc[rowid]['feature_ids'].values[0], new_pids)
            flag = screen(dbdir=db_dname, pharmacophore=p, pids=pd_search.iloc[rowid]['feature_ids'].values[0],
                          new_pids=new_pids, iteration=iteration, output_dir=args.output, ncpu=args.ncpu,
                          pd_search=pd_search, pharmit_path=args.pharmit)
            select(iteration, pd_search)
            flags.append(flag)
        if any(flags):
            iteration += 1
        else:
            break


if __name__ == '__main__':
    main()
