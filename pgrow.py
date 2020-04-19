#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import shutil
import random
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from crem.crem import grow_mol, grow_mol2


class PharmModel:

    def __init__(self, fname):
        p = json.load(open(fname))['points']
        self.p = [i for i in p if i['enabled']]
        self.clusters = defaultdict(set)

    def xyz(self, ids=None, cluster=None):
        """

        :param ids: list of feature ids or None (all features)
        :param cluster: number of a cluster, if not None it has higher precedence than ids
        :return: numpy array of shape (N, 3)
        """
        if cluster:
            ids = self.clusters[cluster]
        if ids:
            m = np.array([(self.p[i]['x'], self.p[i]['y'], self.p[i]['z']) for i in ids])
        else:
            m = np.array([(i['x'], i['y'], i['z']) for i in self.p])
        return m

    def set_clusters(self, clustering_threshold):
        c = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_threshold).fit(self.xyz())
        for i, j in enumerate(c.labels_):
            self.clusters[j].add(i)

    def select_nearest_cluster(self, ids):

        def min_distance(ids1, ids2):
            xyz1 = self.xyz(ids1)
            xyz2 = self.xyz(ids2)
            return np.min(cdist(xyz1, xyz2))

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
        if ids:
            p = [self.p[i] for i in ids]
        else:
            p = self.p
        p = {'points': p}
        json.dump(p, open(fname, 'wt'))


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


def __embed_multiple_confs_constrained(mol, template_mol, nconf, seed=42):
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


def __gen_confs(mol, template_mol=None, nconf=10, seed=42):
    mol = Chem.AddHs(mol)
    if template_mol:
        mol = __embed_multiple_confs_constrained(mol, template_mol, nconf, seed)
    else:
        AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=nconf*4, randomSeed=seed)
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
                for c in mol.GetConformers():
                    w.write(mol, c.GetId())
        else:
            for mol in mols:
                mol = __gen_confs(mol, template_mol=template_mol, nconf=nconf, seed=seed)
                for c in mol.GetConformers():
                    w.write(mol, c.GetId())
        w.close()


def create_db(fname, dbname, ncpu=1):
    subprocess.run(['/home/pavel/pharmit/pharmit/src/build/pharmit',
                    'dbcreate', '-dbdir', dbname, '-in', fname, '-nthreads', str(ncpu)])


def search_db(dbname, query_fname, out_fname, ncpu=1):
    subprocess.run(['/home/pavel/pharmit/pharmit/src/build/pharmit',
                    'dbsearch', '-dbdir', dbname, '-in', query_fname, '-out', out_fname,
                    '-reduceconfs', '1', '-sort-rmsd', '-nthreads', str(ncpu)])


def sdf_not_empty(fname):
    for m in Chem.SDMolSupplier(fname):
        if m:
            return True
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
    ids = np.where(dist <= np.min(dist) + tol)
    return tuple(map(int, np.flatnonzero(ids)))


def grow_molecule():
    pass


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
    parser.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=100,
                        help='number of conformers generated per structure. Default: 100.')
    parser.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=-1,
                        help='seed for random number generator to get reproducible output. Default: -1.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use. Default: 1.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    if args.ncpu > 1:
        pool = Pool(args.ncpu)
    else:
        pool = None

    tree_search = dict()

    p = PharmModel(args.query)
    p.set_clusters(args.clustering_threshold)

    print(p.clusters)

    iteration = 0

    conf_fname = os.path.join(args.output, f'iter{iteration}.sdf')
    if args.fragments.endswith('.smi'):
        frags = load_frags(args.fragments)
        append_generated_confs_file(frags, conf_fname, nconf=10, pool=pool, seed=args.seed)
    elif args.fragments.endswith('.sdf'):
        shutil.copyfile(args.fragments, conf_fname)

    if os.path.isdir(args.fragments):
        db_dname = args.fragments
    else:
        db_dname = os.path.join(args.output, f'iter{iteration}_db')
        create_db(conf_fname, db_dname, ncpu=args.ncpu)

    p_ids = tuple(args.ids)

    while True:

        if iteration > 0:
            new_ids = p.select_nearest_cluster(p_ids)
            p_ids = p_ids + new_ids
            conf_fname = os.path.join(args.output, f'iter{iteration}.sdf')
            mols = [mol for mol in Chem.SDMolSupplier(found_poses_fname) if mol]
            for i, mol in enumerate(mols):
                replace_ids = get_grow_points(mol, p.xyz(new_ids))
                new_mols = list(grow_mol(mol, args.db, radius=3, min_atoms=1, max_atoms=3, max_replacements=None,
                                         replace_ids=replace_ids, return_mol=True, ncores=args.ncpu))
                new_mols = [item[1] for item in new_mols]
                for j, new_mol in enumerate(new_mols):
                    new_mol.SetProp('_Name', f'mol-{iteration}-{str(i).zfill(5)}-{str(j).zfill(6)}')
                append_generated_confs_file(new_mols, fname=conf_fname, template_mol=mol, nconf=args.nconf,
                                            pool=pool, seed=args.seed)
            db_dname = os.path.join(args.output, f'iter{iteration}_db')
            create_db(conf_fname, db_dname, ncpu=args.ncpu)

        p_fname = os.path.join(args.output, f'iter{iteration}.json')
        p.write_model(p_fname, ids=p_ids)

        found_poses_fname = os.path.join(args.output, f'iter{iteration}_found.sdf')
        search_db(db_dname, p_fname, found_poses_fname, ncpu=args.ncpu)

        if sdf_not_empty(found_poses_fname):
            tree_search[p_ids] = found_poses_fname
            iteration += 1
        else:
            break


if __name__ == '__main__':
    main()
