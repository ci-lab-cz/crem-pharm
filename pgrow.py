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
from itertools import combinations
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from crem.crem import grow_mol, grow_mol2
from openbabel import openbabel as ob
from openbabel import pybel


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
            m = np.array([(self.p[i]['x'], self.p[i]['y'], self.p[i]['z']) for i in set(ids)])
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
        if ids:
            p = [self.p[i] for i in set(ids)]
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


def __embed_multiple_confs_constrained(mol, template_mol, nconf, seed=42):
    mol = AllChem.ConstrainedEmbed(Chem.AddHs(mol), template_mol, randomseed=seed)
    ids = set(range(mol.GetNumAtoms())) - set(mol.GetSubstructMatch(template_mol))
    ids = [i + 1 for i in ids]   # ids of atoms to rotate
    mol_rdkit = mol

    mol = pybel.readstring('mol', Chem.MolToMolBlock(mol)).OBMol   # convert mol from RDKit to OB

    pff = ob.OBForceField_FindType("mmff94")
    if not pff.Setup(mol):  # if OB FF setup fails use RDKit conformer generation (slower)
        return __embed_multiple_confs_constrained_rdkit(mol_rdkit, template_mol, nconf, seed)

    constraints = ob.OBFFConstraints()
    for atom in ob.OBMolAtomIter(mol):
        atom_id = atom.GetIndex() + 1
        if atom_id not in ids:
            constraints.AddAtomConstraint(atom_id)
    pff.SetConstraints(constraints)

    pff.DiverseConfGen(0.5, 1000, 50, False)   # rmsd, nconf_tries, energy, verbose

    pff.GetConformers(mol)
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('sdf')

    output_strings = []
    for conf_num in range(max(0, mol.NumConformers() - nconf), mol.NumConformers()):   # save last nconf conformers (it seems the last one is the original conformer)
        mol.SetConformer(conf_num)
        output_strings.append(obconversion.WriteString(mol))

    return ''.join(output_strings)


def __gen_confs(mol, template_mol=None, nconf=10, seed=42):
    try:
        if template_mol:
            mol = __embed_multiple_confs_constrained(mol, template_mol, nconf, seed)
        else:
            AllChem.EmbedMultipleConfs(Chem.AddHs(mol), numConfs=nconf, maxAttempts=nconf*4, randomSeed=seed)
    except ValueError:
        return None
    return mol  # string if restrained or RDKit mol with multiple conformers if not restrained


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
                    if isinstance(mol, Chem.Mol):
                        for c in mol.GetConformers():
                            w.write(mol, c.GetId())
                    else:
                        f.write(mol)
        else:
            for mol in mols:
                mol = __gen_confs(mol, template_mol=template_mol, nconf=nconf, seed=seed)
                if mol:
                    if isinstance(mol, Chem.Mol):
                        for c in mol.GetConformers():
                            w.write(mol, c.GetId())
                    else:
                        f.write(mol)
        w.close()


def create_db(fname, dbname, pharmit_path, ncpu=1):
    subprocess.run([pharmit_path, 'dbcreate', '-dbdir', dbname, '-in', fname, '-nthreads', str(ncpu)])


def search_db(dbname, query_fname, out_fname, pharmit_path, ncpu=1):
    subprocess.run([pharmit_path, 'dbsearch', '-dbdir', dbname, '-in', query_fname, '-out', out_fname,
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
            found_poses_fname = os.path.splitext(p_fname)[0] + '_found.sdf'
            search_db(dbdir, p_fname, found_poses_fname, pharmit_path=pharmit_path, ncpu=ncpu)
            if sdf_not_empty(found_poses_fname):
                pd_search.loc[pd_search.shape[0]] = [iteration, cur_pids, found_poses_fname]
                flag = True
        if flag:
            break
    return flag


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
    parser.add_argument('-p', '--pharmit', metavar='pharmit executable', required=False, type=str,
                        default='/home/pavel/pharmit/pharmit/src/build/pharmit',
                        help='path to pharmit executable.')
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

    args.ids = tuple(sorted(set(args.ids)))

    # tree_search = defaultdict(dict)  # {iteration_number: {pids: sdf_fname, pids: sdf_fname, ...}, ...}
    pd_search = pd.DataFrame(columns=['iteration', 'feature_ids', 'found_sdf'])

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

    new_pids = tuple(args.ids)
    pd_search.loc[0] = [-1, tuple(), '']

    while True:

        if iteration > 0:
            # new_pids will be the same for different lines from the same iteration because only a cluster with
            # non intersected feature ids canbe selected
            new_pids = p.select_nearest_cluster(pd_search.loc[pd_search['iteration'] == iteration - 1].iloc[0]['feature_ids'])
            # new_pids = p.select_nearest_cluster(tmp['feature_ids'])
            # all generated compounds are stored in a single file, however their parent compounds could match different
            # subsets of features. This was done due to fast screening, so we will not lose much time, but greatly
            # simplify manipulation with data
            conf_fname = os.path.join(args.output, f'iter{iteration}.sdf')
            # to avoid enumeration of the same compounds we keep canonical isomeric smiles of already read molecules,
            # because ot is unlikely that the same compound will have different orientations in matching different
            # subsets of features (these subsets will substantially intersect)
            read_smi = set()
            for i, rowid in enumerate(np.where(pd_search['iteration'] == iteration - 1)):
                for j, mol in enumerate(Chem.SDMolSupplier(pd_search.iloc[rowid]['found_sdf'].values[0])):
                    if mol:
                        smi = Chem.MolToSmiles(mol)
                        if smi not in read_smi:
                            print(mol.GetProp('_Name'))
                            read_smi.add(smi)
                            replace_ids = get_grow_points(mol, p.xyz(new_pids))
                            new_mols = list(grow_mol(mol, args.db, radius=3, min_atoms=1, max_atoms=8,
                                                     max_replacements=None, replace_ids=replace_ids, return_mol=True,
                                                     ncores=args.ncpu))
                            new_mols = [item[1] for item in new_mols]
                            for k, new_mol in enumerate(new_mols):
                                new_mol.SetProp('_Name', f'mol-{iteration}-{str(j).zfill(5)}-{str(k).zfill(6)}')
                            # add parent compounds with the same conformation, because it may accidentally match
                            # a large subset of features. But since we do not change its conformation this would be
                            # unlikely
                            new_mols += [mol]
                            append_generated_confs_file(new_mols, fname=conf_fname, template_mol=mol, nconf=args.nconf,
                                                        pool=pool, seed=args.seed)
                db_dname = os.path.join(args.output, f'iter{iteration}_db')
                create_db(conf_fname, db_dname, pharmit_path=args.pharmit, ncpu=args.ncpu)

        flags = []
        for rowid in np.where(pd_search['iteration'] == iteration - 1):
            print(pd_search.iloc[rowid]['feature_ids'].values[0], new_pids)
            flag = screen(dbdir=db_dname, pharmacophore=p, pids=pd_search.iloc[rowid]['feature_ids'].values[0],
                          new_pids=new_pids, iteration=iteration, output_dir=args.output, ncpu=args.ncpu,
                          pd_search=pd_search, pharmit_path=args.pharmit)
            flags.append(flag)
        if any(flags):
            iteration += 1
        else:
            break


if __name__ == '__main__':
    main()
