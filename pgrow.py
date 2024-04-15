#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
import json
import pickle
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import numpy as np
import sqlite3
import subprocess
import timeit
import tempfile
import yaml
from math import cos, sin, pi

from rdkit import Chem
from rdkit.Chem import AllChem
from pmapper.pharmacophore import Pharmacophore as P
from psearch.database import DB
from read_input import read_input
from dask.distributed import Client, as_completed

from expand_mol import remove_confs_exclvol, select_mols, merge_confs, remove_confs_rms
from pharm_class import PharmModel2


def is_collinear(p, epsilon=0.1):
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
        if (d[i, id1] + d[i, id2]) - max_dist > epsilon:
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
        remove_conf_ids = [x.GetId() for x in m.GetConformers() if x.GetId() not in rmsd_dict.keys()]
        for conf_id in sorted(remove_conf_ids, reverse=True):
            m.RemoveConformer(conf_id)
        m = remove_confs_rms(m)
        for conf in m.GetConformers():
            conf_id = conf.GetId()
            w.write(m, conf_id)
            if rotate_mat is not None:  # rotate molecule if pharmacophore features are collinear
                for _ in range(360 // theta - 1):
                    AllChem.TransformMol(m, rotate_mat, conf_id, keepConfs=True)
                    w.write(m, conf_id)
    w.close()
    pool.close()

    return True


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
                    "processing INTEGER NOT NULL, "
                    "priority INTEGER NOT NULL, "
                    "time TEXT NOT NULL)")
        conn.commit()


def save_res(mols, parent_mol_id, db_fname):

    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()

        if parent_mol_id is not None:
            cur.execute(f'UPDATE mols SET used = 1, processing = 0 WHERE id = {parent_mol_id}')
            conn.commit()

        if mols:
            cur.execute('SELECT MAX(id) FROM mols')
            mol_id = cur.fetchone()[0]
            if mol_id is None:
                mol_id = 0
            parent_mol_id = mols[0].GetPropsAsDict().get('parent_mol_id', None)
            if parent_mol_id is not None:
                cur.execute(f'SELECT distinct(priority) FROM mols where id = {parent_mol_id}')
                parent_priority = cur.fetchone()[0]
                priority = parent_priority + 1
            else:
                priority = 1
            for mol in mols:
                parent_mol_id = mol.GetPropsAsDict().get('parent_mol_id', None)
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
                    sql = 'INSERT INTO mols (id, conf_id, mol_block, matched_ids, visited_ids, ' \
                          '                  matched_ids_count, visited_ids_count, parent_mol_id, ' \
                          '                  parent_conf_id, priority, used, processing, nmols, time) ' \
                          'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)'
                    cur.execute(sql, (mol_id, conf.GetId(), mol_block, matched_ids, visited_ids, matched_ids_count,
                                      visited_ids_count, parent_mol_id, parent_conf_id, priority, 0, 0, 0))
                priority += 3
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


def expand_mol_cli(mol, pharm_fname, config_fname):

    input_fd, input_fname = tempfile.mkstemp(suffix='_input.pkl', text=True)
    with open(input_fname, 'wb') as f:
        pickle.dump(mol, f)

    output_fd, output_fname = tempfile.mkstemp(suffix='_output.pkl', text=True)
    conf_count_fd, conf_count_fname = tempfile.mkstemp(suffix='_conf_count.txt', text=True)
    debug_fd, debug_fname = tempfile.mkstemp(suffix='_debug.txt', text=True)

    try:
        dname = os.path.dirname(os.path.realpath(__file__))
        python_exec = sys.executable
        cmd = f'{python_exec} {os.path.join(dname, "expand_mol.py")} -i {input_fname} -o {output_fname} ' \
              f'-p {pharm_fname} --config {config_fname} --debug {debug_fname} --conf_count {conf_count_fname}'
        # start_time = timeit.default_timer()
        subprocess.run(cmd, shell=True)
        # run_time = round(timeit.default_timer() - start_time, 1)

        with open(output_fname, 'rb') as f:
            new_mols = pickle.load(f)

        with open(debug_fname) as f:
            debug = ''.join(f.readlines())

        with open(conf_count_fname) as f:
            conf_mol_count = int(f.readline().strip())

    finally:
        os.close(input_fd)
        os.close(output_fd)
        os.close(conf_count_fd)
        os.close(debug_fd)
        os.unlink(input_fname)
        os.unlink(output_fname)
        os.unlink(conf_count_fname)
        os.unlink(debug_fname)

    # return tuple([1, tuple(), 3, 'asdf'])
    return tuple([int(mol.GetProp('_Name')), tuple(new_mols), conf_mol_count, debug])


def get_mol_to_expand(db_fname, max_features):
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        cur.execute(f'SELECT id, MIN(priority) '
                    f'FROM mols '
                    f'WHERE used = 0 AND processing = 0 AND visited_ids_count < {max_features}')
        res = cur.fetchone()
        if res == (None, None):
            return None

        mol_id, priority = res

        cur.execute(f'SELECT id, conf_id, mol_block, matched_ids, visited_ids '
                    f'FROM mols '
                    f'WHERE id = {mol_id}')
        res = cur.fetchall()

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

        cur.execute(f'UPDATE mols SET processing = 1 WHERE id = {mol_id}')

        return mol


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
    parser.add_argument('--mw', metavar='NUMERIC', required=False, type=float, default=450,
                        help='Maximum molecular weight of generated compounds. Default: 450.')
    parser.add_argument('--tpsa', metavar='NUMERIC', required=False, type=float, default=120,
                        help='Maximum TPSA of generated compounds. Default: 120.')
    parser.add_argument('--rtb', metavar='NUMERIC', required=False, type=float, default=7,
                        help='Maximum number of rotatable bonds in generated compounds. Default: 7.')
    parser.add_argument('--logp', metavar='NUMERIC', required=False, type=float, default=4,
                        help='Maximum logP of generated compounds. Default: 4.')
    parser.add_argument('--hash_db', metavar='FILENAME', required=False, default=None,
                        help='database with 3D pharmacophore hashes for additional filtering of fragments for growing.')
    parser.add_argument('--hash_db_bin_step', metavar='NUMERIC', required=False, default=1, type=float,
                        help='bin step used to create 3D pharmacophore hashes.')
    parser.add_argument('-u', '--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    parser.add_argument('-w', '--num_workers', metavar='INTEGER', required=False, type=int, default=1,
                        help='the number of workers to be spawn by the dask cluster. This will limit the maximum '
                             'number of processed molecules simultaneously. Default: 1.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use per molecule. Default: 1.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    if args.hostfile is not None:
        with open(args.hostfile) as f:
            dask_client = Client(f.readline().strip() + ':8786')
    else:
        dask_client = Client(n_workers=args.num_workers)

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'setup.json'), 'wt') as f:
        f.write(json.dumps(vars(args), indent=4))

    shutil.copyfile(args.query, os.path.join(args.output, os.path.basename(args.query)))

    args.ids = tuple(sorted(set(args.ids)))

    p = PharmModel2()
    p.load_from_xyz(args.query)
    p.set_clusters(args.clustering_threshold, args.ids)

    print(p.clusters, flush=True)

    res_db_fname = os.path.join(args.output, 'res.db')

    if not os.path.isfile(res_db_fname):  # create DB and match starting fragments

        create_db(res_db_fname)

        conf_fname = os.path.join(args.output, f'iter0.sdf')

        new_pids = tuple(args.ids)

        print(f"===== Initial screening =====")
        start = timeit.default_timer()

        flag = screen_pmapper(query_pharm=p.get_subpharmacophore(new_pids), db_fname=args.fragments,
                              output_sdf=conf_fname, rmsd=0.2, ncpu=args.ncpu)
        if not flag:
            exit('No matches between starting fragments and the chosen subpharmacophore.')

        print(f'{round(timeit.default_timer() - start, 4)}')

        mols = [mol for mol, mol_name in read_input(conf_fname, sdf_confs=True)]
        mols = [remove_confs_exclvol(mol, p.exclvol, args.exclusion_volume) for mol in mols]
        mols = [m for m in mols if m]  # remove None objects
        mols = select_mols(mols, ncpu=args.ncpu)
        ids = ','.join(map(str, new_pids))
        for mol in mols:
            mol.SetProp('visited_ids', ids)
            for conf in mol.GetConformers():
                conf.SetProp('matched_ids', ids)
        mols = merge_confs({None: mols})   # return list of mols
        mols = [remove_confs_rms(m) for m in mols]
        for mol in mols:
            for conf in mol.GetConformers():
                conf.SetProp('parent_conf_id', 'None')
        save_res(mols, None, res_db_fname)

    else:  # set all processing flags to 0 (for restart)
        with sqlite3.connect(res_db_fname) as conn:
            conn.execute("UPDATE mols SET processing = 0 WHERE processing = 1")

    pharm_fd, pharm_fname = tempfile.mkstemp(suffix='_pharm.pkl', text=True)
    with open(pharm_fname, 'wb') as f:
        pickle.dump(p, f)

    config_fd, config_fname = tempfile.mkstemp(suffix='_config.yml', text=True)
    with open(config_fname, 'wt') as f:
        yaml.safe_dump({'additional_features': args.additional_features,
                        'max_mw': args.mw, 'max_tpsa': args.tpsa, 'max_rtb': args.rtb,
                        'max_logp': args.logp, 'hash_db': args.hash_db,
                        'hash_db_bin_step': args.hash_db_bin_step, 'crem_db': args.db,
                        'radius': args.radius, 'max_replacements': args.max_replacements,
                        'nconf': args.nconf, 'conf_gen': args.conf_gen, 'dist': args.dist,
                        'exclusion_volume_dist': args.exclusion_volume, 'seed': args.seed,
                        'output_dir': args.output, 'dask_num_workers': 0, 'ncpu': args.ncpu},
                       f)

    try:

        max_tasks = 2 * args.num_workers
        futures = []
        for _ in range(max_tasks):
            m = get_mol_to_expand(res_db_fname, p.get_num_features())
            if m:
                futures.append(dask_client.submit(expand_mol_cli, m, pharm_fname=pharm_fname, config_fname=config_fname))
        seq = as_completed(futures, with_results=True)
        for i, (future, (parent_mol_id, new_mols, nmols, debug)) in enumerate(seq, 1):
            save_res(new_mols, parent_mol_id, res_db_fname)
            if nmols:
                update_db(res_db_fname, parent_mol_id, nmols)
            if debug:
                print(f'===== {parent_mol_id} =====')
                print(debug)
                sys.stdout.flush()
            del future
            for _ in range(max_tasks - seq.count()):
                m = get_mol_to_expand(res_db_fname, p.get_num_features())
                if m:
                    new_future = dask_client.submit(expand_mol_cli, m, pharm_fname=pharm_fname, config_fname=config_fname)
                    seq.add(new_future)

    finally:

        os.close(pharm_fd)
        os.close(config_fd)
        os.unlink(pharm_fname)
        os.unlink(config_fname)


if __name__ == '__main__':
    main()
