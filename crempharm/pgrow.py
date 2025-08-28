#!/usr/bin/env python3

import argparse
import os
import sys
import shutil
import json
import pickle
from multiprocessing import Pool, cpu_count

from scipy.spatial.distance import cdist
import numpy as np
import sqlite3
import subprocess
import timeit
import tempfile
import yaml
import logging
from math import cos, sin, pi
from functools import partial
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem
from pmapper.pharmacophore import Pharmacophore as P
from psearch.database import DB
from dask.distributed import Client, as_completed

from crempharm.expand_mol import remove_confs_exclvol, select_mols, merge_confs, remove_confs_rms
from crempharm.pharm_class import PharmModel2
from crempharm.database import create_db, save_res, update_db, get_stat_string_from_db


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


def screen(mol_name, mol, pharm_list, query_mol, query_nfeatures, rmsd_to_query, theta, rotate_matrix, exclvol_xyz, exclvol_dist):

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
        if min_rmsd <= rmsd_to_query:
            rmsd_dict[i] = AllChem.GetAlignmentTransform(Chem.Mol(m1), query_mol,
                                                         atomMap=tuple(zip(min_ids, range(query_nfeatures))))

    # combine conformers in one Mol object and rotate if needed
    m = Chem.Mol(mol)
    for k, (rms, matrix) in rmsd_dict.items():
        AllChem.TransformMol(m, matrix, k, keepConfs=True)
    remove_conf_ids = [x.GetId() for x in m.GetConformers() if x.GetId() not in rmsd_dict.keys()]
    for conf_id in sorted(remove_conf_ids, reverse=True):
        m.RemoveConformer(conf_id)
    m = remove_confs_rms(m)
    new_m = Chem.Mol(m)
    new_m.RemoveAllConformers()
    for conf in m.GetConformers():
        new_m.AddConformer(conf, assignId=True)
        if rotate_matrix is not None:  # rotate molecule if pharmacophore features are collinear
            conf_id = conf.GetId()
            for _ in range(360 // theta - 1):
                AllChem.TransformMol(m, rotate_matrix, conf_id, keepConfs=True)
                new_m.AddConformer(m.GetConformer(id=conf_id), assignId=True)

    new_m.SetProp('_Name', mol_name)
    new_m = remove_confs_exclvol(new_m, exclvol_xyz, exclvol_dist)   # can return None
    if new_m:
        new_m = remove_confs_rms(new_m, keep_nconf=20)

    return new_m


def screen_mp(args, **kwargs):
    return screen(*args, **kwargs)


def supply_screen(db):
    mol_names = db.get_mol_names()
    for mol_name in mol_names:
        pharm_dict = db.get_pharm(mol_name)
        try:
            mol = db.get_mol(mol_name)[0]   # because there is only one stereoisomer for each entry
            pharm_list = pharm_dict[0]   # because there is only one stereoisomer for each entry
        except KeyError:
            sys.stderr.write(f'{mol_name} does not have mol[0] or pharm_dict[0]\n')
            continue
        yield mol_name, mol, pharm_list


def screen_pmapper(query_pharm, db_fname, output_sdf, rmsd_to_query, exclvol_xyz, exclvol_dist, ncpu):

    db = DB(db_fname)
    query_mol = query_pharm.get_mol()

    # for additional sampling of collinear pharmacophores
    theta = 10
    if is_collinear(query_pharm):
        rotate_mat = get_rotate_matrix_from_collinear_pharm(query_pharm, theta)
    else:
        rotate_mat = None

    pool = Pool(ncpu)
    output = []
    for mol in pool.imap_unordered(partial(screen_mp,
                                           query_mol=query_mol,
                                           query_nfeatures=query_mol.GetNumAtoms(),
                                           rmsd_to_query=rmsd_to_query,
                                           theta=theta,
                                           rotate_matrix=rotate_mat,
                                           exclvol_xyz=exclvol_xyz,
                                           exclvol_dist=exclvol_dist),
                                   supply_screen(db)):
        if mol:
            output.append(mol)
    pool.close()

    return output


def choose_mol_to_grow(db_fname, max_features, mol_ids=None):

    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()

        res = None
        if mol_ids:
            cur.execute(f"""SELECT id, conf_id, mol_block, matched_ids, visited_ids 
                            FROM mols 
                            WHERE id = (
                              SELECT id
                              FROM mols
                              WHERE visited_ids_count < {max_features} AND used = 0 AND processing = 0 AND id IN ({",".join("?" * len(mol_ids))})
                              ORDER BY
                                visited_ids_count - matched_ids_count,
                                matched_ids_count DESC,
                                rowid DESC
                              LIMIT 1
                            )""", mol_ids)
            res = cur.fetchall()

        if not res:
            cur.execute(f"""SELECT id, conf_id, mol_block, matched_ids, visited_ids 
                            FROM mols 
                            WHERE id = (
                              SELECT c.id FROM (
                                SELECT DISTINCT 
                                  a.id, 
                                  a.matched_ids_count, 
                                  a.visited_ids_count, 
                                  ifnull(b.nmols, a.nmols) AS parent_nmols,
                                  ifnull(b.processing_nmols, a.processing_nmols) AS parent_processing_nmols
                                FROM (SELECT * FROM mols WHERE visited_ids_count < {max_features} AND used = 0 AND processing = 0) AS a
                                LEFT JOIN mols b ON b.id = a.parent_mol_id
                                ORDER BY
                                  a.visited_ids_count,
                                  a.visited_ids_count - a.matched_ids_count,
                                  parent_processing_nmols,
                                  parent_nmols
                                LIMIT 1
                              ) AS c
                            )""")

            res = cur.fetchall()

        if not res:
            return None

        mol_id = res[0][0]
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

        cur.execute('UPDATE mols SET processing = 1 WHERE id = ?', (mol_id, ))
        conn.commit()
        update_db(db_fname, mol_id, 'processing_nmols', 1)

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


def test_additional_features(cremdb, radius):
    required_columns = {'nA', 'nD', 'nH', 'nAr', 'nN', 'nP'}
    with sqlite3.connect(cremdb) as conn:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info(radius{radius})")
        columns_info = cur.fetchall()
        existing_columns = {col[1] for col in columns_info}
        return required_columns.issubset(existing_columns)
    return False


def entry_point():
    parser = argparse.ArgumentParser(description='Grow structures to fit query pharmacophore.',
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=80))

    group1 = parser.add_argument_group("Input /output")
    group1.add_argument('-q', '--query', metavar='FILENAME', required=True,
                        help='pharmacophore model.')
    group1.add_argument('-f', '--fragments', metavar='FILENAME', required=True,
                        help='file with initial fragments - DB in pmapper format.')
    group1.add_argument('-o', '--output', metavar='DIRNAME', required=True,
                        help='path to directory where intermediate and final results will be stored. '
                             'If output db file (res.db) exists in the directory the computation will be continued '
                             '(skip screening of initial fragment DB).')

    group2 = parser.add_argument_group("Generation settings")
    group2.add_argument('--ids', metavar='INTEGER', required=True, nargs='+', type=int,
                        help='ids of pharmacophore features in a query used for initial screening. 0-index based.')
    group2.add_argument('-t', '--clustering_threshold', metavar='NUMERIC', required=False, type=float, default=3,
                        help='threshold to determine clusters.')
    group2.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=20,
                        help='number of conformers generated per structure.')
    group2.add_argument('--conf_gen', metavar='STRING', required=False, type=str, default='cdpkit',
                        choices=['cdpkit', 'rdkit', 'ob'],
                        help='conformer generator from "cdpkit", "rdkit" and "ob". "cdpkit" and "ob" require '
                             'installation.')
    group2.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=-1,
                        help='seed for random number generator to get reproducible output.')
    group2.add_argument('--dist', metavar='NUMERIC', required=False, type=float, default=1,
                        help='maximum distance between corresponding ligand and protein pharmacophore centers.')
    group2.add_argument('-e', '--exclusion_volume', metavar='NUMERIC', required=False, type=float, default=-1,
                        help='radius of exclusion volumes (distance to heavy atoms). By default exclusion volumes are '
                             'disabled even if they are present in a query pharmacophore. To enable them set '
                             'a positive numeric value.')
    group2.add_argument('--hash_db', metavar='FILENAME', required=False, default=None,
                        help='database with 3D pharmacophore hashes for additional filtering of fragments for growing.')
    group2.add_argument('--hash_db_bin_step', metavar='NUMERIC', required=False, default=1, type=float,
                        help='bin step used to create 3D pharmacophore hashes.')

    group3 = parser.add_argument_group("CReM settings")
    group3.add_argument('-d', '--db', metavar='FILENAME', required=True,
                        help='database with interchangeable fragments.')
    group3.add_argument('-r', '--radius', metavar='INTEGER', type=int, choices=[1, 2, 3, 4, 5], default=3,
                        help='radius of a context of attached fragments.')
    group3.add_argument('--max_replacements', metavar='INTEGER', type=int, default=None,
                        help='maximum number of fragments considered for growing. By default all fragments are '
                             'considered, that may cause combinatorial explosion in some cases.')

    group4 = parser.add_argument_group("Physicochemical properties")
    group4.add_argument('--mw', metavar='NUMERIC', required=False, type=float, default=450,
                        help='Maximum molecular weight of generated compounds.')
    group4.add_argument('--tpsa', metavar='NUMERIC', required=False, type=float, default=120,
                        help='Maximum TPSA of generated compounds.')
    group4.add_argument('--rtb', metavar='NUMERIC', required=False, type=float, default=7,
                        help='Maximum number of rotatable bonds in generated compounds.')
    group4.add_argument('--logp', metavar='NUMERIC', required=False, type=float, default=4,
                        help='Maximum logP of generated compounds.')

    group5 = parser.add_argument_group("Running settings")
    group5.add_argument('-u', '--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    group5.add_argument('-w', '--num_workers', metavar='INTEGER', required=False, type=int, default=1,
                        help='the number of workers to be spawn by the dask cluster. This will limit the maximum '
                             'number of processed molecules simultaneously.')
    group5.add_argument('--log', metavar='FILENAME', required=False, type=str, default=None,
                        help='log file to collect progress and debug messages. If omitted, the log file with the same '
                             'name as output DB will be created.')
    group5.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use per molecule.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    if args.log:
        if not os.path.exists(os.path.dirname(args.log)):
            os.makedirs(os.path.dirname(args.log))
        logging.basicConfig(filename=args.log, filemode='a', level=logging.DEBUG,
                            encoding='utf-8', datefmt='%Y-%m-%d %H:%M:%S',
                            format='[%(asctime)s] %(levelname)s: (PID:%(process)d) %(message)s')

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
    try:
        p.load_from_xyz(args.query)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    p.set_clusters(args.clustering_threshold, args.ids)
    if args.log:
        logging.info('Pharmacophore model was parsed.')

    print(p.clusters, flush=True)

    res_db_fname = os.path.join(args.output, 'res.db')

    if not os.path.isfile(res_db_fname):  # create DB and match starting fragments

        create_db(res_db_fname)
        if args.log:
            logging.info('Empty output database was created.')

        conf_fname = os.path.join(args.output, f'iter0.sdf')

        new_pids = tuple(args.ids)

        print(f"===== Initial screening =====")
        start = timeit.default_timer()

        if args.log:
            logging.info('Screening of starting fragments was started.')
        mols = screen_pmapper(query_pharm=p.get_subpharmacophore(new_pids), db_fname=args.fragments,
                              output_sdf=conf_fname, rmsd_to_query=min(0.25, args.dist),
                              ncpu=min(args.ncpu * args.num_workers, cpu_count()),
                              exclvol_xyz=p.exclvol, exclvol_dist=args.exclusion_volume)
        if args.log:
            logging.info(f'Screening of starting fragments was finished. Found fragments: {len(mols)}.')
        if not mols:
            if args.log:
                logging.info('Program finished.')
            exit('Program finished. No matches between starting fragments and the chosen subpharmacophore.')

        print(f'screening: {round(timeit.default_timer() - start, 4)}')
        start = timeit.default_timer()
        c = Counter((m.GetNumHeavyAtoms() for m in mols))
        print(f'select_mols: from {len(mols)} molecules')
        print(sorted(c.items()))

        mols = select_mols(mols, ncpu=min(args.ncpu * args.num_workers, cpu_count()))
        ids = ','.join(map(str, new_pids))
        for mol in mols:
            mol.SetProp('visited_ids', ids)
            for conf in mol.GetConformers():
                conf.SetProp('visited_ids', ids)
                conf.SetProp('matched_ids', ids)
                conf.SetProp('parent_conf_id', 'None')
        save_res(mols, None, res_db_fname)
        if args.log:
            logging.info(f'Non-redundant starting fragments were stored to the database: {len(mols)} fragments.')

        print(f'select_mols: {round(timeit.default_timer() - start, 4)}')

    else:  # set all processing flags to 0 (for restart)
        with sqlite3.connect(res_db_fname) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE mols SET processing = 0, processing_nmols = 0")
            conn.commit()
        if args.log:
            logging.info(f'The output database exists. Attempt to continue generation.')

    pharm_fd, pharm_fname = tempfile.mkstemp(suffix='_pharm.pkl', text=True)
    with open(pharm_fname, 'wb') as f:
        pickle.dump(p, f)

    # whether crem db contains pharmacophore feature count
    additional_features = test_additional_features(args.db, args.radius)

    config_fd, config_fname = tempfile.mkstemp(suffix='_config.yml', text=True)
    with open(config_fname, 'wt') as f:
        yaml.safe_dump({'additional_features': additional_features,
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
            m = choose_mol_to_grow(res_db_fname, p.get_num_features())
            if m:
                futures.append(dask_client.submit(expand_mol_cli, m, pharm_fname=pharm_fname, config_fname=config_fname))
        seq = as_completed(futures, with_results=True)
        for i, (future, (parent_mol_id, new_mols, nmols, debug)) in enumerate(seq, 1):
            new_mol_ids = save_res(new_mols, parent_mol_id, res_db_fname)
            update_db(res_db_fname, parent_mol_id, 'processing_nmols', -1)
            if nmols:
                update_db(res_db_fname, parent_mol_id, 'nmols', nmols)
            if args.log:
                logging.info(f'{get_stat_string_from_db(res_db_fname)}')
                # logging.debug(f'===== {parent_mol_id} =====\n' + debug)
            if debug:
                print(f'===== {parent_mol_id} =====')
                print(debug)
                sys.stdout.flush()
            del future
            for _ in range(max_tasks - seq.count()):
                m = choose_mol_to_grow(res_db_fname, p.get_num_features(), mol_ids=new_mol_ids)
                new_mol_ids = None  # select only one mol to search deep
                if m:
                    new_future = dask_client.submit(expand_mol_cli, m, pharm_fname=pharm_fname, config_fname=config_fname)
                    seq.add(new_future)

    finally:

        os.close(pharm_fd)
        os.close(config_fd)
        os.unlink(pharm_fname)
        os.unlink(config_fname)


if __name__ == '__main__':
    entry_point()
