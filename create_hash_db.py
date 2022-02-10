#!/usr/bin/env python3

import argparse
import os
import sqlite3
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from pmapper.pharmacophore import Pharmacophore as P
from pmapper.customize import load_smarts
from rdkit import Chem
from rdkit.Chem import AllChem


smarts = load_smarts()
smarts["T"] = (Chem.MolFromSmarts('[#0]'),)


def create_db(fname):
    con = sqlite3.connect(fname)
    con.execute("CREATE TABLE IF NOT EXISTS frags("
                "id INTEGER PRIMARY KEY, "
                "smi TEXT NOT NULL UNIQUE)")
    con.execute("CREATE TABLE IF NOT EXISTS hashes("
                "id INTEGER NOT NULL, "
                "hash TEXT NOT NULL, "
                "FOREIGN KEY (id) REFERENCES frags (id))")
    con.commit()
    con.close()


def read_smi(fname, dbname):
    with sqlite3.connect(dbname) as con:
        with open(fname) as f:
            for line in f:
                smi = line.strip().split()[0]
                if not con.execute("SELECT EXISTS(SELECT 1 FROM frags WHERE smi = ?)", (smi, )).fetchone()[0]:
                    yield smi


def process_smi(smi, nconf, seed, binstep, min_features, max_features, directed):
    mol = gen_confs(mol=Chem.MolFromSmiles(smi), nconf=nconf, seed=seed)
    hashes = gen_hashes(mol=mol, binstep=binstep, min_features=min_features, max_features=max_features, directed=directed)
    return smi, hashes


def gen_confs(mol, nconf, seed):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=nconf * 4, randomSeed=seed)
    return mol


def get_complementary_point_xyz(mol, conf_id, start_atom_id, binstep):
    """
    Find coordinates of a point on a distance of a bin step in the direction of the neighboring atom to
    the attachment point.
    :param mol:
    :param conf_id:
    :param start_atom_id: id of the attachment point (atom with atomic number 0)
    :param binstep:
    :return: tuple of xyz coordinates
    """
    end_atom_id = mol.GetAtomWithIdx(start_atom_id).GetNeighbors()[0].GetIdx()
    conf = mol.GetConformer(conf_id)
    a = conf.GetAtomPosition(start_atom_id)
    b = conf.GetAtomPosition(end_atom_id)
    output_pos = a + (b - a) / np.linalg.norm(a - b) * binstep
    return tuple(output_pos)


def gen_hashes(mol, binstep, min_features, max_features, directed):
    """
    Attachment point features will be always present in output combination. Thus, min_features is the minimum number of
    added other features, max_features is correspondingly the maximum number of other features.
    Attachment point is represented by two features to simulate its direction. Feature of the attachment point itself is
    encoded by label T, feature designating the end of the attachment point is labeled Q.

    :param mol: multi-conformer Mol
    :param binstep:
    :param min_features:
    :param max_features:
    :param directed: False if attachment point is encoded by undirected feature and True if it should be encoded
                     by a directed feature (a combination of starting and ending features)
    :return:
    """
    hashes = []
    p = P(cached=True, bin_step=binstep)
    atom_ids = p._get_features_atom_ids(mol, smarts)

    for conf in mol.GetConformers():

        conf_id = conf.GetId()
        p.load_from_atom_ids(mol, atom_ids, conf_id)

        feature_ids = p.get_feature_ids()
        att_id = feature_ids['T'][0]
        other_ids = []
        for k, v in feature_ids.items():
            if k != 'T':
                other_ids.extend(v)

        if directed:
            # determine coordinates of the end of the attachment point feature and add it to pharmacophore
            end_xyz = get_complementary_point_xyz(mol, conf_id, att_id, binstep)
            end_id = p.add_feature('Q', end_xyz)
            fixed_ids = [att_id, end_id]
        else:
            fixed_ids = [att_id]

        max_features = min(max_features, len(other_ids))
        for n in range(min_features, max_features + 1):
            for comb in combinations(other_ids, n):
                hashes.append(p.get_signature_md5(ids=fixed_ids + list(comb)))

    return tuple(set(hashes))


def main():
    parser = argparse.ArgumentParser(description='Generate database of 3D pharmacophore hashes of fragments having '
                                                 'one attachment point.')
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='SMILES files. No header.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True,
                        help='SQLite3 DB file.')
    parser.add_argument('-b', '--binstep', metavar='NUMERIC', required=False, type=float, default=1.5,
                        help='binning step to generate 3D pharmacophore hashes.')
    parser.add_argument('-d', '--directed', required=False, action='store_true', default=False,
                        help='if set attachment points will be considered as directed features.')
    parser.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=50,
                        help='number of conformers generated for each input fragment.')
    parser.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=0,
                        help='random seed. Default: 0.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use. Default: 1.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    pool = Pool(args.ncpu)

    if not os.path.isfile(args.output):
        create_db(args.output)

    con = sqlite3.connect(args.output)
    cur = con.cursor()

    for i, (smi, hashes) in enumerate(pool.imap_unordered(partial(process_smi,
                                                                  nconf=args.nconf,
                                                                  seed=args.seed,
                                                                  binstep=args.binstep,
                                                                  min_features=1,
                                                                  max_features=5,
                                                                  directed=args.directed),
                                                          read_smi(args.input, args.output)), 1):
        smi_id = list(cur.execute("INSERT OR IGNORE INTO frags(smi) VALUES(?) RETURNING id", (smi, )))[0][0]
        cur.executemany("INSERT INTO hashes(id, hash) VALUES(?, ?)", [(smi_id, h)for h in hashes])
        con.commit()

    sql = "CREATE INDEX hashes_hash_idx ON hashes(hash)"
    con.execute(sql)
    con.commit()


if __name__ == '__main__':
    main()
