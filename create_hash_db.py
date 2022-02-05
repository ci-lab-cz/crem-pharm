#!/usr/bin/env python3

import argparse
import sqlite3
from functools import partial
from itertools import combinations
from multiprocessing import Pool

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


def read_smi(fname):
    with open(fname) as f:
        for line in f:
            smi = line.strip().split()[0]
            yield smi


def process_smi(smi, nconf, seed, binstep, min_features, max_features):
    mol = gen_confs(mol=Chem.MolFromSmiles(smi), nconf=nconf, seed=seed)
    hashes = gen_hashes(mol=mol, binstep=binstep, min_features=min_features, max_features=max_features)
    return smi, hashes


def gen_confs(mol, nconf, seed):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=nconf * 4, randomSeed=seed)
    return mol


def gen_hashes(mol, binstep, min_features, max_features):
    """
    Attachment point feature will be always present in output combination. Thus, min_features is the minimum number of
    added other features

    :param mol:
    :param binstep:
    :param min_features:
    :param max_features:
    :return:
    """
    hashes = []
    p = P(cached=True, bin_step=binstep)
    atom_ids = p._get_features_atom_ids(mol, smarts)
    for conf in mol.GetConformers():
        p.load_from_atom_ids(mol, atom_ids, conf.GetId())
        feature_ids = p.get_feature_ids()
        att_id = feature_ids['T'][0]
        other_ids = []
        for k, v in feature_ids.items():
            if k != 'T':
                other_ids.extend(v)
        max_features = min(max_features, len(other_ids))
        for n in range(min_features, max_features + 1):
            for comb in combinations(other_ids, n):
                hashes.append(p.get_signature_md5(ids=[att_id] + list(comb)))
    return set(hashes)


def main():
    parser = argparse.ArgumentParser(description='Generate database of 3D pharmacophore hashes of fragments.')
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='SMILES files. No header.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True,
                        help='SQLite3 DB file.')
    parser.add_argument('-b', '--binstep', metavar='NUMERIC', required=False, type=float, default=1.5,
                        help='binning step to generate 3D pharmacophore hashes.')
    parser.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=50,
                        help='number of conformers generated for each input fragment.')
    parser.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=0,
                        help='random seed. Default: 0.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use. Default: 1.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    pool = Pool(args.ncpu)

    create_db(args.output)
    con = sqlite3.connect(args.output)
    cur = con.cursor()

    for i, (smi, hashes) in enumerate(pool.imap_unordered(partial(process_smi,
                                                                  nconf=args.nconf,
                                                                  seed=args.seed,
                                                                  binstep=args.binstep,
                                                                  min_features=1,
                                                                  max_features=5),
                                                          read_smi(args.input)), 1):
        smi_id = list(cur.execute("INSERT OR IGNORE INTO frags(smi) VALUES(?) RETURNING id", (smi, )))[0][0]
        cur.executemany("INSERT INTO hashes(id, hash) VALUES(?, ?)", [(smi_id, h)for h in hashes])
        con.commit()


if __name__ == '__main__':
    main()
