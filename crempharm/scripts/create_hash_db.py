#!/usr/bin/env python3

import argparse
import sqlite3
from functools import partial
from itertools import combinations, chain
from multiprocessing import Pool
from os import path

from pmapper.utils import load_multi_conf_mol
from pmapper.customize import load_smarts
from rdkit import Chem
from rdkit.Chem import AllChem

from old_scripts.read_input import read_input

smarts = load_smarts(path.join(path.dirname(path.realpath(__file__)), 'smarts_features.txt'))


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
        for mol, title in read_input(fname, sdf_confs=fname.endswith('.sdf')):
            if not con.execute("SELECT EXISTS(SELECT 1 FROM frags WHERE smi = ?)", (title, )).fetchone()[0]:
                mol = Chem.RWMol(mol)
                for a in mol.GetAtoms():
                    if a.GetAtomicNum() == 0:
                        a.SetAtomicNum(9)
                        a.SetIsotope(50)
                mol = Chem.Mol(mol)
                yield mol


def process_mol(mol, nconf, seed, binstep, tolerance=0, min_features=1, max_features=6):
    if len(mol.GetConformers()) <= 1:
        mol = gen_confs(mol=mol, nconf=nconf, seed=seed)
    hashes = gen_hashes(mol=mol, binstep=binstep, tolerance=tolerance, min_features=min_features, max_features=max_features)
    return mol.GetProp("_Name"), hashes


def gen_confs(mol, nconf, seed):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=nconf * 4, randomSeed=seed)
    return mol


def gen_hashes(mol, binstep, tolerance=0, min_features=1, max_features=6):
    """
    Attachment point features will be always present in output combination. Thus, min_features is the minimum number of
    added other features, max_features is correspondingly the maximum number of other features.
    Attachment point is represented by two features to simulate its direction. Feature of the attachment point itself is
    encoded by label T, feature designating the end of the attachment point is labeled Q.

    :param mol: multi-conformer Mol
    :param binstep:
    :param tolerance:
    :param min_features:
    :param max_features:
    :return:
    """
    hashes = []

    for p in load_multi_conf_mol(mol, smarts_features=smarts, bin_step=binstep, cached=True):

        feature_ids = p.get_feature_ids()
        att_id = feature_ids['T'][0]
        del feature_ids['T']
        other_ids = list(sorted(chain.from_iterable(feature_ids.values())))

        if min_features > len(other_ids):
            return tuple()

        max_features_ = min(max_features, len(other_ids))
        for n in range(min_features, max_features_ + 1):
            for comb in combinations(other_ids, n):
                hashes.append(p.get_signature_md5(ids=[att_id] + list(comb), tol=tolerance))

    return tuple(set(hashes))


def main():
    parser = argparse.ArgumentParser(description='Generate database of 3D pharmacophore hashes of fragments having '
                                                 'one attachment point.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='SMILES files (no header) or SDF (may contain multiple conformers going consecutively).')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True,
                        help='SQLite3 DB file.')
    parser.add_argument('-b', '--binstep', metavar='NUMERIC', required=False, type=float, default=1,
                        help='binning step to generate 3D pharmacophore hashes.')
    parser.add_argument('-tol', '--tolerance', metavar='NUMERIC', type=int, default=0,
                        help='tolerance used for calculation of a stereoconfiguration sign')
    parser.add_argument('-n', '--nconf', metavar='INTEGER', required=False, type=int, default=50,
                        help='number of conformers generated for each input fragment.')
    parser.add_argument('-s', '--seed', metavar='INTEGER', required=False, type=int, default=0,
                        help='random seed.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, type=int, default=1,
                        help='number of cpu cores to use.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parser.parse_args()

    pool = Pool(args.ncpu)

    if not path.isfile(args.output):
        create_db(args.output)

    con = sqlite3.connect(args.output)
    cur = con.cursor()

    for i, (smi, hashes) in enumerate(pool.imap_unordered(partial(process_mol,
                                                                  nconf=args.nconf,
                                                                  seed=args.seed,
                                                                  binstep=args.binstep,
                                                                  tolerance=args.tolerance,
                                                                  min_features=1,
                                                                  max_features=6),
                                                          read_smi(args.input, args.output)), 1):
        if not hashes:
            continue
        try:
            smi_id = list(cur.execute("INSERT OR IGNORE INTO frags(smi) VALUES(?) RETURNING id", (smi, )))[0][0]
            cur.executemany("INSERT INTO hashes(id, hash) VALUES(?, ?)", [(smi_id, h) for h in hashes])
            con.commit()
        except IndexError:
            print(i, smi, len(hashes))
            print('-')

    sql = "CREATE INDEX hashes_hash_idx ON hashes(hash)"
    con.execute(sql)
    con.commit()


if __name__ == '__main__':
    main()
