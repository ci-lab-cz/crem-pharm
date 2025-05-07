#!/usr/bin/env python3

import argparse
import sys
import sqlite3
from multiprocessing import Pool
from rdkit import Chem
from pmapper.pharmacophore import Pharmacophore as P
from pmapper.customize import load_smarts


def calc(items):
    core_id, core_smi = items
    m = Chem.MolFromSmiles(core_smi)
    if m:
        p = P()
        q = p._get_features_atom_ids(m, smarts)
        q = ','.join(f'n{k if k != "a" else "Ar"} = {len(v)}' for k, v in q.items())  # nA = 5, nD = 2, nAr = 1
                                                                                      # nAr - number of aromatic features
                                                                                      # used due to case-insensistive
                                                                                      # column names in SQLite
        return core_id, q


def init():
    global smarts
    smarts = load_smarts()


def entry_point():
    parser = argparse.ArgumentParser(description='Label fragments in CReM DB with pharmacophore feature counts '
                                                 'using pmapper.')
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='CReM fragment database.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, default=1, type=int,
                        help='number of CPU cores to use.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress to STDERR.')

    args = parser.parse_args()

    pool = Pool(args.ncpu, initializer=init)

    col_names = ['nA', 'nD', 'nH', 'nAr', 'nN', 'nP']

    with sqlite3.connect(args.input) as conn:
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'frags'")
        if not cur.fetchone():
            sys.stderr.write('Table frags is absent in the input database. '
                             'Individual radius tables will be altered.')
            cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'radius%'")
            tables = cur.fetchall()
            tables = [i[0] for i in tables]
            frags_table = False
        else:
            tables = ['frags']
            frags_table = True

        # create columns
        for table in tables:
            for col_name in col_names:
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} INTEGER DEFAULT NULL")
                except sqlite3.OperationalError as e:
                    sys.stderr.write(str(e) + '\n')
            conn.commit()

        if frags_table:

            cur.execute(f"SELECT core_id, core_smi FROM frags WHERE " +
                        " AND ".join([f"{col} IS NULL" for col in col_names]))
            res = cur.fetchall()
            for i, (core_id, upd_str) in enumerate(pool.imap_unordered(calc, res), 1):
                cur.execute(f"UPDATE frags SET {upd_str} WHERE core_id = {core_id}")
                if i % 10000 == 0:
                    conn.commit()
                    if args.verbose:
                        sys.stderr.write(f'\r{i} fragments processed')
            conn.commit()

        else:

            for table in tables:
                sys.stderr.write(f'\nTable {table}\n')
                cur.execute(f"SELECT rowid, core_smi FROM {table} WHERE " +
                            " AND ".join([f"{col} IS NULL" for col in col_names]))
                res = cur.fetchall()
                for i, (rowid, upd_str) in enumerate(pool.imap_unordered(calc, res), 1):
                    cur.execute(f"UPDATE {table} SET {upd_str} WHERE rowid = '{rowid}'")
                    if i % 10000 == 0:
                        conn.commit()
                        if args.verbose:
                            sys.stderr.write(f'\r{i} fragments processed')
                conn.commit()

        for table in tables:
            for col_name in col_names:
                cur.execute(f'UPDATE {table} SET {col_name} = 0 WHERE {col_name} IS NULL')
            conn.commit()

    sys.stderr.write(f'\nFinished processing of {args.input}\n')


if __name__ == '__main__':
    entry_point()
