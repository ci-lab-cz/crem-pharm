#!/usr/bin/env python3
"""
Label fragments in a CReM database with pharmacophore feature counts using pmapper.

Supports:
  - Schema version 1 (new): frags table with core_smi_id (detected via PRAGMA user_version = 1)
  - Old schema: frags table with core_id, or individual radiusX tables

Features are stored as integer columns nA, nD, nH, nAr, nN, nP in the target table.
"""

import argparse
import sqlite3
import sys
from multiprocessing import Pool

from rdkit import Chem
from pmapper.pharmacophore import Pharmacophore as P
from pmapper.customize import load_smarts


COL_NAMES = ['nA', 'nD', 'nH', 'nAr', 'nN', 'nP']

# Mapping from column name to pmapper feature key
_FEAT_KEY = {'nA': 'A', 'nD': 'D', 'nH': 'H', 'nAr': 'a', 'nN': 'N', 'nP': 'P'}

# Tuning constants
_FETCH_BATCH = 50000   # rows fetched from DB at a time
_WRITE_BATCH = 10000   # rows written per executemany
_IMAP_CHUNK = 500      # task chunksize for pool.imap_unordered


def _init_worker():
    global _smarts
    _smarts = load_smarts()


def _calc(item):
    row_id, core_smi = item
    m = Chem.MolFromSmiles(core_smi)
    if m is None:
        return row_id, [0] * len(COL_NAMES)
    p = P()
    feat = p._get_features_atom_ids(m, _smarts)
    return row_id, [len(feat.get(_FEAT_KEY[c], [])) for c in COL_NAMES]


def _add_columns(conn, table):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for col in COL_NAMES:
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER DEFAULT NULL")
    conn.commit()


def _process_table(conn, pool, table, id_col, smi_col, verbose, fetch_batch, write_batch, imap_chunk):
    null_filter = " AND ".join(f"{c} IS NULL" for c in COL_NAMES)

    total = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {null_filter}").fetchone()[0]
    if total == 0:
        if verbose:
            sys.stderr.write(f"  No unlabeled rows in {table}\n")
        return

    if verbose:
        sys.stderr.write(f"  {total} rows to label in {table}\n")

    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, {smi_col} FROM {table} WHERE {null_filter}")

    update_sql = (
        f"UPDATE {table} SET "
        + ", ".join(f"{c} = ?" for c in COL_NAMES)
        + f" WHERE {id_col} = ?"
    )

    def _stream():
        while True:
            rows = cur.fetchmany(fetch_batch)
            if not rows:
                break
            yield from rows

    batch = []
    processed = 0

    for row_id, vals in pool.imap_unordered(_calc, _stream(), chunksize=imap_chunk):
        batch.append((*vals, row_id))
        if len(batch) >= write_batch:
            conn.executemany(update_sql, batch)
            conn.commit()
            processed += len(batch)
            batch = []
            if verbose:
                sys.stderr.write(f"\r  {processed}/{total}")
                sys.stderr.flush()

    if batch:
        conn.executemany(update_sql, batch)
        conn.commit()
        processed += len(batch)

    if verbose:
        sys.stderr.write(f"\r  {processed}/{total}\n")


def entry_point():
    parser = argparse.ArgumentParser(
        description='Label fragments in a CReM database with pharmacophore feature counts using pmapper.'
    )
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='CReM fragment database.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', default=1, type=int,
                        help='Number of CPU cores to use (default: 1).')
    parser.add_argument('--fetch-batch', metavar='INTEGER', default=_FETCH_BATCH, type=int,
                        help=f'Rows fetched from DB per batch (default: {_FETCH_BATCH}).')
    parser.add_argument('--write-batch', metavar='INTEGER', default=_WRITE_BATCH, type=int,
                        help=f'Rows per DB write batch (default: {_WRITE_BATCH}).')
    parser.add_argument('--imap-chunk', metavar='INTEGER', default=_IMAP_CHUNK, type=int,
                        help=f'Task chunksize for multiprocessing imap (default: {_IMAP_CHUNK}).')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print progress to STDERR.')
    args = parser.parse_args()

    pool = Pool(args.ncpu, initializer=_init_worker)

    with sqlite3.connect(args.input) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -65536")

        version = conn.execute("PRAGMA user_version").fetchone()[0]

        if version == 1:
            # New schema: frags table with core_smi_id as PK
            if args.verbose:
                sys.stderr.write("Schema version 1: labeling frags table\n")
            _add_columns(conn, 'frags')
            _process_table(
                conn, pool, 'frags', 'core_smi_id', 'core_smi',
                args.verbose, args.fetch_batch, args.write_batch, args.imap_chunk,
            )

        else:
            # Old schema: frags with core_id, or individual radiusX tables
            has_frags = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='frags'"
            ).fetchone() is not None

            if has_frags:
                if args.verbose:
                    sys.stderr.write("Old schema: labeling frags table\n")
                _add_columns(conn, 'frags')
                _process_table(
                    conn, pool, 'frags', 'core_id', 'core_smi',
                    args.verbose, args.fetch_batch, args.write_batch, args.imap_chunk,
                )
            else:
                tables = [
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'radius%'"
                    )
                ]
                if args.verbose:
                    sys.stderr.write(f"Old schema: labeling radius tables: {tables}\n")
                for table in tables:
                    if args.verbose:
                        sys.stderr.write(f"\nTable {table}\n")
                    _add_columns(conn, table)
                    _process_table(
                        conn, pool, table, 'rowid', 'core_smi',
                        args.verbose, args.fetch_batch, args.write_batch, args.imap_chunk,
                    )

        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    pool.close()
    pool.join()

    sys.stderr.write(f"\nFinished: {args.input}\n")


if __name__ == '__main__':
    entry_point()