#!/usr/bin/env python3

import argparse
import sqlite3
import sys


def entry_point():
    parser = argparse.ArgumentParser(description='Extract a sequence of parent molecules from DB starting for '
                                                 'the given id and conf_id.')
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='CReM-pharm database.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True,
                        help='SDF file.')
    parser.add_argument('-m', '--mol_id', metavar='INTEGER', required=True, type=int,
                        help='id of a starting molecule')
    parser.add_argument('-c', '--conf_id', metavar='INTEGER', required=False, type=int, default=0,
                        help='conf_id of a starting molecule. Default: 0.')

    args = parser.parse_args()

    with sqlite3.connect(args.input) as conn:
        with open(args.output, 'w') as out:

            cur = conn.cursor()

            sql = """
            WITH RECURSIVE parent_chain AS (
                SELECT id, conf_id, mol_block, parent_mol_id, parent_conf_id, 0 AS level
                FROM mols
                WHERE id = ? AND conf_id = ?
                UNION ALL
                SELECT m.id, m.conf_id, m.mol_block, m.parent_mol_id, m.parent_conf_id, pc.level + 1
                FROM mols m
                JOIN parent_chain pc
                    ON m.id = pc.parent_mol_id AND m.conf_id = pc.parent_conf_id
            )
            SELECT mol_block
            FROM parent_chain
            ORDER BY level
            """

            cur.execute(sql, (args.mol_id, args.conf_id))
            i = 0
            for mol_block in cur.fetchall():
                out.write(mol_block[0] + '$$$$\n')
                i += 1
    sys.stderr.write(f'{i} molecules were extracted\n')


if __name__ == '__main__':
    entry_point()
