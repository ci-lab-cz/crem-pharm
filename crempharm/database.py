import sqlite3

from rdkit import Chem


def create_db(db_fname):
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS mols")
        cur.execute("CREATE TABLE mols("
                    "id INTEGER NOT NULL, "
                    "conf_id INTEGER NOT NULL, "
                    "smi TEXT NOT NULL, "
                    "mol_block TEXT NOT NULL, "
                    "matched_ids TEXT NOT NULL, "
                    "visited_ids TEXT NOT NULL, "
                    "matched_ids_count INTEGER NOT NULL, "
                    "visited_ids_count INTEGER NOT NULL, "
                    "parent_mol_id INTEGER, "
                    "parent_conf_id INTEGER, "
                    "nmols INTEGER NOT NULL, "
                    "processing_nmols INTEGER NOT NULL, "
                    "used INTEGER NOT NULL, "
                    "processing INTEGER NOT NULL, "
                    "priority INTEGER NOT NULL, "
                    "time TEXT NOT NULL)")
        conn.commit()


def save_res(mols, parent_mol_id, db_fname):

    output = []

    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()

        if parent_mol_id is not None:
            cur.execute('UPDATE mols SET used = 1, processing = 0 WHERE id = ?',
                        (parent_mol_id, ))
            conn.commit()

        cur.execute('SELECT MAX(id) FROM mols')
        mol_id = cur.fetchone()[0]
        if mol_id is None:
            mol_id = 0

        # parent_mol_id = mols[0].GetPropsAsDict().get('parent_mol_id', None)
        if parent_mol_id is not None:
            cur.execute(f'SELECT distinct(priority) FROM mols where id = {parent_mol_id}')
            parent_priority = cur.fetchone()[0]
            priority = parent_priority + 1
        else:
            priority = 1

        for mol in mols:
            # parent_mol_id = mol.GetPropsAsDict().get('parent_mol_id', None)
            mol_id += 1
            output.append(mol_id)
            mol.SetProp('_Name', str(mol_id))
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            for conf in mol.GetConformers():
                mol_block = Chem.MolToMolBlock(mol, confId=conf.GetId())
                visited_ids = conf.GetProp('visited_ids')
                visited_ids_count = visited_ids.count(',') + 1
                matched_ids = conf.GetProp('matched_ids')
                matched_ids_count = matched_ids.count(',') + 1
                parent_conf_id = conf.GetProp('parent_conf_id')
                if parent_conf_id == 'None':
                    parent_conf_id = None
                sql = 'INSERT INTO mols (id, conf_id, smi, mol_block, matched_ids, visited_ids, ' \
                      '                  matched_ids_count, visited_ids_count, parent_mol_id, ' \
                      '                  parent_conf_id, priority, used, processing, nmols, processing_nmols, time) ' \
                      'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)'
                cur.execute(sql, (mol_id, conf.GetId(), smi, mol_block, matched_ids, visited_ids,
                                  matched_ids_count, visited_ids_count, parent_mol_id, parent_conf_id, priority,
                                  0, 0, 0, 0))
                conn.commit()
            priority += 3

    return output


def update_db(db_fname, mol_id, field, value):
    """
    Recursively update field values from bottom to top parent
    :param db_fname:
    :param mol_id: id of a starting molecule to update all its parents
    :param field: field name, normally nmols and processing_nmols
    :param value: increment value, can be negative for processing_nmols
    :return:
    """
    with sqlite3.connect(db_fname) as conn:
        while mol_id is not None:
            cur = conn.cursor()
            cur.execute('SELECT %s FROM mols WHERE id = %i' % (field, mol_id))
            n = cur.fetchone()[0] + value
            cur.execute('UPDATE mols SET %s = %i WHERE id = %i' % (field, n, mol_id))
            conn.commit()
            cur.execute('SELECT parent_mol_id FROM mols WHERE id = %i' % (mol_id, ))
            mol_id = cur.fetchone()[0]


def get_stat_string_from_db(db_fname):
    with sqlite3.connect(db_fname) as conn:
        cur = conn.cursor()
        # nmols_embedded_3d = sum(cur.execute('SELECT min(nmols) FROM mols WHERE parent_mol_id IS NULL GROUP BY id').fetchall())
        nmol_stored = sum(cur.execute('SELECT COUNT(DISTINCT id) FROM mols').fetchone())
        # cur.execute('SELECT c, count(rowid) FROM '
        #             '(SELECT id, max(matched_ids_count) as c FROM mols GROUP BY id)')
        # nmol_stored = cur.fetchall()
        nmol_not_expaded = sum(cur.execute('SELECT COUNT(DISTINCT id) FROM mols WHERE used = 0').fetchone())
        output = (f'Total number of mols found: {nmol_stored}, '
                  f'number of remained not expanded mols: {nmol_not_expaded}')
    return output
