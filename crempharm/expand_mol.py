#!/usr/bin/env python3

import argparse
import os
import sys
import platform
import pickle
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
from itertools import combinations, product
from operator import itemgetter
import numpy as np
import pandas as pd
import re
import sqlite3
import subprocess
import tempfile
import timeit
import traceback
import yaml

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.AllChem import GetConformerRMSMatrix
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Geometry.rdGeometry import Point3D
from crem.crem import grow_mol
from pmapper.utils import load_multi_conf_mol
from pmapper.pharmacophore import Pharmacophore as P
from openbabel import openbabel as ob
from openbabel import pybel
from sklearn.cluster import AgglomerativeClustering


def gen_stereo(mol):
    stereo_opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=32)
    for b in mol.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOANY:
            b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    isomers = tuple(EnumerateStereoisomers(mol, options=stereo_opts))
    return isomers


def ConstrainedEmbedMultipleConfs(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342, **kwargs):

    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    cids = list(cids)
    if len(cids) == 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        for cid in cids:
            ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            for i, idxI in enumerate(match):
                for j in range(i + 1, len(match)):
                    idxJ = match[j]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.)
            ff.Initialize()
            n = 4
            more = ff.Minimize()
            while more and n:
                more = ff.Minimize()
                n -= 1
            # rotate the embedded conformation onto the core:
            rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        for cid in cids:
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
            ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            conf = core.GetConformer()
            for i in range(core.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
            ff.Initialize()
            n = 4
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            while more and n:
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                n -= 1
            # realign
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    return mol


def __embed_multiple_confs_constrained_ob(mol, template_mol, nconf, seed=42, **kwargs):
    # generate constrained conformers using OpenBabel, if fails use RDKit
    # **kwargs arguments are for RDKit generator

    mol = AllChem.ConstrainedEmbed(Chem.AddHs(mol), template_mol, randomseed=seed)
    ids = set(range(mol.GetNumAtoms())) - set(mol.GetSubstructMatch(template_mol))
    ids = [i + 1 for i in ids]   # ids of atoms to rotate
    mol_rdkit = mol

    mol = pybel.readstring('mol', Chem.MolToMolBlock(mol)).OBMol   # convert mol from RDKit to OB

    pff = ob.OBForceField_FindType("mmff94")
    if not pff.Setup(mol):  # if OB FF setup fails use RDKit conformer generation (slower)
        return ConstrainedEmbedMultipleConfs(Chem.AddHs(mol_rdkit), Chem.RemoveHs(template_mol), nconf, randomseed=seed, **kwargs)

    constraints = ob.OBFFConstraints()
    for atom in ob.OBMolAtomIter(mol):
        atom_id = atom.GetIndex() + 1
        if atom_id not in ids:
            constraints.AddAtomConstraint(atom_id)
    pff.SetConstraints(constraints)

    pff.DiverseConfGen(0.5, 1000, 50, False)   # rmsd, nconf_tries, energy, verbose

    pff.GetConformers(mol)
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('mol')

    output_strings = []
    for conf_num in range(max(0, mol.NumConformers() - nconf), mol.NumConformers()):   # save last nconf conformers (it seems the last one is the original conformer)
        mol.SetConformer(conf_num)
        output_strings.append(obconversion.WriteString(mol))

    out_mol = Chem.MolFromMolBlock(output_strings[0])
    for a in output_strings[1:]:
        out_mol.AddConformer(Chem.MolFromMolBlock(a).GetConformer(0), assignId=True)
    return out_mol


def __gen_confs(mol, template_mol=None, nconf=10, seed=42, alg='rdkit', **kwargs):
    # alg - 'rdkit' or 'ob'
    try:
        if template_mol:
            if alg == 'rdkit':
                mol = ConstrainedEmbedMultipleConfs(Chem.AddHs(mol), Chem.RemoveHs(template_mol), nconf, randomseed=seed, **kwargs)
            elif alg == 'ob':
                mol = __embed_multiple_confs_constrained_ob(mol, template_mol, nconf, seed, **kwargs)
        else:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=nconf*4, randomSeed=seed)
    except ValueError:
        return None
    return mol


def __get_grow_points2(mol_xyz, pharm_xyz, tol=2):
    dist = np.min(cdist(mol_xyz, pharm_xyz), axis=1)
    ids = np.flatnonzero(dist <= (np.min(dist) + tol))
    return ids.tolist()


def get_grow_atom_ids(mol, pharm_xyz, tol=2):

    # search for the closest growing points among heavy atoms with attached hydrogens only

    ids = set()
    atoms_with_h = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1 and a.GetTotalNumHs() > 0:
            atoms_with_h.append(a.GetIdx())
    for c in mol.GetConformers():
        ids.update(__get_grow_points2(c.GetPositions()[atoms_with_h], pharm_xyz, tol))
    res = list(sorted(atoms_with_h[i] for i in ids))
    return res


def remove_confs_rms(mol, rms=0.25, keep_nconf=None):
    """
    The function uses AgglomerativeClustering to select conformers.

    :param mol: input molecule with multiple conformers
    :param rms: discard conformers which are closer than given value to a kept conformer
    :param keep_nconf: keep at most the given number of conformers. This parameter has precedence over rms
    :return:
    """

    def gen_ids(ids):
        for i in range(1, len(ids)):
            for j in range(0, i):
                yield j, i

    if keep_nconf and mol.GetNumConformers() <= keep_nconf:
        return mol

    if mol.GetNumConformers() <= 1:
        return mol

    mol_tmp = Chem.RemoveHs(mol)   # calc rms for heavy atoms only
    rms_ = GetConformerRMSMatrix(mol_tmp, prealigned=True)

    cids = [c.GetId() for c in mol_tmp.GetConformers()]
    arr = np.zeros((len(cids), len(cids)))
    for (i, j), v in zip(gen_ids(cids), rms_):
        arr[i, j] = v
        arr[j, i] = v
    if keep_nconf:
        cl = AgglomerativeClustering(n_clusters=keep_nconf, linkage='complete', metric='precomputed').fit(arr)
    else:
        cl = AgglomerativeClustering(n_clusters=None, linkage='complete', metric='precomputed', distance_threshold=rms).fit(arr)

    keep_ids = []
    for i in set(cl.labels_):
        ids = np.where(cl.labels_ == i)[0]
        j = arr[np.ix_(ids, ids)].mean(axis=0).argmin()
        keep_ids.append(cids[ids[j]])
    remove_ids = set(cids) - set(keep_ids)

    for cid in sorted(remove_ids, reverse=True):
        mol_tmp.RemoveConformer(cid)

    return mol_tmp


def reassing_conf_ids(mol):
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)


def remove_conf(mol, cids):
    for cid in set(cids):
        mol.RemoveConformer(cid)
    reassing_conf_ids(mol)
    return mol


def fused_ring_atoms(m):
    # count rings considering fused and spiro cycles as a single ring system
    # print(rings('C1CC23CCC2CC13'))  # 1
    # print(rings('O=C(N1CCNCC1)c1ccc(=O)oc1'))  # 2
    # print(rings('O=C(C1CCC(=O)C23CCCC2CCC13)N1CCNC2CCCC12'))  # 2
    # print(rings('O=C(C1CCC(=O)C23CCCC2CCC13)N1CCNC2C1CCC21CCCC1'))  # 2
    # print(rings('C1CC2(C1)CC1(C2)C2CCC22CCC12'))  # 1
    # print(rings('CC12CCC(C1)C1CCC21'))  # 1
    # print(rings('CC12CCC3(CCC3C1)C2'))  # 1
    # print(rings('CC'))  # 0
    # print(rings('C1CC2CCCC(C1)CCCC2'))  # 1
    q = m.GetRingInfo()
    rings = [set(r) for r in q.AtomRings()]
    go_next = True
    while go_next:
        go_next = False
        for i, j in combinations(range(len(rings)), 2):
            if rings[i] & rings[j]:
                q = rings[i] | rings[j]
                del rings[j], rings[i]
                rings.append(q)
                go_next = True
                break
    return rings


def check_substr_mols(large, large_ring_ids, small):
    """

    :param large: larger mol
    :param large_ring_ids: list of sets with ids of ring systems of a larger mol
    :param small: smaller mol
    :return:
    """
    small_nrings = rdMolDescriptors.CalcNumRings(small)
    if small_nrings == 0:
        return large.HasSubstructMatch(small)
    else:
        for ids in large.GetSubstructMatches(small):
            ids = set(ids)
            ring_intersections = []   # collects True - if a ring is a full subset of matched features or
                                      # do not intersect, False - otherwise
                                      # only mol having all True values are valid for removal
            for r_ids in large_ring_ids:
                intersection = r_ids & ids
                ring_intersections.append((len(intersection) == 0) | (len(intersection) == len(r_ids)))
            if all(ring_intersections):
                return True
        return False


def select_mols(mols, ncpu=1):
    """
    Remove those molecules which are superstructure of another one. Thus, if CO and CCO matched a pharmacophore
    the latter is superfluous and can be removed. It is expected that if needed the former will be able to grow
    to CCO and further.
    :param mols:
    :param ncpu:
    :return:
    """

    pool = Pool(ncpu) if ncpu > 1 else None

    try:
        # mol, hac, list of sets with ids of ring systems required for substructure check
        mols = [(Chem.RemoveHs(mol), mol.GetNumHeavyAtoms(), fused_ring_atoms(Chem.RemoveHs(mol))) for mol in mols]
        mols = sorted(mols, key=itemgetter(1))
        hacs = np.array([item[1] for item in mols])
        deleted = np.zeros(hacs.shape)

        for i, (mol, hac, ring_ids) in enumerate(mols):
            if not deleted[i]:
                mol_ids = np.where(np.logical_and(hacs >= hac, deleted == 0))[0]
                mol_ids = np.delete(mol_ids, np.argwhere(mol_ids == i))
                if pool is None:
                    remove_ids = [j for j in mol_ids if check_substr_mols(mols[j][0], mols[j][2], mol)]
                else:
                    mask = list(pool.starmap(partial(check_substr_mols, small=mol),
                                             ((mols[mol_id][0], mols[mol_id][2]) for mol_id in mol_ids)))
                    remove_ids = mol_ids[mask]

                deleted[remove_ids] = 1
        output = [mols[i][0] for i in np.where(deleted == 0)[0]]
    finally:
        if pool:
            pool.close()

    return output


def combine_conformers(mols):
    mol = Chem.Mol(mols[0])
    for m in mols[1:]:
        ids = mol.GetSubstructMatch(m, useChirality=True)
        for c in m.GetConformers():
            pos = c.GetPositions()
            for query_id, atom_id in enumerate(ids):
                x, y, z = pos[query_id,]
                c.SetAtomPosition(atom_id, Point3D(x, y, z))
            mol.AddConformer(c, assignId=True)
    return mol


def merge_confs(mols_dict, ncpu=1):
    # mols_dict - dict {parent_conf_id: [mol1, mol2, ...], ...}
    # molecules with identical smiles are combined into one with multiple conformers
    smiles = defaultdict(list)  # {smi: [Mol], ...}
    for parent_conf_id, mols in mols_dict.items():
        for mol in mols:
            visited_ids = mol.GetProp('visited_ids')
            for c in mol.GetConformers():
                # c.SetProp('parent_conf_id', str(parent_conf_id))
                c.SetProp('visited_ids', visited_ids)
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles[smi].append(mol)

    pool = Pool(ncpu)
    try:
        mols = list(pool.imap_unordered(combine_conformers, smiles.values()))
    finally:
        pool.close()
    return mols


def remove_confs_exclvol(mol, exclvol_xyz, threshold):
    # if threshold < 0 (default -1) means ignore excl volumes
    if threshold >= 0 and exclvol_xyz is not None:
        cids = []
        ids = [atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()]
        for c in mol.GetConformers():
            d = cdist(c.GetPositions()[ids], exclvol_xyz)
            if (d < threshold).any():
                cids.append(c.GetId())
        if len(cids) == mol.GetNumConformers():
            return None
        for i in cids:
            mol.RemoveConformer(i)
    return mol


def get_pharm_xyz(pharm, ids=None):
    ids = pharm._get_ids(ids)
    coords = pharm.get_feature_coords(ids)
    df = pd.DataFrame([(i, label, *c) for i, (label, c) in zip(ids, coords)], columns=['id', 'label', 'x', 'y', 'z'])
    return df


def remove_confs_match(mol, pharm, matched_ids, new_ids, dist):

    remove_cids = []
    cids = [c.GetId() for c in mol.GetConformers()]
    plist = load_multi_conf_mol(mol)
    matched_xyz = get_pharm_xyz(pharm, matched_ids)
    new_xyz = get_pharm_xyz(pharm, new_ids)

    for cid, p in zip(cids, plist):

        conf_xyz = get_pharm_xyz(p)

        # match new pharmacophore features
        mask = np.array([i != j for i, j in product(conf_xyz['label'], new_xyz['label'])]).\
            reshape(conf_xyz.shape[0], new_xyz.shape[0])
        d = cdist(conf_xyz[['x', 'y', 'z']], new_xyz[['x', 'y', 'z']])
        d[mask] = dist + 1
        d = np.min(d, axis=0) <= dist
        new_matched_ids = new_xyz[d]['id'].tolist()
        if not new_matched_ids:
            remove_cids.append(cid)
            continue
        else:
            mol.GetConformer(cid).SetProp('matched_ids', ','.join(map(str, matched_ids + new_matched_ids)))
            mol.GetConformer(cid).SetIntProp('matched_ids_count', len(matched_ids) + len(new_matched_ids))

        # match previously matched pharmacophore features
        mask = np.array([i != j for i, j in product(conf_xyz['label'], matched_xyz['label'])]).\
            reshape(conf_xyz.shape[0], matched_xyz.shape[0])
        d = cdist(conf_xyz[['x', 'y', 'z']], matched_xyz[['x', 'y', 'z']])
        d[mask] = dist + 1
        if not (np.min(d, axis=0) <= dist).all():
            remove_cids.append(cid)
            continue

    if len(remove_cids) == mol.GetNumConformers():
        return None
    else:
        for cid in remove_cids:
            mol.RemoveConformer(cid)
        return mol


def get_confs_mp(items, template_mol, nconfs, conf_alg, pharm, new_pids, dist, evol, seed):
    mol, template_conf_id = items
    return get_confs(mol, template_conf_id, template_mol, nconfs, conf_alg, pharm, new_pids, dist, evol, seed)


def get_confs(mol, template_conf_id, template_mol, nconfs, conf_alg, pharm, new_pids, dist, evol, seed):

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    mol = Chem.AddHs(mol)

    try:
        mol = __gen_confs(mol, Chem.RemoveHs(template_mol), nconf=nconfs, alg=conf_alg, seed=seed,
                          coreConfId=template_conf_id, ignoreSmoothingFailures=False)
    except Exception as e:
        sys.stderr.write(f'the following error was occurred for in __gen_confs ' + str(e) + '\n')
        return template_conf_id, None

    if not mol:
        return template_conf_id, None

    mol = remove_confs_exclvol(mol, pharm.exclvol, evol)
    if not mol:
        return template_conf_id, None

    mol = remove_confs_match(mol,
                             pharm=pharm,
                             matched_ids=list(map(int, template_mol.GetConformer(template_conf_id).GetProp('matched_ids').split(','))),
                             new_ids=new_pids,
                             dist=dist)

    if not mol:
        return template_conf_id, None

    mol.SetProp('visited_ids', template_mol.GetProp('visited_ids') + ',' + ','.join(map(str, new_pids)))
    for conf in mol.GetConformers():
        conf.SetProp('parent_conf_id', str(template_conf_id))
    return template_conf_id, mol


def get_features(p, pids, add_features):
    if add_features:
        c = Counter(label for label, xyz in p.get_feature_coords(pids))
        output = dict()
        for k, v in c.items():
            if k == 'a':
                output['nAr'] = (v, 10)
            else:
                output[f'n{k}'] = (v, 10)
        return output
    else:
        return dict()


def get_property_restrictions(mol, max_mw, max_tpsa, max_rtb, max_logp):
    mw = max_mw - MolWt(mol)
    tpsa = max_tpsa - CalcTPSA(mol)
    rtb = max_rtb - CalcNumRotatableBonds(mol) - 1  # it is necessary to take into account the formation of bonds during the growth of the molecule
    if rtb == -1:
        rtb = 0
    logp = max_logp - MolLogP(mol) + 0.5
    return {'mw': (1, mw), 'tpsa': (0, tpsa), 'rtb': (0, rtb), 'logp': (-100, logp)}


def remove_mols_by_property(mols, max_mw, max_tpsa, max_rtb, max_logp):
    """

    :param mols: list of 2-tuple (smi, mol)
    :param max_mw:
    :param max_tpsa:
    :param max_rtb:
    :param max_logp:
    :return: list of 2-tuple (smi, mol)
    """
    output = []
    for smi, mol in mols:
        if MolWt(mol) > max_mw or \
                CalcTPSA(mol) > max_tpsa or \
                CalcNumRotatableBonds(mol) > max_rtb or \
                MolLogP(mol) > max_logp:
            continue
        output.append((smi, mol))
    return output


def filter_by_hashes(row_ids, cur, radius, db_hashes, hashes):
    """

    :param row_ids: selected row_ids which should be filtered (necessary parameter)
    :param cur: SQLite cursor object of CReM DB (necessary parameter)
    :param radius: radius in CReM DB (necessary parameter)
    :param db_hashes: file name of DB of 3D pharmacophore hashes
    :param hashes: list of 3D pharmacophore hashes to keep fragments
    :return:
    """

    if not row_ids:
        return []
    batch_size = 32000
    row_ids = list(row_ids)
    smis = defaultdict(list)
    for start in range(0, len(row_ids), batch_size):
        batch = row_ids[start:start + batch_size]
        sql = f"SELECT rowid, core_smi FROM radius{radius} WHERE rowid IN ({','.join('?' * len(batch))})"
        for i, smi in cur.execute(sql, batch).fetchall():
            smis[smi].append(i)

    con = sqlite3.connect(db_hashes)
    sql = f"SELECT DISTINCT(frags.smi) FROM frags, hashes WHERE frags.id == hashes.id AND hashes.hash IN ({','.join('?' * len(hashes))})"
    res = [item[0] for item in con.execute(sql, list(hashes)).fetchall()]

    output_row_ids = []
    for smi in res:
        output_row_ids.extend(smis[smi])
    return output_row_ids


def enumerate_hashes_directed(mol, att_ids, feature_positions, bin_step, directed):

    output_hashes = set()

    for conf in mol.GetConformers():
        for att_id in att_ids:
            t_pos = conf.GetAtomPosition(att_id)

            if directed:
                for end_atom in mol.GetAtomWithIdx(att_id).GetNeighbors():
                    if end_atom.GetAtomicNum() == 1:
                        end_atom_id = end_atom.GetIdx()
                        h_pos = conf.GetAtomPosition(end_atom_id)
                        q_pos = t_pos + (h_pos - t_pos) / np.linalg.norm(t_pos - h_pos) * bin_step
                        p = P(cached=True, bin_step=bin_step)
                        p.load_from_feature_coords(feature_positions + [('T', tuple(t_pos)), ('Q', tuple(q_pos))])
                        output_hashes.add(p.get_signature_md5())

            else:
                p = P(cached=True, bin_step=bin_step)
                p.load_from_feature_coords(feature_positions + [('T', tuple(t_pos))])
                output_hashes.add(p.get_signature_md5())

    return output_hashes


def read_multi_conf_sdf(fname):
    output = dict()
    try:
        for mol in Chem.SDMolSupplier(fname):
            if mol:
                mol_name = mol.GetProp('_Name')
                if mol_name not in output:
                    output[mol_name] = mol
                else:
                    output[mol_name].AddConformer(mol.GetConformer(), assignId=True)
    except Exception:
        traceback.print_exc()
    return list(output.values())


def gen_confs_cdpkit(mols, template_mol, nconf, ncpu):
    """

    :param mols: list of RDKit Mol
    :param template_mol: RDKit Mol
    :param nconf:
    :param ncpu:
    :return: list of multi-conf RDKit Mol
    """

    # import CDPL.Chem as CDPLChem
    # import CDPL.Base as CDPLBase
    # from io import BytesIO, StringIO

    # sio = StringIO()
    # w = Chem.SDWriter(sio)
    # for m in mols:
    #     w.write(m)
    # w.close()
    # sio.seek(0)

    input_fd, input_fname = tempfile.mkstemp(suffix=f'_{template_mol.GetProp("_Name")}_input.sdf', text=True)
    template_fd, template_fname = tempfile.mkstemp(suffix='_template.mol', text=True)
    output_fd, output_fname = tempfile.mkstemp(suffix=f'_{template_mol.GetProp("_Name")}_output.sdf', text=True)

    try:

        w = Chem.SDWriter(input_fname)
        for mol in mols:
            w.write(mol)
        w.close()

        with open(template_fname, 'wt') as f:
            f.write(Chem.MolToMolBlock(template_mol))

        args = f' -i {input_fname} -o {output_fname} -j {template_fname} -n {nconf} -t {ncpu} -a -m systematic --progress 0 '
        if platform.system() == 'Windows':
            cmd = f'confgen.exe {args} > NUL 2>&1'
        else:
            cmd = f'confgen {args} > /dev/null 2>/dev/null'
        subprocess.run(cmd, shell=True)
        output = read_multi_conf_sdf(output_fname)

    finally:
        os.close(input_fd)
        os.close(template_fd)
        os.close(output_fd)
        os.unlink(input_fname)
        os.unlink(template_fname)
        os.unlink(output_fname)

    return output


def filter_confs_mp(items, template_mol, pharm, evol, dist, new_pids):
    return filter_confs(*items, template_mol, pharm, evol, dist, new_pids)


def filter_confs(mol, template_conf_id, template_mol, pharm, evol, dist, new_pids):

    try:

        mol = remove_confs_exclvol(mol, pharm.exclvol, evol)
        if not mol:
            return template_conf_id, None

        mol = remove_confs_match(mol,
                                 pharm=pharm,
                                 matched_ids=list(map(int, template_mol.GetConformer(template_conf_id).GetProp('matched_ids').split(','))),
                                 new_ids=new_pids,
                                 dist=dist)
        if not mol:
            return template_conf_id, None

        mol.SetProp('visited_ids', template_mol.GetProp('visited_ids') + ',' + ','.join(map(str, new_pids)))
        for conf in mol.GetConformers():
            conf.SetProp('parent_conf_id', str(template_conf_id))

    except Exception:
        print(traceback.format_exc())

    return template_conf_id, mol


def expand_mol(mol, pharmacophore, additional_features, max_mw, max_tpsa, max_rtb, max_logp, hash_db, hash_db_bin_step,
               crem_db, radius, max_replacements, nconf, conf_gen, dist, exclusion_volume_dist, seed,
               output_dir, dask_num_workers=0, ncpu=1):

    timings = []
    start = timeit.default_timer()

    # with open('/home/pavel/python/crem-pharm/test/test1/output/mol_expand_test_obj.pkl', 'wb') as f:
    #     for item in (mol, pharmacophore, additional_features, max_mw, max_tpsa, max_rtb, max_logp, hash_db, hash_db_bin_step,
    #            crem_db, radius, max_replacements, nconf, conf_gen, dist, exclusion_volume_dist, seed,
    #            output_dir, dask_num_workers, ncpu):
    #         pickle.dump(item, f)

    new_pids = pharmacophore.select_nearest_cluster(tuple(map(int, mol.GetProp('visited_ids').split(','))))
    atom_ids = get_grow_atom_ids(mol, pharmacophore.get_xyz(new_pids))
    kwargs = get_features(pharmacophore, new_pids, additional_features)
    kwargs = {**kwargs, **get_property_restrictions(mol, max_mw=max_mw, max_tpsa=max_tpsa, max_rtb=max_rtb,
                                                    max_logp=max_logp)}

    # create additional constrains for selection of fragments which will be attached during growing
    __max_features = 6  # max number of enumerated feature combinations in 3D pharm hash db
    use_hash_db = hash_db is not None and len(new_pids) <= __max_features
    hashes = []
    if use_hash_db:
        feature_positions = pharmacophore.get_feature_coords(new_pids)
        hashes = enumerate_hashes_directed(mol=mol, att_ids=atom_ids, feature_positions=feature_positions,
                                           bin_step=hash_db_bin_step, directed=False)

    timings.append(f'preprocessing: {round(timeit.default_timer() - start, 4)}')
    start2 = timeit.default_timer()

    new_mols = list(grow_mol(mol, crem_db, radius=radius, min_atoms=1, max_atoms=12,
                             max_replacements=max_replacements, replace_ids=atom_ids, return_mol=True,
                             ncores=4,
                             filter_func=partial(filter_by_hashes, db_hashes=hash_db, hashes=hashes) if use_hash_db else None,
                             **kwargs))

    timings.append(f'mol grow: {len(new_mols)} mols, {round(timeit.default_timer() - start2, 4)}')
    start2 = timeit.default_timer()

    if new_mols:
        new_mols = remove_mols_by_property(new_mols, max_mw=max_mw, max_tpsa=max_tpsa, max_rtb=max_rtb, max_logp=max_logp)

    timings.append(f'mol grow after physchem rules: {len(new_mols)} mols, {round(timeit.default_timer() - start2, 4)}')
    start2 = timeit.default_timer()

    new_isomers = []
    if new_mols:
        for smi, m in new_mols:
            new_isomers.extend(gen_stereo(m))

    timings.append(f'stereo enumeration: {len(new_isomers)} isomers, {round(timeit.default_timer() - start2, 4)}')
    start2 = timeit.default_timer()

    new_mols_dict = defaultdict(list)  # {parent_conf_id_1: [mol1, mol2, ...], ... }

    if new_isomers:

        # create a list of input mols where names are ConfId of a parent mol
        inputs = []
        for i, (new_isomer, conf) in enumerate(product(new_isomers, mol.GetConformers())):
            new_isomer.SetProp('_Name', f'{i}_{conf.GetId()}')
            inputs.append(new_isomer)

        new_mols = gen_confs_cdpkit(inputs, template_mol=mol, nconf=nconf, ncpu=ncpu)

        timings.append(f'conf generation: {len(new_mols)} molecules, {round(timeit.default_timer() - start2, 4)}')
        start2 = timeit.default_timer()

        inputs = [(m1, int(re.findall(r'_(.*)$', m1.GetProp('_Name'))[0])) for m1 in new_mols]

        pool = Pool(ncpu)
        for conf_id, m in pool.imap_unordered(partial(filter_confs_mp,
                                                      template_mol=mol,
                                                      pharm=pharmacophore,
                                                      new_pids=new_pids,
                                                      dist=dist,
                                                      evol=exclusion_volume_dist),
                                              inputs):
            if m:
                new_mols_dict[conf_id].append(m)

        # if dask_num_workers:
        #     b = bag.from_sequence(inputs, npartitions=dask_num_workers * 2)
        #     for conf_id, m in b.starmap(get_confs,
        #                                 template_mol=mol,
        #                                 nconfs=nconf,
        #                                 conf_alg=conf_gen,
        #                                 pharm=pharmacophore,
        #                                 new_pids=new_pids,
        #                                 dist=dist,
        #                                 evol=exclusion_volume_dist,
        #                                 seed=seed).compute():
        #         if m:
        #             new_mols_dict[conf_id].append(m)
        # else:
        #     pool = Pool(ncpu)
        #     for conf_id, m in pool.imap_unordered(partial(get_confs_mp,
        #                                                   template_mol=mol,
        #                                                   nconfs=nconf,
        #                                                   conf_alg=conf_gen,
        #                                                   pharm=pharmacophore,
        #                                                   new_pids=new_pids,
        #                                                   dist=dist,
        #                                                   evol=exclusion_volume_dist,
        #                                                   seed=seed),
        #                                           inputs):
        #         if m:
        #             new_mols_dict[conf_id].append(m)
        #     pool.close()

    timings.append(f'conf filtration: {sum(len(v) for v in new_mols_dict.values())} molecules, {round(timeit.default_timer() - start2, 4)}')
    start2 = timeit.default_timer()

    if new_mols_dict:
        # keep only conformers with maximum number of matched features
        max_count = 0
        for v in new_mols_dict.values():
            for m in v:
                for c in m.GetConformers():
                    if c.GetIntProp('matched_ids_count') > max_count:
                        max_count = c.GetIntProp('matched_ids_count')

        for v in new_mols_dict.values():  # list of mols
            for i in reversed(range(len(v))):
                cids = []
                for c in v[i].GetConformers():
                    if c.GetIntProp('matched_ids_count') < max_count:
                        cids.append(c.GetId())
                if len(cids) == v[i].GetNumConformers():
                    del v[i]
                else:
                    for cid in cids:
                        v[i].RemoveConformer(cid)

    timings.append(f'conf filtering: {sum(len(v) for v in new_mols_dict.values())} compounds, {round(timeit.default_timer() - start2, 4)}')
    start2 = timeit.default_timer()

    if new_mols_dict:
        new_mols = merge_confs(new_mols_dict, ncpu=ncpu)  # return list of mols
        pool = Pool(ncpu)
        try:
            new_mols = list(pool.imap_unordered(remove_confs_rms, new_mols))
        finally:
            pool.close()
    else:
        new_mols = []

    timings.append(f'merge confs and remove by rms: {len(new_mols)} compounds, {round(timeit.default_timer() - start2, 4)}')
    start2 = timeit.default_timer()

    if new_mols:
        # with open(os.path.join(output_dir, f'{mol.GetProp("_Name")}.pkl'), 'wb') as f:
        #     pickle.dump(new_mols, f)
        new_mols = select_mols(new_mols, ncpu=ncpu)

    timings.append(f'mol selection: {len(new_mols)} compounds, {round(timeit.default_timer() - start2, 4)}')

    for m in new_mols:
        m.SetProp('parent_mol_id', mol.GetProp('_Name'))

    timings.append(f'overall time: {round(timeit.default_timer() - start, 4)}')

    return new_mols, len(new_isomers), timings


def main():
    parser = argparse.ArgumentParser(description='Expand a molecule to match the closest pharmacophore features.')
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True,
                        help='PKL (pickled) file with a single molecule to expand.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True,
                        help='PKL (pickled) file with multiple output molecules. Additional fields should have all '
                             'necessary information.')
    parser.add_argument('-p', '--pharm', metavar='FILENAME', required=True,
                        help='PKL (pickled) file name of a pharmacophore object.')
    parser.add_argument('--config', metavar='FILENAME', required=True,
                        help='YAML file name with all other settings settings.')
    parser.add_argument('--conf_count', metavar='FILENAME', required=True,
                        help='text file with the number of molecules for which conformers were embedded.')
    parser.add_argument('--debug', metavar='FILENAME', required=False,
                        help='text file with some additional information.')

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.pharm, 'rb') as f:
        pharm = pickle.load(f)

    with open(args.input, 'rb') as f:
        mol = pickle.load(f)

    RDLogger.DisableLog('rdApp.warning')
    new_mols, conf_count, timings = expand_mol(mol=mol, pharmacophore=pharm, **config)
    RDLogger.EnableLog('rdApp.warning')

    with open(args.output, 'wb') as f:
        pickle.dump(new_mols, f)

    if args.debug:
        with open(args.debug, 'wt') as f:
            f.write('\n'.join(timings) + '\n')

    with open(args.conf_count, 'wt') as f:
        f.write(str(conf_count) + '\n')


if __name__ == '__main__':
    main()
