# CReM-pharm: enumeration of compounds based on 3D pharmacophores

Starting fragments matching a subset of pharmacophoree features are grown to match all remaining features. CReM generator is used for structure growing that provides flexible control over synthetic feasibility of generated structures. The search is designed in a balanced way, where the algorithm tries to expand all seach branches and later explore them exhaustively. Therefore, the generation may be interrupted at will (and continue later if necessary).  

## Installation

```
conda install -c conda-forge python rdkit scikit-learn openbabel networkx=3.3 pyyaml dask distributed
pip install crem pmapper

# Specific version of psearch supporting pre-compiled pharmacophore databases
pip install git+https://github.com/meddwl/psearch.git@crempharm

pip install crempharm
# or
pip install git+https://github.com/ci-lab-cz/crem-pharm.git 
```
### installation of conformer generators

CReM-pharm supports three conformer generators: CDPKit, RDKit, Openbabel

**CDPKit** (highly recommended)  
Fast generation of high quality conformers (~90x speed up relatively to RDKit). Install all binaries, not Python-bindings only - https://cdpkit.org/installation.html#installation-via-installer-package
 
## Generation of a database of conformers of starting fragments

Generate 10 conformers per molecule using 10 cores.

```
gen_db -i frags.smi -o frags.dat -n 10 -c 10 -v
```
The script supports filtering conformers by energy and RMSD, enumerates stereoisomers for undefined stereocenters and double bonds, etc.  
An example of starting fragments can be found at this [repository](https://doi.org/10.5281/zenodo.17174627)

## Pharmacophore query format

Pharmacophore is supplied as xyz-file. The first row is blank or any text can be there, the second row can be keep constant `bin_step=1` or keep it blank. Each other row is a definision of a pharmacophore feature: a type and three coordinates separated by spaces.  
Feature types are:  
A - H-bond acceptor  
D - H-bond donor  
a - aromatic  
H - hydrophobic  
P - positively charged center  
N - negatively charged center  
e - exclusion volume  

Definition of pharmacophore features are determined by [pmapper](https://github.com/DrrDom/pmapper) by default. Exclusion volumes can be read from a model file or specified manually.

```text

bin_step=1
H 4.48 27.86 7.1
H -1.33 28.56 11.28
A 1.63 31.91 7.0
A 7.47 25.97 8.66
A 6.82 24.61 6.65
D 3.11 30.12 6.57
D 0.01 33.4 7.84
D 5.53 24.31 8.82
e 3.32 23.52 0.21
e 3.51 22.56 0.98
...
```

Pharmacophore models from Pharmit and LiganScout can be converted to xyz format by means of `pmapper`.  

```python
from pmapper.pharmacophore import Pharmacophore as P
p = P()
# Pharmit
p.load_from_pharmit('model.json')
p.save_to_xyz('model.xyz')
# LigandScout
p.load_ls_model('model.pml')
p.save_to_xyz('model.xyz')
```

### Note on exclusion volumes
Usually exclusion volumes are quite sparsely placed in pharmacophore models. Therefore, to avoid proliferation of a ligand between exclusion volumes during the generation one may choose a larger radius of exclusion volumes (`--exclusion_volume` argument) or assigned exclusion volume to each protein atom in vicinity of a reference ligand used for structure-based model retrieval. Increasing the radius of exclusion volumes may result in a smaller cavity available to a ligand to grow.

Exclusion volumes are optional. They only prevents of generation of unnecessary large molecules, which will be discarded later nevertheless.

## Run CReM-pharm

Example of a run

```bash
crempharm --query model.xyz --ids 2 5 6 --output output_dir --clustering_threshold 3 \
  --db crem_fragment.db --conf_gen cdpkit --nconf 10 --seed 42 --dist 1.5 --exclusion_volume 2.2 \
  --fragments frags.dat --mw 450 --tpsa 120 --rtb 7 --logp 4 --ncpu 3 -w 10 --log log.txt
```

`--query model.xyz` - 3D pharmacophore model  
`--ids 2 5 6` - list of pharmacophore model feature ids (0-based indexing), which will be used at the first iteration to screening the starting fragments. It is recommened to choose 3 or 4 features which are placed close enough to be able to be matched by a single starting fragment.   
`--output output_dir` - the output directory, where output files will be stored. If the directory exists and contains `res.db`, the generation will be automatically continued.  
`--clustering_threshold 3` - remaining pharmacophore features (not specified at the start) will be clustered to determine groups of featurtes to be used on each iteration of structure expansion. These groups are determined by agglomerative clustering using a specific threshold.  
`--db crem_fragment.db` - a database of precompiled CReM fragments using for structure generation  
`--conf_gen cdpkit` - conformer generator to use  
`--nconf 10` - number of conformers  
`--seed 42` - seed only applicable to conformer generation
`--dist 1.5` - the maximum distance from a query features to a pharmacophore center of a molecule (all features has the same distance)  
`--exclusion_volume 2.2` - minimum distance from any atom of a molecule to any exclusion volume feature (all exclusion volumes has the same distance)  
`--fragments frags.dat` - a starting fragment database created as described above  
`--mw 450 --tpsa 120 --rtb 7 --logp 4` - maximum allowed physicochemical properties of generated structures, it they are exceeded a molecule will not grow anymore  
`--ncpu 3` - number of cores used for expand a single molecule
`-w 10` - number of molecules expanded simultaneously. The product of `--ncpu` and `-w` may exceed the total number of cores to fully utilize resources.  
`--log log.txt` - log file, where a user may monitor progress. A more detailed output is printed to a console.  

### CReM databses and their enhancement with pharmacophore feature count

Precompiled CReM fragment databases suitable for CReM-pharm can be downloaded [here](https://qsar4u.com/pages/crem.php).  

If a custom CReM fragment database is used it is worth to enhance it. Fragments for growing can be selected based on the minimal number of required features count. If an adding fragment should match an H-bond donor, there is no sense to try to embed a fragment which does not containg this type of features. Therefore, CReM database can be enhanced by adding these feature counts.

```
crempharm_add_pmapper -i crem.db -c 10 -v
```
Availability of feature counts will be detected automatically and the generation will be adjusted accordongly. The precompiled databases already contain these additional columns. 

### Extract generated structures

To extract structures from an output database we recommend to use `get_sdf_from_easydock` utility from [EasyDock](https://github.com/ci-lab-cz/easydock). To install EasyDock `pip install easydock`

To get all conformers for all compounds stored in a database. 

`get_sdf_from_easydock -i res.db -o res.sdf`

To get only the first conformer for all compounds. It may be that there will be several conformers matching a pharmacophore features.  

`get_sdf_from_easydock -i res.db -o res.sdf -f`

To get first conformers of compounds which matched at least 4 features in a model.  

`get_sdf_from_easydock -i res.db -o res.sdf -f --add_sql 'matched_ids_count >= 4'`

To get SMILES of generated structures along with their ids in a database.  

`get_sdf_from_easydock -i res.db -o res.smi -f --fields id`  

## License

GPLv3

## Citation


3D pharmacophore models used in the study, structures of all generative runs of CReM-pharm and PGMG as well as ZINC compounds and active compounds from ChEMBL are accessible at https://doi.org/10.5281/zenodo.17174628. Pre-compiled CReM fragments databases are available at https://doi.org/10.5281/zenodo.16909328.