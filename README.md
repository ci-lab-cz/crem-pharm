This version supports additional filtering of growing fragments by pre-compiled db of 3D pharmacophore hashes for up to 6 feature combinations and fragments with up to 12 heavy atoms.

Requires the corresponding version of crem module, which supports filter_func argument.

By default the program should work as previously.  


#### Installation of modules required for running program supporting additional filtering of fragments by 3D hashes

```
conda install -c conda-forge python rdkit scikit-learn openbabel networkx pyyaml dask distributed
pip install crem

# Specific version of psearch supporting pre-compiled pharmacophore databases
pip install git+https://github.com/meddwl/psearch.git@gen_pharms
```
Optional installations of conformer generators

**CDPKit** (highly recommended) - https://cdpkit.org/installation.html#installation-via-installer-package (install all binaries. not Python-bindings only)  

**OpenBabel**
