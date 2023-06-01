This version supports additional filtering of growing fragments by pre-compiled db of 3D pharmacophore hashes for up to 6 feature combinations and fragments with up to 12 heavy atoms.

Requires the corresponding version of crem module, which supports filter_func argument.

By default the program should work as previously.  


##### Installation of modules required for running program supporting additional filtering of fragments by 3D hashes

RDKit should be installed in advance with python 3.7+

crem version 0.2.11 is required

```
pip install pmapper crem
```

```
pip install git+https://github.com/meddwl/psearch.git@gen_pharms
```

```
conda install -c conda-forge scikit-learn openbabel networkx pyyaml dask distributed
``` 
