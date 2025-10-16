# Examples

## CReM fragment database

You may download them from this link https://doi.org/10.5281/zenodo.16909328

## Starting fragments

Starting fragments used in a study can be accessef by this link https://doi.org/10.5281/zenodo.17174628

## Run generation

```
crempharm -q ${QUERY}_full.xyz --ids ${IDS} -o output_dir \
  -t 3 -f starting_fragments.dat -d chembl33_sa25_f5.db \
  -r 3 -n 20 --conf_gen cdpkit --dist 1.5 -e 2.2 --mw 450 \
  --tpsa 120 --rtb 7 --logp 4 -c 2 -w 4
```

Recommended starting feature ids for individual pharmacophore models

QUERY | IDS
--|--
3ral | 2 5 6 
2btr | 2 3 
2fvd | 3 6 7 
6guh | 0 4 7 
3fuk | 2 3 
4ey7 | 3 4 
4gv1 | 0 5 9 
6b8y | 2 6 
6cm4 | 0 2 
6uwp | 3 6 7 
7ont | 0 4 5 6 
8dv7 | 2 6 7 
