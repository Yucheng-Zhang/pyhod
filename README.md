# pyhod
Python code for Halo Occupation Distribution.

- Yucheng Zhang, Shadab Alam

```python
from halod.hod import hod

# basic procedure, look at the code for default parameter values

# initialize
wsk = hod()
# load halo catalog
wsk.load_halos('halos.pksc')
# compute mean
wsk.c_meanNC()
wsk.c_meanNS()
# populate
wsk.populate_C()
wsk.populate_S()
# output galaxy catalog
wsk.make_header()
wsk.write_gcat('test.gcat')
wsk.write_rdzw('test_rdzw.dat') # RA, DEC, Z, weight
```

## TODO

- Emperical relations for velocity dispersion.
- Optimize radial dist of satellites.
