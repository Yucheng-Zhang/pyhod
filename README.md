# halod
Halo Occupation Distribution based on Shadab's code.

```python
from halod.hod import hod

# Basic procedure, look at the code for default parameter values

# initialize
wsk = hod()
# load halo catalog
wsk.load_halos('halos.pksc')
# compute mean
wsk.c_meanBCG()
wsk.c_meanSat()
# populate
wsk.populate_BCG()
wsk.populate_Sat()
# output galaxy catalog
wsk.write_gcat('test.gcat')
```

## TODO

- Convert to Hubble units
- Emperical relations for velocity dispersion
