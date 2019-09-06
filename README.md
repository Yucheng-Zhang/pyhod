# pyhod
Python code for Halo Occupation Distribution.

- Yucheng Zhang, Shadab Alam
- Halo file type supported: `WebSky`

```python
from pyhod.hod import hod

# basic procedure, look at the code for default parameter values

# initialize
wsk = hod()
# load halo catalog
wsk.load_halos('halos.pksc', 0.2, 1.2) # output redshift range
# set HOD paramerters
wsk.set_hod_params()
# run HOD
wsk.populate(froot='hod_test')
```

## TODO

- Emperical relations for velocity dispersion.
- Optimize radial dist of satellites.
