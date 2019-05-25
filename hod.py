'''
Halo Occupation Distribution.
'''
import numpy as np
import argparse
import utils
from kernel import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Populate halo with galaxies using HOD.')
    # input & output data files
    parser.add_argument('-hcat', type=str, default='',
                        help='Input halo catalog')
    parser.add_argument('-hfmt', type=str, default='websky',
                        help='Halo catalog format')
    parser.add_argument('-z1z2', type=float, default=[0., 6.], nargs='+',
                        help='Redshift range of the halo catalog')
    parser.add_argument('-gcat', type=str, default='hod_out.gcat',
                        help='Output galaxy catalog')

    # cosmological parameters
    parser.add_argument('-H0', type=float, default=68.0, help='Hubble')
    parser.add_argument('-Om0', type=float, default=0.31, help='Matter')

    # HOD parameters
    # mfof to m200
    parser.add_argument('-cfof', type=int, default=1, help='')
    # MeanBCG
    parser.add_argument('-LMcut', type=float, default=13.249, help='')
    parser.add_argument('-sigma', type=float, default=0.8975, help='')
    # MeanSat
    parser.add_argument('-kappa', type=float, default=0.136867, help='')
    parser.add_argument('-LM1', type=float, default=14.1789, help='')
    parser.add_argument('-alpha', type=float, default=1.15071, help='')
    # hod
    parser.add_argument('-gammaHV', type=float, default=1.0, help='')
    parser.add_argument('-gammaIHV', type=float, default=1.0, help='')
    # NFW_setup
    parser.add_argument('-redshift', type=float, default=0.8, help='')
    # populate_satelite
    parser.add_argument('-satelite', type=str, default='NFW', help='Profile')
    parser.add_argument('-fconc', type=float, default=1.0, help='')

    args = parser.parse_args()


if __name__ == "__main__":
    print('>> Loading halo catalog: {}'.format(args.hcat))
    halos = load_halos(args)

    hod(args, halos, write=1)
