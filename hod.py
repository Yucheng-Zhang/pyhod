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
    parser.add_argument('-satelite', type=str, default='NFW', help='Profile')

    args = parser.parse_args()


if __name__ == "__main__":
    print('>> Loading halo catalog: {}'.format(args.hcat))
    halos = load_halos(args)
