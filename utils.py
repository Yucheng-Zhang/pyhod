'''
Some useful functions.
'''

import healpy as hp
import numpy as np
import pandas as pd


def load_data_pd(fn, tp=''):
    '''Load data file w/ pandas.'''
    print('>> Loading data: {}'.format(fn))
    df = pd.read_csv(fn, delim_whitespace=True, comment='#', header=None)
    df = df.to_numpy()

    return df


def get_ra_dec(theta, phi):
    '''Get RA, DEC [degree] from theta, phi [radians] used in Healpy.'''
    rot = hp.Rotator(coord=['G', 'C'])
    theta_equ, phi_equ = rot(theta, phi)
    dec, ra = 90. - np.rad2deg(theta_equ), np.rad2deg(phi_equ)
    # move RA in [-180,0) to [180,360)
    ra = np.where(ra < 0., ra + 360., ra)

    return ra, dec


def get_theta_phi(ra, dec):
    '''Get theta, phi [radians] used in Healpy from RA, DEC [degree].'''
    rot = hp.Rotator(coord=['C', 'G'])
    theta_equ, phi_equ = np.deg2rad(90.-dec), np.deg2rad(ra)
    theta_gal, phi_gal = rot(theta_equ, phi_equ)

    return theta_gal, phi_gal
