'''
Kernel functions.
'''
import numpy as np
import collections
from astropy.cosmology import FlatLambdaCDM


def load_halos(args):
    '''Load the halo catalog.'''
    halos = collections.OrderedDict()
    Om0, H0, h = args.Om0, args.H0, args.H0 / 100.
    z1, z2 = args.z1z2[0], args.z1z2[1]

    if args.hfmt == 'websky':
        hf = open(args.hcat)
        Nhalo = np.fromfile(hf, count=3, dtype=np.int32)[0]

        catalog = np.fromfile(hf, count=Nhalo*10, dtype=np.float32)
        catalog = np.reshape(catalog, (Nhalo, 10))

        halos['xyz'] = catalog[:, :3]  # Mpc (comoving)
        # to say whether halo co-ordinates are in unit box or not
        halos['xyztype'] = 'sky'
        halos['vxyz'] = catalog[:, 3:6]  # km/s

        # for now assuming the velocity is 0.2 times the velocity as place holder
        # need to use some empirical relation here
        halos['sigv'] = 0.2 * \
            np.sqrt(np.sum(np.power(catalog[:, 3:6], 2), axis=1))

        halos['r200'] = catalog[:, 6]  # Mpc

        rho = 2.775e11 * Om0 * h**2  # Msun/Mpc^3

        halos['Mvir'] = 4. / 3. * np.pi * np.power(halos['r200'], 3) * rho
        halos['Mhalo'] = np.copy(halos['Mvir'])

        # get the redshift of the halos
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        zs = np.linspace(z1, z2, 1000)
        rs = cosmo.comoving_distance(zs).value
        r_dist = np.sqrt(np.sum(np.power(halos['xyz'], 2), axis=1))
        halos['redshift'] = np.interp(r_dist, rs, zs)

        # Concetraition is estimated usin Duffy et.al. Table 1
        # https://arxiv.org/pdf/0804.2486.pdf
        concen = 7.85 * np.power(halos['Mhalo'] / (2e+12/h), -0.081)
        concen = concen / np.power(1.0+halos['redshift'], 0.71)
        halos['rsr200'] = 1. / concen

    print('>> Finish loading, number of halos: {0:d}'.format(
        halos['xyz'].shape[0]))

    # print range
    print('>> Halo limits:')
    for p in ['xyz', 'vxyz']:
        print('-- {}'.format(p))
        for i, c in enumerate(['x', 'y', 'z']):
            print('-- {0:s}: {1:12.5e} {2:12.5e}'.format(
                c, halos[p][:, i].min(), halos[p][:, i].max()))
    print('-- Mhalo: {0:12.5e} {1:12.5e}'.format(
        halos['Mhalo'].min(), halos['Mhalo'].max()))

    # halo mass function
    Nhalo, binhalo = np.histogram(np.log10(halos['Mhalo']), bins=100)
    mhalo_mid = np.power(10, 0.5 * (binhalo[:-1] + binhalo[1:]))
    halos['halohist'] = np.column_stack([mhalo_mid, Nhalo])

    return halos
