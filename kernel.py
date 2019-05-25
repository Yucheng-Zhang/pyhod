'''
Kernel functions.
'''
import numpy as np
import collections
from astropy.cosmology import FlatLambdaCDM
from scipy import special
from scipy import interpolate
import sys


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
        print('++ catalog loaded')

        halos['xyz'] = catalog[:, :3]  # Mpc (comoving)
        # to say whether halo co-ordinates are in unit box or not
        halos['xyztype'] = 'sky'
        halos['vxyz'] = catalog[:, 3:6]  # km/s

        halos['r200'] = catalog[:, 6]  # Mpc

        del catalog
        print('-- catalog unloaded')

        # for now assuming the velocity is 0.2 times the velocity as place holder
        # need to use some empirical relation here
        print('> Computing sigV...')
        halos['sigv'] = 0.2 * \
            np.sqrt(np.sum(np.power(halos['vxyz'], 2), axis=1))

        rho = 2.775e11 * Om0 * h**2  # Msun/Mpc^3

        print('> Computing mass of the halos...')
        halos['Mvir'] = 4. / 3. * np.pi * np.power(halos['r200'], 3) * rho
        halos['Mhalo'] = np.copy(halos['Mvir'])

        print('> Computing redshifts of the halos...')
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        zs = np.linspace(z1, z2, 1000)
        rs = cosmo.comoving_distance(zs).value
        r_dist = np.sqrt(np.sum(np.power(halos['xyz'], 2), axis=1))
        halos['redshift'] = np.interp(r_dist, rs, zs)

        del cosmo, zs, rs, r_dist

        # Concentraition is estimated usin Duffy et.al. Table 1
        # https://arxiv.org/pdf/0804.2486.pdf
        print('> Computing concentration...')
        concen = 7.85 * np.power(halos['Mhalo'] / (2e+12/h), -0.081)
        concen = concen / np.power(1.0 + halos['redshift'], 0.71)
        halos['rsr200'] = 1. / concen

        del concen

    print('>> Finish loading, number of halos: {0:d}'.format(
        halos['xyz'].shape[0]))

    # print range
    print('>> Halo limits:')
    for p in ['xyz', 'vxyz']:
        print('-- {}'.format(p))
        for i, c in enumerate(['x', 'y', 'z']):
            print('-- {0:s}: {1:12.5e} {2:12.5e}'.format(
                c, halos[p][:, i].min(), halos[p][:, i].max()))
    print('-- Mhalo:\n-- {0:12.5e} {1:12.5e}'.format(
        halos['Mhalo'].min(), halos['Mhalo'].max()))

    # halo mass function
    Nhalo, binhalo = np.histogram(np.log10(halos['Mhalo']), bins=100)
    mhalo_mid = np.power(10, 0.5 * (binhalo[:-1] + binhalo[1:]))
    halos['halohist'] = np.column_stack([mhalo_mid, Nhalo])

    return halos


def Mfof2M200(mfof):
    '''Convert mfof to m200 using Surhud's formula.'''
    Lmfof = np.log10(mfof)
    pbest = np.array([1.21128825e+01, 9.33904756e-01, 1.20913807e-02])
    Lm200b = pbest[2] * (Lmfof-12.0)**2 + pbest[1] * (Lmfof-12.0) + pbest[0]
    m200b = np.power(10, Lm200b)
    return m200b


def MeanBCG(args, Mhalo):
    '''Find mean number of BCG based on halo mass.'''
    logMcut = np.log(10.0) * args.LMcut
    Mdiff = (logMcut - np.log(Mhalo)) / (np.sqrt(2.0) * args.sigma)
    NBCG = 0.5 * special.erfc(Mdiff)
    return NBCG


def MeanSat(args, Mhalo, NBCG):
    '''Find mean number of satelite.'''
    Mdiff = (Mhalo - args.kappa * np.power(10, args.LMcut)) / \
        np.power(10, args.LM1)
    Nsat = np.zeros(Mdiff.size)
    index = Mdiff > 0
    Nsat[index] = NBCG[index] * np.power(Mdiff[index], args.alpha)
    return Nsat


def NFW_setup(args):
    # copied from Sebastien code which is taken from NFW 1996 paper
    tabm200 = np.array([3.519, 20.59, 22.01, 22.65, 26.16, 28.15,
                        29.67, 102.68, 224.35, 1109.9, 1931.5, 3009.7])
    tabr200 = np.array([394, 710, 726, 733, 769, 788,
                        802, 1213, 1574, 2682, 3226, 3740])
    tabrsr200 = np.array([0.068, 0.078, 0.088, 0.065, 0.077, 0.124,
                          0.131, 0.110, 0.121, 0.151, 0.188, 0.143])
    tabv200 = np.array([196.0, 353.2, 361.1, 364.6, 382.5, 392.0,
                        398.9, 603.4, 783.0, 1334., 1605., 1861.])
    Logm200 = np.log10(tabm200)
    #Logr200  =np.log10(tabr200)
    # Logrsr200=np.log10(tabrsr200)
    # //critical density with redshift
    rho_c = 0.92
    rho_c = rho_c * (args.Om0 * pow(1.0+args.redshift, 3) + 1. - args.Om0)
    interp_r200 = interpolate.splrep(Logm200, tabr200, s=0, k=1)
    interp_rsr200 = interpolate.interp1d(Logm200, tabrsr200)

    return interp_r200, interp_rsr200, rho_c


def populate_satelite(args, halos, hindex, galcat, nbcgsel, rho_c, meanSat, r200, rsr200):
    '''Populate satelite.'''
    if meanSat.size != r200.size:
        sys.exit('Error: Number of satelite and r200 does not match')

    nsat = 0
    Ngal = nbcgsel

    # get the rho_0 and rs
    if args.satelite == 'NFW':
        delta_c = args.fconc * 1.0 / rsr200
        delta_c = 200.0 * np.power(delta_c, 3) / \
            (np.log(1.0 + delta_c) - (delta_c / (1.0 + delta_c)))
        rho_0 = delta_c * rho_c
        rs = r200 * rsr200
        r200 = args.fconc * r200
    elif args.satelite == 'bolshoiNFW':
        rs = rsr200 * r200
        r200 = args.fconc * r200
        # see Mvir equation in :
        # https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile
        delta_c = args.fconc * 1.0 / rsr200
        delta_c = np.log(1.0 + delta_c) - (delta_c / (1.0 + delta_c))
        rho_0 = halos['Mvir'][hindex] / (np.pi * 4.0 * np.power(rs, 3))
        rho_0 = rho_0 / delta_c
    else:
        sys.exit('Error: Invalid satelite variable')

    dr = 1.0
    drby2 = dr / 2.0
    nr = np.floor((r200-1.0)/dr)
    nbin = np.max(nr)+1
    rr = np.linspace(1.0, dr*nbin, nbin)
    dv = 4.0*np.pi*(np.power(rr+drby2, 3)-np.power(rr-drby2, 3))/3.0

    a = 1.0 / (1 + args.redshift)
    H = args.H0 * np.sqrt(args.Om0 / np.power(a, 3) + 1.-args.Om0)
    hxyz = halos['xyz'][hindex, :]
    hmhalos = halos['Mhalo'][hindex]
    if galcat.shape[1] > 3:  # whenever velocity is needed
        hvxyz = args.gammaHV * halos['vxyz'][hindex, :]
        hsigv = args.gammaIHV * halos['sigv'][hindex]
    for ii in range(0, r200.size):
        Nsat = meanSat[ii]
        rhodv = dv*rho_0[ii]*dr/(rr*np.power(1+(rr/rs[ii]), 2)/rs[ii])
        cumrho = np.cumsum(rhodv[:np.int(nr[ii])], dtype=float)
        cumrho = cumrho/cumrho[-1]
        rands = np.random.random(Nsat)
        theta = np.pi*np.random.random(Nsat)
        phi = 2*np.pi*np.random.random(Nsat)
        if galcat.shape[1] > 3:  # whenever velocity is needed
            vsatx = np.random.normal(
                loc=hvxyz[ii, 0], scale=hsigv[ii], size=Nsat)
            vsaty = np.random.normal(
                loc=hvxyz[ii, 1], scale=hsigv[ii], size=Nsat)
            vsatz = np.random.normal(
                loc=hvxyz[ii, 2], scale=hsigv[ii], size=Nsat)
        for jj in range(0, Nsat):
            # satelite co-ordinates
            rsat = rr[np.argmin(np.abs(cumrho-rands[jj]))]/1000.0  # in Mpc
            if halos['xyztype'] == 'unit':
                rsat = rsat / args['Lbox']  # in unit box
            # add entries to galcat if its full in terms of rows
            if galcat.shape[0] == Ngal:
                if Ngal > 2*halos['xyz'].shape[0]:
                    print('****Warning: Too many galaxies!!', ii, Ngal,
                          halos['xyz'].shape[0], r200.size,
                          np.sum(meanSat))
                galcat = np.row_stack(
                    [galcat, np.zeros(10000*galcat.shape[1]).reshape(10000, galcat.shape[1])])

            galcat[Ngal, 0] = hxyz[ii, 0]+rsat * \
                np.sin(theta[jj])*np.cos(phi[jj])
            galcat[Ngal, 1] = hxyz[ii, 1]+rsat * \
                np.sin(theta[jj])*np.sin(phi[jj])
            galcat[Ngal, 2] = hxyz[ii, 2]+rsat*np.cos(theta[jj])
            # if hcat in unit
            if halos['xyztype'] == 'unit':
                galcat[Ngal, :3] = np.mod(galcat[Ngal, :3], 1.0)
            elif halos['xyztype'] != 'sky':
                galcat[Ngal, :3] = np.mod(galcat[Ngal, :3], args['Lbox'])

            if galcat.shape[1] > 3:  # whenever velocity is needed
                # satelite velocity
                galcat[Ngal, 3] = vsatx[jj]  # vfactor[jj]*hvxyz[ii,0]
                galcat[Ngal, 4] = vsaty[jj]  # vfactor[jj]*hvxyz[ii,1]
                galcat[Ngal, 5] = vsatz[jj]  # vfactor[jj]*hvxyz[ii,2]

            if galcat.shape[1] > 6:  # extra information is not needed many times
                galcat[Ngal, 6] = hsigv[ii]
                galcat[Ngal, 7] = hmhalos[ii]  # halo mass
                # satelite flag
                galcat[Ngal, 8] = 10

            Ngal = Ngal + 1
            nsat = nsat + 1

    return galcat, nsat, Ngal


def hod(args, halos, galcat='', int_r200='', int_rsr200='', rho_c=0, write=0):
    '''Main function of HOD.'''
    nhalo = halos['xyz'].shape[0]  # number of halos

    if args.cfof:  # convert mfof to m200 using Surhud's formula
        halos['Mhalo'] = Mfof2M200(halos['Mhalo'])

    # find mean number of BCG based on halo mass
    meanBCG = MeanBCG(args, halos['Mhalo'])

    # find mean number of satelite
    meanSat = MeanSat(args, halos['Mhalo'], meanBCG)

    # populte BCG
    print('> Populating BCG...')
    urand = np.random.random(nhalo)
    index = urand < meanBCG
    nbcgsel = np.sum(index)

    print('> Number of BCG {0:d}'.format(nbcgsel))

    if isinstance(galcat, str):
        ntmp = np.int(1.2 * nbcgsel)
        if write == 1:
            ncols = 9
        else:
            ncols = 3
        if 'vxyz' not in halos.keys():
            ncols = 3

        galcat = np.zeros(ntmp * ncols).reshape(ntmp, ncols)

    galcat[:nbcgsel, :3] = halos.['xyz'][index, :]
    if galcat.shape[1] > 3:
        galcat[:nbcgsel, 3:6] = args.gammaHV * halos['vxyz'][index, :]

    if galcat.shape[1] > 6:
        galcat[:nbcgsel, 6] = args.gammaIHV * halos['sigv'][index]
        galcat[:nbcgsel, 7] = halos['Mhalo'][index]
        galcat[:nbcgsel, 8] = 100

    # populate satelite
    print('> Populating satelite...')
    Nsat = np.random.poisson(lam=meanSat)
    index = Nsat > 0  # halos with non-zero satelites

    # figure out rs and rsr200 for NFW
    if 'r200' not in halos.keys() or 'rsr200' not in halos.keys():
        if int_r200 == '' or int_rsr200 == '' or rho_c == 0:  # interpolation needed
            int_r200, int_rsr200, rho_c = NFW_setup(args)
            print('*** Interpolating the r200 and rsr200 for NFW')

        r200 = interpolate.splev(
            np.log10(halos['Mhalo'][index])-12.0, int_r200, der=0)
        #rsr200 = interpolate.int_rsr200(np.log10(hcat[index,Mind])-12.0)
        rsr200 = interp_exterp1d(halos['Mhalo'][index], int_rsr200)
        # interpolate and linearly extrapolate
        galcat, nsatsel, Ngal = populate_satelite(args, hcat, index, galcat,
                                                  nbcgsel, rho_c, Nsat[index], r200, rsr200)
    else:
        galcat, nsatsel, Ngal = populate_satelite(args, hcat, index, galcat,
                                                  nbcgsel, 0, Nsat[index],
                                                  halos['r200'][index], halos['rsr200'][index])

    print('> Number of Sat: {0:d}'.format(nsatsel))

    if write == 1:
        header = 'Galaxy catalog:\n'
        header += 'x   y   z   Vx   Vy   Vz   sigV   Mhalo   flag (10: Sat, 100: BCG)'
        fmt = '16.8g ' * galcat.shape[1]
        np.savetxt(args.gcat, galcat[:Ngal, :], header=header, fmt=fmt)
        print(':: Galaxy catalog: {}'.format(args.gcat))
