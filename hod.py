'''
Halo Occupation Distribution.
'''
import numpy as np
import collections
from astropy.cosmology import FlatLambdaCDM
from scipy import special, interpolate
import time
import healpy as hp
from . import utils


class hod:
    '''HOD main class.'''

    def __init__(self, Om0=0.31, h0=0.68):

        self.Om0 = Om0
        self.h0 = h0
        self.H0 = 100 * h0
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0)

        self.hfile, self.hftype = None, None

        self.zmin, self.zmax = None, None

        ### halo properties ###
        self.halos = collections.OrderedDict()
        self.halos['nhalo'] = None  # num. of halos
        # position, velocity & redshift
        self.halos['xyz'] = None
        self.halos['vxyz'] = None
        self.halos['redshift'] = None
        # overdensity w/ respect to rho_m(z), mass & radius
        self.halos['Delta'] = None
        self.halos['R'] = None
        self.halos['M'] = None
        # mean num. of central & satellite galaxies
        self.halos['meanNc'] = None
        self.halos['meanNs'] = None
        # velocity dispersion
        self.halos['sigv'] = None

        ### HOD parameters ###
        # mean BCG
        self.LMmin, self.sigma = None, None
        # mean satellite
        self.LMcut, self.LMsat, self.alpha = None, None, None
        # velocity related
        self.gammaHV, self.gammaIHV = None, None

        ### galaxy properties ###
        self.gcat = collections.OrderedDict()
        self.gcat['xyz'] = None
        self.gcat['vxyz'] = None
        self.gcat['type'] = None  # 0 for BCG; 1 for Sat.
        self.gcat['nBCG'] = None
        self.gcat['nSat'] = None

        ### output header ###
        self.header = None

        ### chi2z interpolator ###
        zs = np.arange(0., 10., 0.001)
        chis = self.cosmo.comoving_distance(zs).value * self.h0  # Mpc/h
        self.chi2z = interpolate.interp1d(chis, zs, kind='cubic',
                                          bounds_error=True)

    def load_halos(self, fn, zmin, zmax, ftype='websky', Delta=200.):
        '''Load halo catalog.'''
        self.hfile, self.hftype = fn, ftype
        self.zmin, self.zmax = zmin, zmax
        print('>> Loading halo catalog: {0:s}'.format(fn))
        t0 = time.time()
        print('>> file type: {0:s}'.format(ftype))

        self.halos['Delta'] = Delta

        if ftype == 'websky':
            hf = open(fn)
            n_halo = np.fromfile(hf, count=3, dtype=np.int32)[0]

            hcat = np.fromfile(hf, count=n_halo*10, dtype=np.float32)
            hcat = np.reshape(hcat, (n_halo, 10))
            chi_h = np.sqrt(np.sum(np.power(hcat[:, :3], 2), axis=1))

            print('>> cut with redshift bin [{0:g}, {1:g}]'.format(zmin, zmax))
            chi_min = self.cosmo.comoving_distance(zmin).value  # Mpc/h
            chi_max = self.cosmo.comoving_distance(zmax).value  # Mpc/h
            idx = np.where((chi_min <= chi_h) & (chi_h <= chi_max))[0]

            self.halos['xyz'] = hcat[idx, :3] * self.h0  # Mpc/h (comoving)
            self.halos['vxyz'] = hcat[idx, 3:6]  # km/s
            R_TH = hcat[idx, 6] * self.h0  # Mpc/h
            chi_h = chi_h[idx] * self.h0  # Mpc/h

            print('>> computing redshift of the halos...')
            self.halos['redshift'] = self.chi2z(chi_h)

            print('>> computing mass of the halos...')
            rho_m0 = 2.775e11 * self.Om0  # h^2Msun/Mpc^3
            self.halos['M'] = 4./3. * np.pi * \
                np.power(R_TH, 3) * rho_m0  # Msun/h

            print('>> computing radius of the halos...')
            # matter density at redshift of each halo
            rho_mz = rho_m0 * np.power(1+self.halos['redshift'], 3)
            self.halos['R'] = np.power(
                self.halos['M'] * 3./(4.*np.pi) / self.halos['Delta'] / rho_mz, 1./3.)  # Mpc/h

            self.halos['nhalo'] = self.halos['xyz'].shape[0]

        print(':: Finish loading, number of halos: {0:d}'.format(
            self.halos['nhalo']))
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

        if self.halos['sigv'] == None:  # set velocity dispersion, need empirical relation
            print('>> setting velocity dispersion...')
            self.halos['sigv'] = 0.2 * \
                np.sqrt(np.sum(np.power(self.halos['vxyz'], 2), axis=1))

    def c_meanNC(self, LMmin=13.67, sigma=0.81):
        '''Compute mean number of BCG.'''
        self.LMmin, self.sigma = LMmin, sigma
        print('>> Computing mean number of BCG...')
        t0 = time.time()
        # lnMmin = np.log(10.0) * self.LMmin  # np.log is ln, np.log10 is log
        Mdiff = (LMmin - np.log10(self.halos['M'])) / self.sigma
        self.halos['meanNc'] = 0.5 * special.erfc(Mdiff)
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def c_meanNS(self, LMcut=11.62, LMsat=14.93, alpha=0.43):
        '''Compute mean number of Satellite.'''
        self.LMcut, self.LMsat, self.alpha = LMcut, LMsat, alpha
        print('>> Computing mean number of satellite...')
        t0 = time.time()
        Mcut, Msat = np.power(10, LMcut), np.power(10, LMsat)
        self.halos['meanNs'] = self.halos['meanNc'] * \
            np.power(self.halos['M']/Msat, alpha) * \
            np.exp(-Mcut/self.halos['M'])
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def populate_C(self, gammaHV=1.0):
        '''Populate BCG.'''
        self.gammaHV = gammaHV
        print('>> Populating BCG...')
        t0 = time.time()
        # Bernoulli (0-1) distribution P(1) = <Nc>
        urand = np.random.random(self.halos['nhalo'])
        idx = np.where(urand < self.halos['meanNc'])[0]
        self.gcat['nBCG'] = idx.shape[0]  # total number of BCGs
        print(':: Number of BCGs: {0:d}'.format(self.gcat['nBCG']))

        # get position & velocity of BCGs
        self.gcat['xyz'] = self.halos['xyz'][idx]
        self.gcat['vxyz'] = self.halos['vxyz'][idx] * self.gammaHV

        self.gcat['type'] = np.full(self.gcat['nBCG'], 0, dtype=np.uint8)

        print(':: Finish populating, time elapsed: {0:.2f} s'.format(
            time.time()-t0))

    def populate_S(self, gammaIHV=1.0):
        '''Populate Satellite.'''
        self.gammaIHV = gammaIHV
        print('>> Populating Satellite...')
        t0 = time.time()
        # Poisson distribution P(N;<Ns>)
        NSat = np.random.poisson(lam=self.halos['meanNs'])
        self.gcat['nSat'] = np.sum(NSat)  # total number of satellites
        print(':: Number of Satellites: {0:d}'.format(self.gcat['nSat']))
        print(':: nSat / nBCG = {0:.2f} %'.format(
            100 * self.gcat['nSat']/self.gcat['nBCG']))

        # extend galaxy catalog
        self.gcat['xyz'] = np.row_stack(
            (self.gcat['xyz'], np.zeros(self.gcat['nSat']*3).reshape(self.gcat['nSat'], 3)))
        self.gcat['vxyz'] = np.row_stack(
            (self.gcat['vxyz'], np.zeros(self.gcat['nSat']*3).reshape(self.gcat['nSat'], 3)))
        self.gcat['type'] = np.concatenate(
            (self.gcat['type'], np.full(self.gcat['nSat'], 1, dtype=np.uint8)))
        print('> gcat extended')

        # only populate halos with non-zero satellites
        idx = np.where(NSat > 0)[0]
        NSat = NSat[idx]
        print('> number of halos contains satellites: {0:d}'.format(
            NSat.shape[0]))
        hM = self.halos['M'][idx]
        hxyz = self.halos['xyz'][idx]
        hvxyz = self.halos['vxyz'][idx] * self.gammaHV
        hredshift = self.halos['redshift'][idx]
        hR = self.halos['R'][idx] * 1000.  # change to Kpc for convenience
        hsigv = self.halos['sigv'][idx] * self.gammaIHV

        # concentration parameter for each halo, Duffy et.al.
        conc = 7.85 * np.power(hM/(2e12/self.h0), -0.081) / \
            np.power(1+hredshift, 0.71)
        print('> concentration parameter computed')

        # matter density at each halo's redshift
        rho_m0 = 2.775e11 * self.Om0 * self.h0**2  # Msun/Mpc^3
        rho_mz = rho_m0 * np.power(1+hredshift, 3)

        # get rho_s and Rs for each halo with NFW profile
        rho_s = self.halos['Delta']/3. * rho_mz * \
            np.power(conc, 3) / (np.log(1+conc) - conc/(1+conc))
        Rs = hR / conc
        print('> rho_s & Rs for NFW profile computed')

        # for radial distribution of satellites
        dr = 1.
        nr = np.floor((hR - dr) / dr).astype(int)
        nbin = np.max(nr) + 1
        rr = np.linspace(1., dr * nbin, nbin)
        dV = 4./3. * np.pi * (np.power(rr+dr/2., 3) - np.power(rr-dr/2., 3))
        rands = np.random.random(self.gcat['nSat'])

        # for angualr distribution of satellites
        theta = np.pi * np.random.random(self.gcat['nSat'])
        phi = 2. * np.pi * np.random.random(self.gcat['nSat'])
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        sat_ct = 0
        # get position & velocity of Satellites
        print('> looping over halos')
        for i in range(NSat.shape[0]):
            rhodv = dV * rho_s[i] * dr / \
                (rr * np.power(1+(rr/Rs[i]), 2) / Rs[i])
            cumrho = np.cumsum(rhodv[:nr[i]], dtype=float)
            cumrho = cumrho / cumrho[-1]

            # Gaussian distribution of velocity
            vsat = np.zeros(NSat[i] * 3).reshape(NSat[i], 3)
            for i_ in range(3):
                vsat[:, i_] = np.random.normal(
                    loc=hvxyz[i, i_], scale=hsigv[i], size=NSat[i])

            for j in range(NSat[i]):
                # get radial position of satellite in Mpc
                rsat = rr[np.argmin(np.abs(cumrho - rands[sat_ct]))] / 1000.
                self.gcat['xyz'][self.gcat['nBCG']+sat_ct, 0] = hxyz[i, 0] + \
                    rsat * sin_theta[sat_ct] * cos_phi[sat_ct]
                self.gcat['xyz'][self.gcat['nBCG']+sat_ct, 1] = hxyz[i, 1] + \
                    rsat * sin_theta[sat_ct] * sin_phi[sat_ct]
                self.gcat['xyz'][self.gcat['nBCG']+sat_ct, 2] = hxyz[i, 2] + \
                    rsat * cos_theta[sat_ct]
                # velocity
                self.gcat['vxyz'][self.gcat['nBCG']+sat_ct] = vsat[j]

                sat_ct += 1

        print(':: Finish populating, number of Satellites: {0:d}'.format(
            self.gcat['nSat']))
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def make_header(self):
        '''Make output header.'''
        self.header = 'HOD galaxy catalog\n'
        self.header += 'Halo file: {0:s}, type: {1:s}\n'.format(
            self.hfile, self.hftype)
        self.header += 'Cosmology: Om0 = {0:g}, H0 = {1:g}\n'.format(
            self.Om0, self.H0)
        self.header += 'Parameters: LMmin = {0:g}, sigma = {1:g}, '.format(
            self.LMmin, self.sigma)
        self.header += 'LMcut = {0:g}, LMsat = {1:g}, alpha = {2:g}\n'.format(
            self.LMcut, self.LMsat, self.alpha)
        self.header += 'Redshift range: [{0:g}, {1:g}]\n'.format(
            self.zmin, self.zmax)
        self.header += 'Number - halo: {0:d}, BCG: {1:d}, Sat: {2:d}, Sat / BCG = {3:.2f} %\n'.format(
            self.halos['nhalo'], self.gcat['nBCG'], self.gcat['nSat'], 100*self.gcat['nSat']/self.gcat['nBCG'])

    def write_gcat(self, fn):
        '''Write the galaxy catalog.'''
        print('>> writing to file: {0:s}'.format(fn))
        t0 = time.time()
        header = self.header + \
            'x (Mpc/h)   y (Mpc/h)   z (Mpc/h)   vx (km/s)   vy (km/s)   vz (km/s)  type (0: BCG; 1: Sat)'
        fmt = '%15.7e  %15.7e  %15.7e  %15.7e  %15.7e  %15.7e  %2d'
        np.savetxt(fn, np.column_stack(
            (self.gcat['xyz'], self.gcat['vxyz'], self.gcat['type'])),
            fmt=fmt, header=header)
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def write_rdzw(self, fn):
        '''Write the galaxy catalog in RA, DEC, Z (real & RSD) and weight.'''
        print('>> writing to file: {0:s}'.format(fn))
        t0 = time.time()
        chi_real = np.sqrt(np.sum(np.power(self.gcat['xyz'], 2), axis=1))
        vlos = np.sum(self.gcat['xyz'] * self.gcat['vxyz'], axis=1) / chi_real

        z_real = self.chi2z(chi_real)  # real space redshift

        theta, phi = hp.vec2ang(self.gcat['xyz'])
        ra, dec = utils.get_ra_dec(theta, phi)

        Hz = self.cosmo.efunc(z_real) * 100.  # km/s/(Mpc/h)
        chi_rsd = chi_real + (1. + z_real) * vlos / Hz
        z_rsd = self.chi2z(chi_rsd)

        weight = np.ones(z_rsd.shape[0])

        header = self.header + 'RA   DEC   Z (RSD)   weight'
        fmt = '%15.7e   %15.7e   %15.7e   %3g'
        np.savetxt(fn, np.column_stack((ra, dec, z_rsd, weight)),
                   fmt=fmt, header=header)

        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def hmf(self):
        '''Get the halo mass function.'''
        pass
