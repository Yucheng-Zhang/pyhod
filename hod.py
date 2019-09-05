'''
Halo Occupation Distribution.
'''
import numpy as np
import collections
from astropy.cosmology import FlatLambdaCDM
from scipy import special
import time


class hod:
    '''HOD main class.'''

    def __init__(self, Om0=0.31, h0=0.68):

        self.Om0 = Om0
        self.h0 = h0
        self.H0 = 100 * h0

        self.hfile, self.hftype = None, None

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

    def load_halos(self, fn, ftype='websky', Delta=200., z1=0.0, z2=5.0, dz=0.001):
        '''Load halo catalog.'''
        self.hfile, self.hftype = fn, ftype
        print('>> Loading halo catalog: {0:s}'.format(fn))
        t0 = time.time()
        print('>> file type: {0:s}'.format(ftype))

        self.halos['Delta'] = Delta

        if ftype == 'websky':
            hf = open(fn)
            n_halo = np.fromfile(hf, count=3, dtype=np.int32)[0]

            hcat = np.fromfile(hf, count=n_halo*10, dtype=np.float32)
            hcat = np.reshape(hcat, (n_halo, 10))

            self.halos['xyz'] = hcat[:, :3]  # Mpc (comoving)
            self.halos['vxyz'] = hcat[:, 3:6]  # km/s

            print('>> computing redshift of the halos...')
            cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0)
            zs = np.arange(z1, z2+dz, dz)
            chis = cosmo.comoving_distance(zs).value
            chi_h = np.sqrt(np.sum(np.power(self.halos['xyz'], 2), axis=1))
            self.halos['redshift'] = np.interp(chi_h, chis, zs)

            print('>> computing mass of the halos...')
            rho_m0 = 2.775e11 * self.Om0 * self.h0**2  # Msun/Mpc^3
            self.halos['M'] = 4./3. * np.pi * \
                np.power(hcat[:, 6], 3) * rho_m0  # hcat[:, 6]: R_TH

            print('>> computing radius of the halos...')
            # matter density at redshift of each halo
            rho_mz = rho_m0 * np.power(1+self.halos['redshift'], 3)
            self.halos['R'] = np.power(
                self.halos['M'] * 3./(4.*np.pi) / self.halos['Delta'] / rho_mz, 1./3.)  # Mpc

            self.halos['nhalo'] = self.halos['xyz'].shape[0]

        print(':: Finish loading, number of halos: {0:d}'.format(
            self.halos['nhalo']))
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

        if self.halos['sigv'] == None:  # set velocity dispersion, need empirical relation
            print('>> setting velocity dispersion...')
            self.halos['sigv'] = 0.2 * \
                np.sqrt(np.sum(np.power(self.halos['vxyz'], 2), axis=1))

    def c_meanNc(self, LMmin=13.67, sigma=0.81):
        '''Compute mean number of BCG.'''
        self.LMmin, self.sigma = LMmin, sigma
        print('>> Computing mean number of BCG...')
        t0 = time.time()
        # lnMmin = np.log(10.0) * self.LMmin  # np.log is ln, np.log10 is log
        Mdiff = (LMmin - np.log10(self.halos['M'])) / self.sigma
        self.halos['meanNc'] = 0.5 * special.erfc(Mdiff)
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def c_meanNs(self, LMcut=11.62, LMsat=14.93, alpha=0.43):
        '''Compute mean number of Satellite.'''
        self.LMcut, self.LMsat, self.alpha = LMcut, LMsat, alpha
        print('>> Computing mean number of satellite...')
        t0 = time.time()
        Mcut, Msat = np.power(10, LMcut), np.power(10, LMsat)
        self.halos['meanNs'] = self.halos['meanNc'] * \
            np.power(self.halos['M']/Msat, alpha) * \
            np.exp(-Mcut/self.halos['M'])
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def populate_BCG(self, gammaHV=1.0):
        '''Populate BCG.'''
        self.gammaHV = gammaHV
        print('>> Populating BCG...')
        t0 = time.time()
        # Bernoulli (0-1) distribution P(1) = <Nc>
        urand = np.random.random(self.halos['nhalo'])
        idx = np.where(urand < self.halos['meanNc'])
        self.gcat['nBCG'] = idx[0].shape[0]  # total number of BCGs
        print(':: Number of BCGs: {0:d}'.format(self.gcat['nBCG']))

        # get position & velocity of BCGs
        self.gcat['xyz'] = self.halos['xyz'][idx]
        self.gcat['vxyz'] = self.halos['vxyz'][idx] * self.gammaHV

        self.gcat['type'] = np.full(self.gcat['nBCG'], 0, dtype=np.uint8)

        print(':: Finish populating, time elapsed: {0:.2f} s'.format(
            time.time()-t0))

    def populate_Sat(self, gammaIHV=1.0):
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
        idx = np.where(NSat > 0)
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

    def write_gcat(self, fn):
        '''Output the galaxy catalog.'''
        t0 = time.time()
        header = 'HOD galaxy catalog\n'
        header += 'Halo file: {0:s}, type: {1:s}\n'.format(
            self.hfile, self.hftype)
        header += 'Parameters: LMmin = {0:g}, sigma = {1:g}, '.format(
            self.LMmin, self.sigma)
        header += 'LMcut = {0:g}, LMsat = {1:g}, alpha = {2:g}\n'.format(
            self.LMcut, self.LMsat, self.alpha)
        header += ' # halo: {0:d}, # BCG: {1:d}, # Sat: {2:d}\n'.format(
            self.halos['nhalo'], self.gcat['nBCG'], self.gcat['nSat'])
        header += 'x   y   z   vx   vy   vz   type (0: BCG; 1: Sat)'
        fmt = '%15.7e  %15.7e  %15.7e  %15.7e  %15.7e  %15.7e  %2d'
        np.savetxt(fn, np.column_stack(
            (self.gcat['xyz'], self.gcat['vxyz'], self.gcat['type'])),
            fmt=fmt, header=header)
        print(':: Galaxy catalog saved: {0:s}'.format(fn))
        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

    def hmf(self):
        '''Get the halo mass function.'''
        pass
