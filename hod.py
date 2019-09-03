'''
Halo Occupation Distribution.
'''
import numpy as np
import collections
from astropy.cosmology import FlatLambdaCDM
from scipy import special


class hod:
    '''HOD main class.'''

    def __init__(self, Om0, h):

        self.Om0 = Om0
        self.h = h
        self.H0 = 100 * h

        # halo properties
        self.halos = collections.OrderedDict()
        self.nhalo = None  # num. of halos
        self.halos['xyz'] = None  # position
        self.halos['vxyz'] = None  # velocity
        self.halos['Rvir'] = None  # virial radius
        self.halos['Mvir'] = None  # virial mass
        self.halos['Mhalo'] = None
        self.halos['redshift'] = None
        self.halos['meanNc'] = None  # mean num. of BCG
        self.halos['meanNs'] = None  # mean num. of Satellite
        self.halos['sigv'] = None  # velocity dispersion

        # mean BCG
        self.LMmin, self.sigma = None, None
        # mean satellite
        self.LM1, self.LM0, self.alpha = None, None, None
        # velocity related
        self.gammaHV, self.gammaIHV = None, None

        # galaxy properties
        self.gcat = collections.OrderedDict()
        self.gcat['xyz'] = None
        self.gcat['vxyz'] = None
        self.gcat['type'] = None  # 0 for BCG; 1 for Sat.
        self.num_BCG = None
        self.num_Sat = None

    def load_halos(self, fn, type, z1=0.0, z2=5.0, dz=0.001):
        '''Load halo catalog.'''
        print('>> Loading halo catalog: {0:s}'.format(fn))
        print('>> file type: {0:s}'.format(type))

        if type == 'websky':
            hf = open(fn)
            n_halo = np.fromfile(hf, count=3, dtype=np.int32)[0]

            hcat = np.fromfile(hf, count=n_halo*10, dtype=np.float32)
            hcat = np.reshape(hcat, (n_halo, 10))

            self.halos['xyz'] = hcat[:, :3]  # Mpc (comoving)
            # to say whether halo co-ordinates are in unit box or not
            self.halos['xyztype'] = 'sky'
            self.halos['vxyz'] = hcat[:, 3:6]  # km/s
            self.halos['Rvir'] = hcat[:, 6]  # virial radius in Mpc

            rho = 2.775e11 * self.Om0 * self.h**2  # Msun/Mpc^3

            print('>> computing mass of the halos...')
            self.halos['Mvir'] = 4./3. * np.pi * \
                np.power(self.halos['Rvir'], 3) * rho
            self.halos['Mhalo'] = np.copy(self.halos['Mvir'])

            print('>> computing redshift of the halos...')
            cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0)
            zs = np.arange(z1, z2+dz, dz)
            chis = cosmo.comoving_distance(zs).value
            chi_h = np.sqrt(np.sum(np.power(self.halos['xyz'], 2), axis=1))
            self.halos['redshift'] = np.interp(chi_h, chis, zs)

            self.nhalo = self.halos['xyz'].shape[0]

        print('>> Finish loading, number of halos: {0:d}'.format(self.nhalo))

        if self.halos['sigv'] == None:  # set velocity dispersion, need empirical relation
            print('>> setting velocity dispersion...')
            self.halos['sigv'] = 0.2 * \
                np.sqrt(np.sum(np.power(self.halos['vxyz'], 2), axis=1))

    def c_meanNc(self, LMmin, sigma):
        '''Compute mean number of BCG.
        Formula: <Nc>(M) = 1/2 * [1 + erf((lnM-lnMmin)/sigma)].'''
        print('>> Computing mean number of BCG...')
        self.LMmin, self.sigma = LMmin, sigma
        lnMmin = np.log(10.0) * self.LMmin  # np.log is ln, np.log10 is log
        Mdiff = (lnMmin - np.log(self.halos['Mhalo'])) / self.sigma
        self.halos['meanNc'] = 0.5 * \
            special.erfc(Mdiff)  # erfc(x) = 1 - erf(x)

    def c_meanNs(self, LM1, LM0, alpha):
        '''Compute mean number of Satellite.
        Formula: <Ns>(M) = <Nc>(M) * ((M-M0)/M1)^alpha for M > M0; 0 for M < m0.'''
        print('>> Computing mean number of satellite...')
        self.LM1, self.LM0, self.alpha = LM1, LM0, alpha
        Mdiff = (self.halos['Mhalo'] - np.power(10, LM0)) / np.power(10, LM1)
        self.halos['meanNs'] = np.zeros(Mdiff.size)
        idx = np.where(Mdiff > 0)
        self.halos['meanNs'][idx] = self.halos['meanNc'][idx] * \
            np.power(Mdiff[idx], alpha)

    def populate_BCG(self, gammaHV=1.0):
        '''Populate BCG.'''
        self.gammaHV = gammaHV
        print('>> Populating BCG...')
        # Bernoulli (0-1) distribution P(1) = <Nc>
        urand = np.random.random(self.nhalo)
        idx = np.where(urand < self.halos['meanNc'])

        # get position & velocity of BCGs
        self.gcat['xyz'] = self.halos['xyz'][idx]
        self.gcat['vxyz'] = self.halos['vxyz'][idx] * self.gammaHV

        self.num_BCG = self.gcat['xyz'].shape[0]  # total number of BCGs
        self.gcat['type'] = np.full(self.num_BCG, 0, dtype=np.uint8)

        print('>> Finish populating, number of BCGs: {0:d}'.format(
            self.num_BCG))

    def populate_Sat(self, gammaIHV=1.0):
        '''Populate Satellite.'''
        self.gammaIHV = gammaIHV
        print('>> Populating Satellite...')
        # Poisson distribution P(N;<Ns>)
        NSat = np.random.poisson(lam=self.halos['meanNs'])
        self.num_Sat = np.sum(NSat)  # total number of satellites

        # extend galaxy catalog
        self.gcat['xyz'] = np.row_stack(
            (self.gcat['xyz'], np.zeros(self.num_Sat*3).reshape(self.num_Sat, 3)))
        self.gcat['vxyz'] = np.row_stack(
            (self.gcat['vxyz'], np.zeros(self.num_Sat*3).reshape(self.num_Sat, 3)))
        self.gcat['type'] = np.row_stack(
            (self.gcat['type'], np.full(self.num_Sat, 1, dtype=np.uint8)))

        # only populate halos with non-zero satellites
        idx = np.where(NSat > 0)
        NSat = NSat[idx]
        hMhalo = self.halos['Mhalo'][idx]
        hxyz = self.halos['xyz'][idx]
        hvxyz = self.halos['vxyz'][idx] * self.gammaHV
        hredshift = self.halos['redshift'][idx]
        hRvir = self.halos['Rvir'][idx]
        hsigv = self.halos['sigv'][idx] * self.gammaIHV

        # concentration parameter for each halo, Duffy et.al.
        conc = 7.85 * np.power(hMhalo/(2e12/self.h), -0.081) / \
            np.power(1+hredshift, 0.71)

        # critical density at each halo's redshift
        H2 = self.H0**2 * \
            (self.Om0 * np.power(1+hredshift, 3) + (1. - self.Om0))
        rho_c = 2.775e7 * H2

        # get rho_0 and Rs for each halo with NFW profile
        rho_0 = 200./3. * rho_c * \
            np.power(conc, 3) / (np.log(1+conc) - conc/(1+conc))
        Rs = hRvir / conc

        # for radial distribution of satellites
        dr = 1.
        nr = np.floor((hRvir - 1.) / dr)
        nbin = np.max(nr) + 1
        rr = np.linspace(1., dr * nbin, nbin)
        dV = 4./3. * np.pi * (np.power(rr+dr/2., 3) - np.power(rr-dr/2., 3))
        rands = np.random.random(self.num_Sat)

        # for angualr distribution of satellites
        theta = np.pi * np.random.random(self.num_Sat)
        phi = 2. * np.pi * np.random.random(self.num_Sat)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        sat_ct = 0
        # get position & velocity of Satellites
        for i in range(NSat.shape[0]):
            rhodv = dV * rho_0[i] * dr / \
                (rr * np.power(1+(rr/Rs[i]), 2) / Rs[i])
            cumrho = np.cumsum(rhodv[:np.int(nr[i])], dtype=float)
            cumrho = cumrho / cumrho[-1]

            # Gaussian distribution of velocity

            for _ in range(NSat[i]):
                # get radial position of satellite in Mpc
                rsat = rr[np.argmin(np.abs(cumrho - rands[sat_ct]))] / 1000.0
                self.gcat['xyz'][self.num_BCG+sat_ct, 0] = hxyz[i, 0] + \
                    rsat * sin_theta[sat_ct] * cos_phi[sat_ct]
                self.gcat['xyz'][self.num_BCG+sat_ct, 1] = hxyz[i, 1] + \
                    rsat * sin_theta[sat_ct] * sin_phi[sat_ct]
                self.gcat['xyz'][self.num_BCG+sat_ct, 2] = hxyz[i, 2] + \
                    rsat * cos_theta[sat_ct]
                # velocity

                sat_ct += 1

        print('>> Finish populating, number of Satellites: {0:d}'.format(
            self.num_Sat))

    def hmf(self):
        '''Get the halo mass function.'''
        pass
