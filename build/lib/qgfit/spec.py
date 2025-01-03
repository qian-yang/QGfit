import os
import numpy as np
from astropy.io import fits
import sfdmap
import extinction

################################################################################
if "SFD_PATH" not in os.environ:
    raise Exception('Set environment `SFD_PATH`')
else:
    SFD_PATH = os.environ.get('SFD_PATH')

################################################################################
#  SpecProcess
################################################################################

class SpecProcess():
    """
    Process the spectrum.

    Input a dict with parameters.

    Examples
    -----
    input = {'wave':, 'flux':, 'fluxerr':, 'ra':, 'dec':, 'path':, 'name':,}
    or
    input = {'sdss_file':, 'z':, 'ra':, 'dec':, 'path':, 'name':}

    Input Parameters
    -----
    z: float, redshift
    ra: float, ra
    dec: float, dec
    sdss_name : str, the sdss file location
    wave : 1-d array, wavelength
    flux : 1-d array, flux density in unit of 1e-17 erg/s/cm2/A
    fluxerr : 1-d array, flux density uncertainty in unit of 1e-17 erg/s/cm2/A
    mask_arr: 1-d array, the mask array, valid value is 0
    path: str, path for the output
    name: str, filename for the output
    config: dict, config

    Returns
    ----
    The run() function returns the processed object (self).

    Notes
    ---
    """

    def __init__(self, z, config, window, **input):
        # ------ initial inputs --------
        # For complicated input types for one object (fit together).
        self.z = z
        self.window = window
        self.ra = input.get('ra', None)
        self.dec = input.get('dec', None)
        self.sdss_file = input.get('sdss_file', None)
        self.wave = input.get('wave', None)
        self.flux = input.get('flux', None)
        self.fluxerr = input.get('fluxerr', None)
        self.mask_arr = input.get('mask_arr', None)
        self.config = config
        # ------- SDSS or non-SDSS ------
        if self.sdss_file is not None:
            self.sdss = True
            self._read_sdss()
        else:
            self.sdss = False
            self.wdisp = None
            if self.fluxerr is None:
                self.fluxerr = np.ones_like(self.wave)
            if self.mask_arr is None:
                self.mask_arr = np.zeros_like(self.wave)
        # --- check data ---
        if (self.wave is None) | (self.flux is None):
            raise Exception('Input wave, flux, or SDSS file name.')

    def _read_sdss(self):
        hdul = fits.open(self.sdss_file)
        hdr = hdul[0].header
        if self.ra is None:
            self.ra = hdr['PLUG_RA']
        if self.dec is None:
            self.dec = hdr['PLUG_DEC']
        spec = hdul[1].data
        self.wave = 10.0**spec['loglam']
        self.flux = spec['flux']
        ivar_sqrt = np.sqrt(spec['ivar'])
        ivar_sqrt[ivar_sqrt == 0.0] = -1.0
        self.fluxerr = 1.0 / ivar_sqrt
        if self.mask_arr is None:
            self.mask_arr = spec['and_mask']

    def _good_spec(self):
        ind = (self.fluxerr != 0) & ~np.isinf(self.fluxerr)
        self.wave, self.flux, self.fluxerr, self.mask_arr = \
            get_ind(ind, self.wave, self.flux, self.fluxerr, self.mask_arr)
        if self.config.get('mask', False):
            self.mask_arr = self.mask_arr
            ind = self.mask_arr == 0
            self.wave, self.flux, self.fluxerr = \
                get_ind(ind, self.wave, self.flux, self.fluxerr)

    def _deredden_spec(self):
        """ galactic deredden """
        if (self.ra is None) | (self.dec is None):  # If not give, not deredden.
            a_v = 0.0
        else:
            a_v = sfd_av(self.ra, self.dec)
        self.flux = deredden(self.wave, self.flux, a_v)
        self.fluxerr = deredden(self.wave, self.fluxerr, a_v)

    def _rest_spec(self):
        """
        Correct the wavelength, flux density, and the uncertainty to the
        rest frame.
        Reference: PETERSON 1997, Table 9.1,
        AN INTRODUCTION TO ACTIVE GALACTIC NUCLEI
        """
        self.wave_rest = self.wave / (1.0 + self.z)
        self.flux_rest = self.flux * (1.0 + self.z)**3
        self.fluxerr_rest = self.fluxerr * (1.0 + self.z)**3

    def _window(self):
        window = self.window
        wave = self.wave_rest
        select = np.zeros_like(self.wave_rest)
        for i_tp in range(len(window)):
            ind_tp = (wave >= window[i_tp][0]) & (wave <= window[i_tp][1])
            select[ind_tp] = 1
        ind_sel = select > 0
        self.wave_rest_conti, self.flux_rest_conti, self.fluxerr_rest_conti = \
            get_ind(ind_sel, self.wave_rest, self.flux_rest, self.fluxerr_rest)

    def run(self, **input):
        """ spectral process
        Step 1: only use good points (and/or mask spec)
        Step 2: galactic deredden
        Step 3: correct to the rest frame
        Step 4: continuum window
        """
        self.wave = np.asarray(self.wave, dtype=np.float64)
        self.flux = np.asarray(self.flux, dtype=np.float64)
        self.fluxerr = np.asarray(self.fluxerr, dtype=np.float64)
        self._good_spec()
        if self.config.get('do_deredden', True):
            self._deredden_spec()
        self._rest_spec()
        ind = (self.wave_rest >= self.config['wave_range'][0]) & \
              (self.wave_rest <= self.config['wave_range'][1])
        self.wave_rest, self.flux_rest, self.fluxerr_rest = get_ind(ind,
            self.wave_rest, self.flux_rest, self.fluxerr_rest)
        self._window()

        return self

################################################################################
#  Process
################################################################################

def deredden(wave, flux, a_v, r_v=3.1):
    """
    Deredden the spectrum with the
    Cardelli, Clayton & Mathis (1989) extinction function.
    or fitzpatrick99
    """
    a_lambda = extinction.ccm89(wave, a_v, r_v)  # unit='aa'
    flux_new = flux / 10.0**(-0.4 * a_lambda)
    return flux_new

def sfd_av(ra, dec, r_v=3.1):
    """Correct the Galactic extinction
    Reference: https://github.com/kbarbary/sfdmap
    """
    map = sfdmap.SFDMap(SFD_PATH)
    ebv = map.ebv(ra, dec)
    a_v = r_v * ebv
    return a_v

################################################################################
# Functional
################################################################################

def get_ind(ind, *args):
    args = (arg[ind] for arg in args)
    return args
