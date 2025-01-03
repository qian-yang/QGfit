import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from astropy.modeling.physical_models import BlackBody
from astropy.convolution import Gaussian1DKernel, convolve_fft
import extinction
from pkg_resources import resource_filename

from . import units

##############################################################################
# Read models and set constants
##############################################################################
datadir = resource_filename('qgfit', 'template')
# ----- Fe templates -----
FE_UV_TEMPLATE = np.genfromtxt(os.path.join(datadir, 'fe_uv.txt'))
FE_OP_TEMPLATE = np.genfromtxt(os.path.join(datadir, 'fe_op.txt'))
# ----- host model -----
SED_DATA = np.load(os.path.join(datadir, 'bc03_ssp_fl_xarr.npz'))
points = (SED_DATA['age'], SED_DATA['wave'])
SED_MODEL = RegularGridInterpolator(points, SED_DATA['sed'],
                                    bounds_error=False, fill_value=0.)

################################################################################
# Continuum models
################################################################################
def conti_model_all_config(xval, pars, config):
    z = config['z']
    factor = config['flux_factor']
    # --- power-law ---
    if config['conti_pl']:
        flux_pl = broken_power_law(xval, pars)
    else:
        flux_pl = np.zeros_like(xval)
    # --- Fe UV ---
    if config['conti_fe_uv']:
        flux_fe_uv = fe_flux(xval, pars, type='uv')
    else:
        flux_fe_uv = np.zeros_like(xval)
    # --- Fe Optical ---
    if config['conti_fe_op']:
        flux_fe_op = fe_flux(xval, pars, type='op')
    else:
        flux_fe_op = np.zeros_like(xval)
    # --- Balmer Continuum ---
    if config['conti_balmer']:
        flux_balmer = balmer_conti(xval, pars)
    else:
        flux_balmer = np.zeros_like(xval)
    # --- Host ---
    if config['conti_host']:
        flux_host = host_sed(xval, pars, config) / factor
    else:
        flux_host = np.zeros_like(xval)
    # --- all ---
    flux_conti = flux_pl + flux_fe_uv + flux_fe_op + flux_balmer + flux_host
    return (flux_conti, flux_pl, flux_fe_uv, flux_fe_op, flux_balmer, flux_host)

def conti_model(xval, pars, config):
    flux_conti, flux_pl, flux_fe_uv, flux_fe_op, flux_balmer, flux_host \
        = conti_model_all_config(xval, pars, config)
    return flux_conti

################################################################################
# host models
################################################################################

def host_npz(wave, age, m_s, config):
    """
    m_s is the stellar mass

    return: flux density in unit of erg/s/cm2/A in the rest frame
    """
    z = config['z']
    yval = np.zeros_like(wave)
    ind_range = (wave >= min(SED_DATA['wave'])) & \
        (wave <= max(SED_DATA['wave']))
    # NOTE: L_lambda is in unit of erg/s/A, where lambda is in the rest frame
    # for one solar mass in the rest frame
    lum_lambda = SED_MODEL((age, wave[ind_range]))
    flux_lum = config['flux_lum']
    flux_observed = m_s * lum_lambda / flux_lum  # convert lum to flux
    # NOTE: according to the fifth equation in Table 9.1 in Peterson's book
    flux_rest = flux_observed * (1.0 + z)**2
    yval[ind_range] = flux_rest
    return yval

def get_host(xval, pars, age, m_s, config):
    xval_new = xval * (1.0 + pars['host_shift'])
    f_host = host_npz(xval_new, age, m_s, config)
    # ----- deredden spec -----
    if config['host_av']:
        f_host = redden(xval_new, f_host, pars['a_v'])
    f_host *= pars['host_factor']
    return f_host

def host_sed_all(xval, pars, config):
    """
    Get host SED, one component for the young stellar population, and
                  one component for the old stellar population.

    Parameters
    ----------
    xval : numpy.ndarray (1-d)
        Wavelengths
    pars : dict
        Continnum parameters

    Returns
    -------
    Host spectrum at each input wavelength.

    Notes
    -----
    Author: Qian Yang, qianyang.astro@gmail.com, July 27, 2022
    If pars['a_v'] > 0, deredden the spectrum with a_v.
    """
    m_s = pars['m_s']
    m_old = m_s * pars['f_old']
    m_young = m_s - m_old
    # ---
    f_young = get_host(xval, pars, pars['age_young'], m_young, config)
    f_old = get_host(xval, pars, pars['age_old'], m_old, config)
    sigpix = pars['sigma'] / 69.0  # km/s
    f_young = conv_spec(f_young, sigpix)
    f_old = conv_spec(f_old, sigpix)
    f_host = f_young + f_old
    # ---
    return (f_host, f_young, f_old)

def host_sed_all_out(xval, pars, config):
    factor = config['flux_factor']
    z = config['z']
    f_host, f_young, f_old = host_sed_all(xval, pars, config)
    fac = 1.0 / factor
    f_host, f_young, f_old = all_multiply(fac, f_host, f_young, f_old)
    return (f_host, f_young, f_old)

def host_sed(xval, pars, config):
    # ----- get models -----
    f_host, f_young, f_old = host_sed_all(xval, pars, config)
    return f_host

def redden(wave, flux, a_v, r_v=3.1):
    """
    Redden the spectrum with the
    Cardelli, Clayton & Mathis (1989) extinction function.

    Parameters
    ----------
    wave : numpy.ndarray (1-d)
        Wavelengths.
    a_v : float
        extinction in magnitudes at characteristic V band wavelength.
    r_v : float
        A_V / E(B-V), default r_v = 3.1

    Returns
    -------
    Reddened spectrum at each input wavelength.
    https://extinction.readthedocs.io/en/latest/api/extinction.ccm89.html
    or fitzpatrick99
    """
    a_lambda = extinction.ccm89(wave, a_v, r_v)  # unit='aa'
    flux_new = flux * 10.0**(-0.4 * a_lambda)
    return flux_new
# ========================== functions from QSOFit ============================
# Version: QSOFit by Yue Shen, https://doi.org/10.5281/zenodo.2565311
#          PyQSOFit by Hengxiao Guo, https://github.com/legolason/PyQSOFit
#

def power_law(xval, norm, slope):
    yval = norm * (xval / 3000.0)**slope
    return yval

def get_power_norm(x0, y0, slope):
    norm = y0 / (x0 / 3000.0)**slope
    return norm

def broken_power_law(xval, pars):
    yval = np.zeros_like(xval)
    xval_broken = 5600
    # Greene & Ho 2005, flux at 5600A equals, a relatively line-free region
    ind_low = (xval <= xval_broken)
    yval[ind_low] = power_law(xval[ind_low], pars['pl_norm'],
        pars['pl_slope1'])
    # for Ha
    ind_high = (xval > xval_broken)
    n_high = np.sum(ind_high)
    if n_high > 10:
        yval_broken = power_law(xval_broken, pars['pl_norm'],
            pars['pl_slope1'])
        norm = get_power_norm(xval_broken, yval_broken, pars['pl_slope2'])
        yval[ind_high] = power_law(xval[ind_high], norm, pars['pl_slope2'])
    else:
        if n_high > 0:
            # do not fit when there is few points at wave > wave_broken
            yval[ind_high] = power_law(xval[ind_high], pars['pl_norm'],
                pars['pl_slope1'])
    return yval

def balmer_conti(xval, pars):
    """
    Balmer continuum from the model of Dietrich+02
    xval = input wavelength, in units of A
    pars = [norm, Te, tau_BE] -- in units of [--, K, --]
    """
    lambda_be = 3646.0  # A
    BB = BlackBody(temperature = pars['balmer_te'] * u.K,
                   scale = 1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr)
                  )
    bbflux = BB(xval * u.AA) * np.pi * u.sr   # in units of ergs/cm2/s/A
    tau = pars['balmer_tau_e'] * (xval / lambda_be)**3
    yval = pars['balmer_norm'] * bbflux * (1.0 - np.exp(-tau))
    yval[xval > lambda_be] = 0.0
    yval = np.asarray(yval, dtype=np.float64)
    return yval

################################################################################

def fe_flux(xval, pars, type='op'):
    """
    Fit the UV/Optical Fe II emission using empirical templates
    # ---------------------------
    # To get sigma in pixel space, QY
    """
    if type == 'uv':
        template = FE_UV_TEMPLATE
        fe_norm = pars['fe_uv_norm']
        fe_fwhm = pars['fe_uv_fwhm']
        fe_shift = pars['fe_uv_shift']
        v_kms = 103.59843733861604  # v_kms = cal_v_kms(wave_fe)
    else:
        template = FE_OP_TEMPLATE
        fe_norm = pars['fe_op_norm']
        fe_fwhm = pars['fe_op_fwhm']
        fe_shift = pars['fe_op_shift']
        v_kms = 106.2907598988584
    # ---------------------------
    wave_fe = 10.0**template[:, 0]
    flux_fe = template[:, 1] * 10.0**15
    xval_new = xval * (1.0 + fe_shift)
    yval = np.zeros_like(xval)
    ind = (xval_new >= min(wave_fe)) & (xval_new <= max(wave_fe))
    if np.sum(ind) > 100:  # at least 100 points, QY
        # ---------------------------
        # QY: VESTERGAARD & WILKES 2001 Eq. (1)
        FWHM_izw1 = 900.0  # the line width of the I Zw 1 spectrum.
        FWHM_QSO = fe_fwhm if fe_fwhm > FWHM_izw1 else 910.0
        FWHM_diff = np.sqrt(FWHM_QSO**2 - FWHM_izw1**2)
        sig_conv = units.fwhm_sigma(FWHM_diff) # in km/s
        # the resolution of the Fe templates are lower than the SDSS spec
        sigpix = sig_conv / v_kms
        flux_fe_conv = conv_spec(flux_fe, sigpix)
        yval[ind] = fe_norm * inter_spec(wave_fe, flux_fe_conv, xval_new[ind])
    return yval

################################################################################
# Functional
################################################################################

def inter_spec(wave, flux, xval):
    spl = interpolate.splrep(wave, flux)
    yval = interpolate.splev(xval, spl)
    return yval

def conv_spec(flux, sigpix):
    if (sigpix > 1):
        kersize = 2 * np.ceil(4 * sigpix) + 1
        kernel = Gaussian1DKernel(sigpix, x_size=kersize)
        yval = convolve_fft(flux, kernel)
    else:
        yval = flux
    return yval

################################################################################
# Not used below
################################################################################
# ----- const -----
C_KMS = 2.99792458e5  # speed of light in km/s

def cal_velscale_wave(wave):
    dw = np.diff(wave, n=1)  # Size of every pixel in Angstrom
    dw = np.append(dw, dw[-1])
    R = np.median(wave / dw)
    velscale = C_KMS / R
    return velscale

def cal_v_kms(wave):
    """
    input: wavelength in unit of A.
    For spectrum with uniform resolution.
    1 / R = Δλ / λ = v / c
    For fe_uv.txt, v = 103.59843733861604 km/s
    For fe_op.txt, v = 106.2907598988584 km/s
    For SDSS spec, v = 69.00601731700418 km/s
    For BOSS spec, v = 68.9942243798227 km/s
    """
    dw = wave[1] - wave[0]
    v_kms = C_KMS * dw/wave[1]
    return v_kms

def cal_velscale_log(loglam):
    """
    `velscale = c*Delta[ln(lam)]` (eq.8 of Cappellari 2017)
    """
    ln_lam_gal = loglam * np.log(10)  # Convert lg --> ln
    # Use full lam range for accuracy
    d_ln_lam_gal = np.diff(ln_lam_gal[[0, -1]]) / (ln_lam_gal.size - 1)
    velscale = C_KMS * d_ln_lam_gal  # Velocity scale in km/s per pixel
    return velscale

def all_multiply(fac, *args):
    args = (arg * fac for arg in args)
    return args
