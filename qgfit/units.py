import numpy as np

################################################################################
# Constants
################################################################################

C_KMS = 2.99792458e5  # speed of light in km/s
C_AS = 2.99792458e5 * 1e13  # speed of light in A/s
FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
# Reference: heasoft-6.29/Xspec/src/XSUtil/Numerics/Numerics.h
KEVTOERG = 1.60217733e-9
KEVTOHZ = 2.4179884076620228e17
KEVTOA = 12.39841974
KEVTOJY = 1.60217733e14
# DEGTORAD = .01745329252
# LIGHTSPEED = 299792.458  #// defined km/s
# AMU = 1.660539040e-24  #// Unified atomic mass unit in g

################################################################################
def mpc_cm(x):
    factor = 3.08567758e24 # NOTE: convert Mpc to cm
    y = x * factor
    return y

################################################################################

def fwhm_sigma(fwhm):
    sigma = fwhm / FACTOR
    return sigma

def sigma_fwhm(sigma):
    fwhm = sigma * FACTOR
    return fwhm

################################################################################
# Flux / Luminosity Units
# ----------------------------------------------
# https://www.stsci.edu/~strolger/docs/UNITS.txt
# http://dmaitra.webspace.wheatoncollege.edu/conversion.html
# [Y erg/s/cm^2/Hz]            = 1000 * [X W/m^2/Hz]
# [Y Jy]                       = 1.0E+26 * [X W/m^2/Hz]
# [Y Jy]                       = 1.0E+23 * [X erg/s/cm^2/Hz]
# 1 Jy = 1e−26 W/m2/Hz
# 1 Jy = 1e-23 erg/s/cm2/Hz
# 1 W/m2/Hz = 1e3 erg/s/cm2/Hz
################################################################################
# NOTE: default fnu unit: erg/s/cm^2/Hz, and default fl unit: erg/s/cm^2/A

def convert_wmhz(x, wave, config):
    """
    input unit: F_nu (W/m^2/Hz) from cat3d
    unit = Jy, mJy, Fnu, or Fl
    wave: in unit of A
    # default: erg/s/cm^2/Hz
    """
    unit = config['unit']
    factor = config['flux_factor']
    flux_lum = config['flux_lum']

    y = wm2_ergscm2(x)
    if unit == 'Fnu':
        y = y
    if unit == 'Jy':
        y = fnu_jy(y) * factor
    if unit == 'mJy':
        y = fnu_mjy(y) * factor
    if unit == 'Fl':
        y = fnu_fl(y, wave)
    if unit == 'flux':
        y = fnu_fl(y, wave) * factor * wave
    if unit == 'lum':
        y = fnu_fl(y, wave) * factor * wave * flux_lum
    return y

def wm2_ergscm2(x):
    """convert unit from F_nu (W/m^2/Hz) to F_nu (erg/s/cm^2/Hz)"""
    y = x * 1e3
    return y

def fnu_jy(x):
    """convert F_nu (erg/s/cm^2/Hz) to F_nu (Jy)"""
    y = x * 1e23
    return y

def jy_mjy(x):
    y = x * 1e3  # mJy is millijansky.
    return y

def mjy_jy(x):
    y = x * 1e-3
    return y

def fnu_mjy(x):
    """convert F_nu (erg/s/cm^2/Hz) to F_nu (mJy)"""
    y = fnu_jy(x)
    y = jy_mjy(y)
    return y

def fnu_fl(fnu, wave):
    """
    convert unit from F_nu (erg/s/cm^2/Hz) to F_lambda (erg/s/cm^2/A)
    wave in A
    l * F_l = v * F_v
    F_l = v / l * Fv = c / l^2 * Fv
    """
    c_as = C_KMS * 1e13 # in A/s
    f_lambda = c_as / wave**2 * fnu  # F_lambda (erg/s/cm^2/A)
    return f_lambda

################################################################################

def convert_fl(x, wave, config):
    """
    input unit: Fl, erg/s/cm^2/A
    unit = Jy, mJy, Fnu, or Fl
    # default same as input: Fl, erg/s/cm^2/A
    """
    unit = config['unit']
    factor = config['flux_factor']

    if unit == 'Fl':
        y = x
    if unit == 'Fnu':
        y = fl_fnu(x, wave)
    if unit == 'Jy':
        y = fl_jy(x * factor, wave)
    if unit == 'mJy':
        y = fl_mjy(x * factor, wave)
    if unit == 'flux':
        y = x * factor * wave
    if unit == 'lum':
        flux_lum = config['flux_lum']
        y = x * factor * wave * flux_lum
    return y

def fl_fnu(f_lambda, wave):
    """
    convert unit from F_lambda (erg/s/cm^2/A) to F_nu (erg/s/cm^2/Hz)
    wave in A

    l * F_l = v * F_v, note: v is freq, l is lambda
    v = c / l
    F_v = l/v * F_l = F_l * l^2 / c
    """
    c_as = C_KMS * 1e13 # in A/s
    f_nu = f_lambda * wave**2 / c_as  # F_nu (erg/s/cm^2/Hz)
    return f_nu

def fl_jy(f_lambda, wave):
    f_nu = fl_fnu(f_lambda, wave)
    y = fnu_jy(f_nu)
    return y

def fl_mjy(f_lambda, wave):
    f_nu = fl_fnu(f_lambda, wave)
    y = fnu_mjy(f_nu)
    return y

################################################################################

def kev_erg(kev):
    y = kev * KEVTOERG
    return y

def erg_freq(erg):
    # Planck's constant
    h = 6.626196e-27  # in units of erg s
    freq = erg / h  # in units of Hz
    return freq

def freq_angstrom(freq):
    c_as = C_KMS * 1e13 # in units of A/s
    wave = c_as / freq  # in units of A
    return wave

def kev_hz(kev):
    """KEVTOHZ"""
    erg = kev_erg(kev)  # in units of erg
    freq = erg_freq(erg)
    return(freq)

def kev_angstrom(kev):
    """KEVTOA"""
    freq = kev_hz(kev)
    wave = freq_angstrom(freq)
    return(wave)

def kev_jy(x):
    """KEVTOJY
    First convert keV to erg.
    Then convert erg/s/cm2/Hz to Jy.
    """
    y = kev_erg(x)
    y = fnu_jy(y)
    return y

def fede_fnu(x):
    # // flux density in keV (ph/cm^2/s/keV)
    # input: flux density * E / dE
    """convert f * E/dE [keV ph/cm2/s/keV] to erg/s/cm2/Hz"""
    y = kev_erg(x) / KEVTOHZ
    return y

def fede_jy(x):
    # fluxdens *= Numerics::KEVTOJY * 1e6 / Numerics::KEVTOHZ;
    y = fede_fnu(x)
    y = fnu_jy(y)
    return y

def fede_mjy(x):
    y = fede_jy(x)
    y *= 1e3
    return y

################################################################################
def cal_err_1sigma(array):
    res = abs(np.quantile(array, 0.84135) - np.quantile(array, 0.15865)) / 2.0
    return(res)

def cal_err_2sigma(array):
    res = abs(np.quantile(array, 0.97275) - np.quantile(array, 0.02275)) / 2.0
    return(res)