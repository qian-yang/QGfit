import numpy as np

from .model import broken_power_law
from .units import cal_err_1sigma

def cal_prop(obj):
    conti = obj['conti']
    pars_set = conti['pars_set']
    config = obj['config']
    conti_wave = np.array(config['conti_wave'], dtype=np.float32)
    if config['save_conti_chains']:
        flatchain = obj['conti']['mcmc']['flatchain']
        nf = len(flatchain)
    prop = []
    for j in range(len(pars_set)):
        prop_one = {}
        pars_one = pars_set[j]
        flux_pl = broken_power_law(conti_wave, pars_one)
        for k in range(len(conti_wave)):
            key = 'logL_' + str(int(conti_wave[k]))
            prop_one[key] = np.log10(flux_pl[k] * \
                conti_wave[k] * config['flux_lum_scale'] * \
                config['flux_factor'])
            # 
            if config['save_conti_chains']:
                tparr = np.zeros(nf)
                for t in range(nf):
                    pars_chain = flatchain.iloc[t].to_dict()
                    keys_chain = pars_chain.keys()
                    if ('pl_norm' not in keys_chain):
                        pars_tp = {}
                        matches = [match for match in keys_chain if '_0' in match]
                        for m in matches:
                            m_tp = m.replace('_0', '')
                            pars_tp[m_tp] = pars_chain[m]
                    else:
                        pars_tp = pars_chain
                    flux_pl_tp = broken_power_law(conti_wave[k], pars_tp)
                    tparr[t] = np.log10(flux_pl_tp * \
                                conti_wave[k] * config['flux_lum_scale'] * \
                                config['flux_factor'])
                prop_one[key + '_err'] = cal_err_1sigma(tparr)
        prop.append(prop_one)
    return prop

def MBH_VP06(FWHM, L5100):
    # Vestergaard & Peterson 2006 Eq. 5
    MBH = 10.0**6.91 * (10.0**(L5100 - 44))**0.5 * (FWHM/1000.0)**2
    log_MBH = np.log10(MBH)
    return log_MBH

def MBH_VP06_err(FWHM, L5100, FWHM_err, L5100_err):
    # Vestergaard & Peterson 2006 Eq. 5
    MBH = 10.0**6.91 * (10.0**(L5100 - 44))**0.5 * (FWHM/1000.0)**2
    log_MBH = np.log10(MBH)
    log_MBH_err = np.sqrt((0.5 * L5100_err)**2 + (2.0 / (FWHM/1000.0) / np.log(10.0) * FWHM_err/1000.0)**2)
    return (log_MBH, log_MBH_err)

def MBH_GH05(FWHM, LHa):
    # Greene & Ho 2005 Eq. 6
    MBH = 2.0e6 * (10.0**(LHa-42))**0.55 * (FWHM/1000.0)**2.06
    log_MBH = np.log10(MBH)
    return log_MBH

def MBH_GH05_err(FWHM, LHa, FWHM_err, LHa_err):
    # Greene & Ho 2005 Eq. 6
    MBH = 2.0e6 * (10.0**(LHa-42))**0.55 * (FWHM/1000.0)**2.06
    log_MBH = np.log10(MBH)
    log_MBH_err = np.sqrt((0.55 * LHa_err)**2 + (2.06 / (FWHM/1000.0) / np.log(10.0) * FWHM_err/1000.0)**2)
    return (log_MBH, log_MBH_err)

def MBH_Greene10(FWHM, L5100):
    # Greene et al. 2010 Eq. 1 ; Ha
    MBH = 9.7e6 * (10.0**(L5100-44))**0.519 * (FWHM/1000.0)**2.06
    log_MBH = np.log10(MBH)
    return log_MBH

def MBH_Greene10_err(FWHM, L5100, FWHM_err, L5100_err):
    # Greene et al. 2010 Eq. 1 ; Ha
    MBH = 9.7e6 * (10.0**(L5100-44))**0.519 * (FWHM/1000.0)**2.06
    log_MBH = np.log10(MBH)
    log_MBH_err = np.sqrt((0.519 * L5100_err)**2 + (2.06 / (FWHM/1000.0) / np.log(10.0) * FWHM_err/1000.0)**2)
    return (log_MBH, log_MBH_err)