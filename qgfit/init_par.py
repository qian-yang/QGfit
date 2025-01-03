import os
import numpy as np
import pandas as pd
from lmfit import Parameters
from pkg_resources import resource_filename

##############################################################################
# Initial parameters
##############################################################################

datadir = resource_filename('qgfit', 'template')
file = os.path.join(datadir, 'conti_pars2.csv')
PARS = pd.read_csv(file)

def init_conti(parf, config, data_conti, k=-1, method='lmfit'):
    if k < 0:
        suffix = ''
    else:
        suffix = '_' + str(k)
    # --- power-law ---
    pars_host = PARS.copy()
    host_input = config['host_input']
    if isinstance(host_input, dict):
        input_keys = ['host_factor', 'age_young', 'age_old', 'm_s', 'f_old', 'a_v']
        for key in input_keys:
            index = np.where(pars_host['par_name'] == key)
            pars_host.at[int(index[0]), 'value'] = host_input[key]
            pars_host.at[int(index[0]), 'init'] = host_input[key]
            pars_host.at[int(index[0]), 'vary'] = 0
    for p, par in pars_host.iterrows():
        vary = bool(par['vary'])
        value = par['init']
        if not (config[par['model']] & config[par['model_secondary']]):
            vary = False
            value = par['value']
        parf.add(par['par_name'] + suffix, value,
                 min=par['min'], max=par['max'], vary=vary)
    # --- update fe_flux pars ---
    minv = min(data_conti[0])
    maxv = max(data_conti[0])
    # if minv > 3499:
    if minv > 3400:
        parf['fe_uv_norm' + suffix].vary = False
        parf['fe_uv_fwhm' + suffix].vary = False
        parf['fe_uv_shift' + suffix].vary = False
    # if maxv < 3686:
    if maxv < 3800:
        parf['fe_op_norm' + suffix].vary = False
        parf['fe_op_fwhm' + suffix].vary = False
        parf['fe_op_shift' + suffix].vary = False
    return parf

##############################################################################
# Functions For Continuum Parameters
##############################################################################

def update_parf(parf, config, k=-1):
    if k < 0:
        suffix = ''
    else:
        suffix = '_' + str(k)
    # --- update host ---
    if not config['conti_host']:
        # --- tie host ---
        if config['host_tie'] == 'separate':
            # --- totally indenpendent host parameters ---
            parf['host_factor' + suffix].set(value=1.0, vary=False)
        # ---------------------------------------
        # need to tie the host parameters in the residual function
        if config['host_tie'] == 'same':
            parf['host_factor' + suffix].set(value=1.0, vary=False)
        if config['host_tie'] == 'similar':
            parf['host_factor' + suffix].set(value=1.0, vary=True,
                min=config['host_factor_low'],
                max=config['host_factor_high'])

    return parf

def tie_pars(parf, par_name, n_spec, config):
    # --- keys need to tie for the host ---
    tie_keys = ['m_s', 'age_young', 'age_old', 'f_old', 'a_v', 'sigma',
                'host_shift']
    if n_spec > 1:
        suffix0 = '_0'
        for k in range(1, n_spec):
            suffix = '_' + str(k)
            for key in tie_keys:
                parf[key + suffix].set(expr=key + suffix0)
    return parf

def get_parf_config(n_spec, config, data_conti):
    parf = Parameters()
    parf = init_conti(parf, config, data_conti[0], k=-1)
    par_name = get_par_name(parf)
    # ---
    if n_spec == 1:
        parf = update_parf(parf, config, k=-1)
    else:
        parf = Parameters()
        for k in range(n_spec):
            parf = init_conti(parf, config, data_conti[k], k=k)
            parf = update_parf(parf, config, k=k)
        if config['conti_host'] & (config['host_tie'] != 'separate'):
            parf = tie_pars(parf, par_name, n_spec, config)
    return (par_name, parf)

def pick_pars(parf, par_name, k, type='value'):
    """
    pars stands for simple parameters in format of, for example,
    {'pl_norm': 0.0,}
    type = 'value' or 'stderr'
    """
    if k < 0:
        suffix = ''
    else:
        suffix = '_' + str(k)
    pars = {}
    for key in par_name:
        pars[key] = getattr(parf[key + suffix], type, '0.0')  # default as 0
    return pars

def get_pars_set(parf, par_name, n_spec, type='value'):
    pars_set = []
    if n_spec == 1:
        pars_one = pick_pars(parf, par_name, -1, type=type)
        pars_set.append(pars_one)
    else:
        for k in range(n_spec):
            pars_one = pick_pars(parf, par_name, k, type=type)
            pars_set.append(pars_one)
    return pars_set

################################################################################
# Functional
################################################################################
def get_pars(parf):
    pars = parf.valuesdict()
    return pars

def get_par_name(parf):
    pars = get_pars(parf)
    par_name = list(pars.keys())
    return par_name

def del_keys(parf, keys, suffix):
    parf_dict = parf.valuesdict()
    for key in keys:
        key_one = key + suffix
        if key_one in parf_dict:
            del parf[key_one]
    return parf

################################################################################
# Output Parameters from Lmfit
################################################################################
def out_attr():
    keys = ['ndata', 'nvarys', 'nfree',
            'method', 'nfev', 'errorbars',
            'chisqr', 'redchi', 'aic', 'bic',
            'var_names', 'init_values',
            'success', 'message']
    types = [0, 0, 0,
             '', 0, False,
             0.0, 0.0, 0.0, 0.0,
             None, None,
             False, '']
    return (keys, types)

def out_attr_mcmc(save_chains=False):
    keys = ['ndata', 'nvarys', 'nfree',
            'method', 'nfev', 'errorbars',
            'chisqr', 'redchi', 'aic', 'bic',
            'var_names', 'init_values']
    types = [0, 0, 0,
             '', 0, False,
             0.0, 0.0, 0.0, 0.0,
             None, None]
    if save_chains:
        keys.extend(['flatchain', 'lnprob', 'acceptance_fraction'])  # or chain
        types.extend([None, None, None])
    return (keys, types)


##############################################################################
# Continuum Window
##############################################################################

def conti_window():
    window = [
              [1150., 1170.],
              # --- Lya 1215.67 ---
              [1275., 1290.],
              # --- OI 1304.35, SiII 1306.82, CII 1335.3 ---
              [1350., 1360.],
              # --- SiIV 1396.76, OIV] 1402.06 ---
              [1445., 1465.],
              # --- CIV 1549.06 ---
              [1690., 1705.],
              # --- NIII] 1750.26 ---
              [1770., 1810.],
              # --- CIII] 1908.73 ---
              [1970., 2400.],  # broad
              # --- [NeIV] 2423.83 ---
              [2480., 2675.],
              # --- MgII 2798.75 ---
              [2925., 3400.],  # broad
              # === [OII] 3728.48 ===
              ## Fe UV [1250.5470021648975, 3499.049455783253]
              ## Fe Op [3685.6492279033864, 7484.415699727123]
              ## [3500, 3686] no Fe template
              [3446., 3499.],  # QY add for host
              [3686., 3710.],  # QY add for host
              # --- [NeV] 3426.84, [OII] 3728.48 ---
              [3740., 3850.],  # QY modified for host [[3775., 3832.]]
              # --- [NeIII] 3869.85, H{epsilon} 3971.2 ---
              [3898., 3950.],  # QY add for host
              [3978., 4000.],  # QY add for host
              [4000., 4050.],
              # --- H{delta} 4102.89 ---
            #   [4152., 4285.],  # QY modified for Fe and host [4200., 4230.]
              [4170., 4230.],  # QY modified for Fe and host [4200., 4230.]
              # --- H{gamma} 4341.68 ---
            #   [4435., 4640.],
              [4420., 4750.],
              # --- H{beta} 4862.68, [OIII] 4960.3, [OIII] 5008.24 ---
              # --- HeII 4687.02 ---
              [5100., 5535.],
              # --- HeI 5877.29 --- [5805, 5956]
              [6005., 6035.],
              # --- [FeVII] 6087.98 ---
              [6110., 6250.],
              # --- [NII] 6549.85, H{alpha} 6564.61, [NII] 6585.28,
              # --- [SII] 6718.29, [SII] 6732.67 ---
              [6800., 7000.],
              # --- weak: HeI 7067.2, [ArIII] 7137.8 ---
              [7160., 7180.],
              # --- weak: [OII] 7321.48 ---
              [7500., 7800.],
              # --- weak: [NiIII] 7892.1, [FeXI] 7894 ---
              [8050., 8150.],
              ]
    #
    return window
