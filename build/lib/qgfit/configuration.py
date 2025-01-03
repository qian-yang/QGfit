import importlib.resources as pkg_resources
from . import template  # Import the template directory as a module

def config_customized():
    # NOTE: convenient for frequently changed config
    config = {
              'method_line': 'leastsq',
              # =========== tie host ============
              # choice: 'same', 'similar', or 'separate'
              'host_tie': 'same',
              # --- fit host runs ---
              # choice: 'leastsq', 'least_squares', or 'nelder'
              'conti_redchi_limit': 1.5,
              'line_redchi_limit': 1.3,
              # ---
              'conti_mcmc': True,
              'conti_workers': 1,
              'conti_nwalkers': 50,
              'conti_steps': 1000, #1000,
              'conti_burn': 50,
              'conti_thin': 1,
              'conti_progress': False,
              # ---
              'line_mcmc': False,
              'line_workers': 1,
              'line_nwalkers': 50,
              'line_steps': 1000, #1000,
              'line_burn': 50,
              'line_thin': 1,
              'line_progress': False,
              # =========== save ============
              'save_conti_chains': False,
              'save_line_chains': False,
              'plot': True,
              'plot_conti_mcmc_corner': True,
              'plot_line': True,
              'plot_vertical_line': True,
              'color_line_SNR': 5,
              'comp_plot': ['Ha', 'OI', 'HeI', 'NI', 'Hb', 'Hr', 'Hd', 'He', 'OII', 'MgII', 'CIII', 'CIV', 'Lya'],
              'plot_host_type': True,
              'min_x': None,
              'max_x': None,
              'min_y': None,
              'max_y': None,
              # =========== print ============
              'verbose_conti': True,
              'verbose_line': True,
              }
    return config

################################################################################
#  All Default Configurations
################################################################################
# NOTE: Please do not change this.

def config_default():
    config = {
              # --- mask ---
              'wave_range': [1275, 8150],
              'mask': False,  # can customize mask_arr, valid value = 0
              # --- host ---
              'host': True,
              'host_extinction': True,
              # --- process spec ---
              'do_deredden': True,
              'flux_factor': 1e-17,  # SDSS default factor
              # --- cosmology --- # FlatLambdaCDM
              'H0': 70,
              'Om0': 0.3,
              # =======================
              # --- fit continuum ---
              'conti_pl': True,
              'conti_fe_uv': True,
              'conti_fe_op': True,
              'conti_balmer': False,
              'conti_host': True,
              # --- host information ---
              'host_zcut': 1.2,  # turn off host when z > host_zcut
              'host_young': True,
              'host_old': True,
              'host_av': True,
              'host_sigma': True,
              # =========== tie host ============
              'host_tie': 'same',  # choice: 'same', 'similar', or 'separate'
              # NOTE: for 'similar'
              'host_factor_low': 0.5,
              'host_factor_high': 2.0,
              'host_input': False,
              # --- conti technique ---
              # least_squares is fastest, but easy to find a local minima
              'check_nelder': False,
              'check_kmpfit': True,
              'ftol': 1e-8,
              'xtol': 1e-8,
              'conti_redchi_limit': 1.5,  # > 1
              'line_redchi_limit': 1.5,
              # ---
              'conti_mcmc': False,
              'conti_workers': 1,
              'conti_nwalkers': 100,
              'conti_steps': 1000,
              'conti_burn': 200,
              'conti_thin': 1,
              'conti_progress': True,
              # =========== line ============
              'line_fit': True,
              'mask_lowSN': True,
              'SN_mask': 1.0,
              'line_file': './template/lines.csv',
              'line_input': False,
              # --- tie narrow lines ---
              'tie_width': True,
              'tie_shift': True,
              'tie_flux': True,
              # ---
              'line_mcmc': False,
              'line_workers': 1,
              'line_nwalkers': 100,
              'line_steps': 1000,
              'line_burn': 200,
              'line_thin': 1,
              'line_progress': True,
              # ---- line parameters ---
              'fwhm_broad_min': 1200, # km/s
              'fwhm_broad_max': 20000, # km/s
              'fwhm_mix_min': 160,
              'fwhm_mix_max': 20000,
              'fwhm_narrow_min': 160, # km/s 67*2.35
              'fwhm_narrow_max': 1200,
              'gammar_max_broad': 1e5,
              'gammar_max_mix': 1e5,
              'gammar_max_narrow': 10.0,
              'skew_limit': 1, 
              'test_add_broad': True,
              # =========== save ============
              'write': True,
              'save_input': True,  # if not sdss_file, will save the input data
              'save_config': True,  # highly recommend to save this
              # 'save_init': True,
              'save_window': True,
              'save_data': True,  # 15kb to 126kb
              'save_conti_chains': False,
              'save_line_chains': False,
              # =========== calculate physical properties ============
              'prop': True,
              'conti_wave': [1350.0, 2500.0, 3000.0, 4400.0, 5100.0] ,
              # =========== save ============
              'plot': True,
              'plot_conti_mcmc_corner': False,
              'plot_line': True,
              'plot_vertical_line': False,
              'comp_plot': ['Ha', 'Hb', 'MgII', 'CIII', 'CIV', 'Lya'],
              # =========== print ============
              'verbose_conti': True,
              'verbose_line': False,
              }
    return config
