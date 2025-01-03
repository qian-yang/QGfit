import numpy as np
import copy
from scipy import integrate, interpolate
import timeit
import pandas as pd
from lmfit import Parameters, minimize
from lmfit.models import SkewedVoigtModel

from kapteyn import kmpfit
from asteval import Interpreter

import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)


from .kmpfit_lmfit import Kmpfit_Lmfit
from . import units
from .units import cal_err_1sigma

##############################################################################

class LineFit():
    def __init__(self, config, wave, flux, fluxerr):

        self.wave = wave
        # NOTE: convert wave into ln
        self.ln_wave = np.log(wave.astype('float'))
        self.flux = flux
        self.fluxerr = fluxerr
        self.config = config

        self.mask_lowSN = config['mask_lowSN']
        self.SN_mask = config['SN_mask']
        self.tie_width = config['tie_width']
        self.tie_shift = config['tie_shift']
        self.tie_flux = config['tie_flux']

        self.comps = self._cut_lines(wave)

        self.line_info = []

    def _cut_lines(self, wave):
        """cut lines in the comp range"""
        lines = self.config['line_list']
        edge = 10  # NOTE: edge 10 A
        ind = (lines['lambda'] > wave.min() + edge) & \
              (lines['lambda'] < wave.max() - edge)
        self.lines = lines[ind]
        # --- unique ---
        ind_uniq = np.unique(self.lines['compname'], return_index=True)[1]
        comps = self.lines.iloc[ind_uniq]
        comps = \
            comps[['compname', 'minwav', 'maxwav']].reset_index(drop=True)
        # --- add parameters ---
        comps['n_gauss'] = 0
        comps['method'] = ''

        comps_new = comps.copy()
        for key, type in zip(self.config['out_attr'], self.config['out_types']):
            comps_new[key] = type

        return comps_new

    def run_fit_line(self):
        config = self.config
        lines_all = None
        for k, comp in self.comps.iterrows():
            comp_name = comp['compname']
            self.comp_name = comp_name
            minv = comp['minwav']
            maxv = comp['maxwav']
            if self.mask_lowSN:
                ind = (self.wave > minv) & (self.wave < maxv) \
                    & (self.flux > - self.fluxerr * self.SN_mask)
            else:
                ind = (self.wave > minv) & (self.wave < maxv)
            wave_comp, flux_comp, fluxer_comp = get_ind(ind,
                self.ln_wave, self.flux, self.fluxerr)
            data_line = (wave_comp, flux_comp, fluxer_comp)
            if len(wave_comp) > 10:  # at least 10 pixels
                comp_name = comp['compname']
                index_line = np.where(self.lines['compname'] == comp_name)[0]
                lines_comp = self.lines.iloc[index_line]
                # ---
                self.comps['n_gauss'][k] = self._get_number(lines_comp)
                # ---
                lines_new = self._create_lines(lines_comp)
                parf = self._init_line_par(lines_new)
                # --- fit ---
                # NOTE: update this to kmpfit
                self.lines_temp = lines_new
                self.ln_lambda_temp = np.array(lines_new['ln_lambda'])

                config['line_mcmc_tp'] = False
                out, method = self.run_fit_line_one(parf, data_line,
                    comp_name) # no mcmc here

                # ======== update lines ========
                kws = create_conti(config)
                params_clean = out.params
                # --- clean lines at boundaries ---
                n_line_vary = 0
                for k_tp, line in lines_new.iterrows():
                    suffix = '_' + str(k_tp)
                    not_vary = not (params_clean['norm' + suffix].vary)
                    # ----
                    type = line['type']
                    minwav = line['minwav']
                    maxwav = line['maxwav']
                    norm = params_clean['norm' + suffix].value
                    ln_lambda = line['ln_lambda']
                    ln_shift = params_clean['ln_shift' + suffix].value
                    center = np.exp(ln_lambda + ln_shift)
                    skew = params_clean['skew' + suffix].value
                    gamma_ratio = params_clean['gamma_ratio' + suffix].value
                    FWHM = params_clean['FWHM' + suffix].value
                    tell_broad = (type == 'broad') & (
                        # (norm <= 0) |
                        (center <= minwav) | (center >= maxwav) |
                        (FWHM >= config['fwhm_broad_max']) | 
                        (FWHM <= config['fwhm_broad_min']) | 
                        (skew <= -abs(config['skew_limit'])) | 
                        (skew >= abs(config['skew_limit'])) |
                        (gamma_ratio >= config['gammar_max_broad']))
                    tell_mix = (type == 'mix') & (
                        # (norm <= 0) |
                        (center <= minwav) | (center >= maxwav) |
                        (FWHM >= config['fwhm_mix_max']) | 
                        (FWHM <= config['fwhm_mix_min']) | 
                        # (skew <= -abs(config['skew_limit'])) | 
                        # (skew >= abs(config['skew_limit'])) |
                        (gamma_ratio >= config['gammar_max_mix']))
                    tell_narrow = (type == 'narrow') & (
                        # (norm <= 0) |
                        (center <= minwav) | (center >= maxwav) |
                        # (FWHM >= config['fwhm_narrow_max']) | 
                        (FWHM <= config['fwhm_narrow_min']) | 
                        # (skew <= -abs(config['skew_limit'])) | 
                        # (skew >= abs(config['skew_limit'])) |
                        (gamma_ratio >= config['gammar_max_narrow']))
                    if not_vary | tell_broad | tell_mix | tell_narrow:
                        # [norm, ln_shift, ln_sigma, skew, gamma_ratio]
                        params_clean['norm' + suffix].value = 0.0
                        params_clean['norm' + suffix].vary = False
                        params_clean['ln_shift' + suffix].value = 0.0
                        params_clean['ln_shift' + suffix].vary = False
                        params_clean['ln_sigma' + suffix].vary = False
                        params_clean['skew' + suffix].value = 0.0
                        params_clean['skew' + suffix].vary = False
                        params_clean['gamma_ratio' + suffix].vary = False
                    else:
                        n_line_vary += 1
                out_clean = self._minimize(params_clean, data_line, comp_name, 
                                            method='leastsq', **kws)
                out = out_clean

                # use BIC to test lines
                if n_line_vary > 1:
                    findex = []
                    bic_all = out_clean.bic
                    params_all = out_clean.params
                    params_tp = params_all.copy()
                    for k_tp, line in lines_new.iterrows():
                        suffix = '_' + str(k_tp)
                        params_modify = params_tp.copy()
                        if params_modify['norm' + suffix].vary:
                            idx = np.array([k_tp])
                            done_test = False
                            # if the fvalue were bounded, test them together
                            if line['findex'] > 0:
                                if line['findex'] not in findex:
                                    findex.append(line['findex'])
                                    ind = lines_new['findex'] == line['findex']
                                    idx = np.array(lines_new[ind].index)
                                else:
                                    done_test = True
                            if not done_test:
                                for id in idx:
                                    suffix_id = '_' + str(id)
                                    # [norm, ln_shift, ln_sigma, skew, gamma_ratio]
                                    params_modify['norm' + suffix_id].value = 0.0
                                    params_modify['norm' + suffix_id].vary = False
                                    params_modify['ln_shift' + suffix_id].value = \
                                        0.0
                                    params_modify['ln_shift' + suffix_id].vary = \
                                        False
                                    params_modify['ln_sigma' + suffix_id].vary = \
                                        False
                                    params_modify['skew' + suffix_id].value = 0.0
                                    params_modify['skew' + suffix_id].vary = False
                                    params_modify['gamma_ratio' + suffix_id].vary \
                                        = False
                                out_modify = self._minimize(params_modify, data_line, 
                                                            comp_name, 
                                                            method='leastsq', **kws)
                                bic_modify = out_modify.bic
                                if bic_modify < bic_all:
                                    params_tp = params_modify
                                    n_line_vary -= 1
                    out = self._minimize(params_tp, data_line, comp_name, method='leastsq', **kws)
                
                # test adding broad lines
                if self.config['test_add_broad']:
                    if n_line_vary > 1:
                        keys = ['norm', 'ln_shift', 'ln_sigma', 'skew', 
                                'gamma_ratio', 'FWHM']
                        bic_all = out.bic
                        params_all = out.params
                        params_tp = params_all.copy()
                        lines_new_tp = lines_new.copy()
                        for k_tp, line in lines_new.iterrows():
                            # broad or mix
                            if (line['type'] != 'narrow') & (line['ngauss'] == 1): 
                                suffix = '_' + str(k_tp)
                                params_modify = params_tp.copy()
                                if params_modify['norm' + suffix].vary:
                                    suffix_tp = '_' + str(len(lines_new_tp))
                                    lines_new_modify = lines_new_tp.copy()
                                    lines_new_modify.loc[
                                        len(lines_new_modify.index)] = line
                                    for key in keys:
                                        one = params_modify[key + suffix]
                                        one_tp = copy.copy(one)
                                        one_tp.name = key + suffix_tp
                                        if key == 'FWHM':
                                            one_tp.expr = \
                                                one_tp.expr.replace(suffix, 
                                                                    suffix_tp)
                                        params_modify.add(one_tp)
                                    self.ln_lambda_temp = \
                                        np.array(lines_new_modify['ln_lambda'])
                                    out_modify = self._minimize(params_modify, 
                                                                data_line, 
                                                                comp_name, 
                                                                method='leastsq', 
                                                                **kws)
                                    bic_modify = out_modify.bic
                                    if bic_modify < bic_all:
                                        params_tp = params_modify
                                        lines_new_tp = lines_new_modify
                        lines_new = lines_new_tp
                        self.ln_lambda_temp = np.array(lines_new['ln_lambda'])
                        self.lines_temp = lines_new_tp
                        out = self._minimize(params_tp, data_line, comp_name, method='leastsq', **kws)
                
                if n_line_vary > 1:
                    if config['line_mcmc']:
                        config['line_mcmc_tp'] = True
                    out, method = self.run_fit_line_one(out.params, data_line,
                        comp_name) # no mcmc here
                else:
                    method = 'leastsq'
                    out = self._minimize(out.params, data_line, comp_name, method=method, **kws)
                self.n_line_vary = n_line_vary

                for key in config['out_attr']:
                    self.comps[key][k] = getattr(out, key)
                self.comps['method'][k] = method

                lines = self._update_res(self.lines_temp, out.params)
                # lines = self._cal_flux(wave_comp, lines)  # QY add
                self.line_info = self._cal_flux_each(wave_comp, lines)

                if lines_all is None:
                    lines_all = lines
                else:
                    lines_all = pd.concat([lines_all, lines],
                        ignore_index=True)
        return (lines_all, self)

    def _run_kmpfit(self, parf, data_line, comp_name):
        if self.config['verbose_line']:
            start = timeit.default_timer()

        KL = Kmpfit_Lmfit(parf)
        self.KL = KL
        par_value, par_parinfo = KL.init_parinfo()

        self.code = 'kmpfit'
        out = kmpfit.Fitter(residuals=self._residual_line,
                                  data=data_line,
                                  params0=par_value,
                                  parinfo=par_parinfo,
                                  maxiter=200,
                                  ftol=1e-10,
                                  xtol=1e-10,
                                  gtol=1e-10)
        out.fit()
        if self.config['verbose_line']:
            stop = timeit.default_timer()
            print('Line: {0} {1} {2}s, redchi2 = {3}'.format(comp_name,
                'kmpfit',
                time_interval(start, stop), str(np.round(out.rchi2_min, 2))))
        return out

    def run_fit_line_one(self, parf, data_line, comp_name):
        kws = create_conti(self.config)
        out_leastsq = self._minimize(parf, data_line, comp_name,
            method='leastsq')
        # out_leastsq = self._minimize(out_leastsq.params, data_line, comp_name,
            # method='leastsq', **kws)
        line_redchi_limit = self.config['line_redchi_limit']
        method = 'leastsq'
        redchi2 = out_leastsq.redchi
        params = out_leastsq.params.copy()
        # NOTE: if leastsq finds a local minima, fit with nelder
        # --- check chi2 ---
        if (self.config['check_nelder']) | (self.config['check_kmpfit']):
            out = out_leastsq
            if self.config['check_kmpfit']:
                if redchi2 > line_redchi_limit:
                    # ------ kmpfit here -----
                    out_kmpfit = self._run_kmpfit(parf, data_line, comp_name)
                    rchi2_min = out_kmpfit.rchi2_min
                    if rchi2_min < redchi2:
                        method = 'kmpfit'
                        out = out_kmpfit
                        redchi2 = out.rchi2_min
                        # update kmpfit results
                        par_kmpfit = out.params
                        for k, key in enumerate(params):
                            params[key].value = par_kmpfit[k]
                    # ------ finish kmpfit ------
            if self.config['check_nelder']:
                if redchi2 > line_redchi_limit:
                    out_nelder = out_new = self._minimize(parf, data_line,
                        comp_name, method='nelder')
                    if out_nelder.redchi < redchi2:
                        method = 'nelder'
                        out = out_nelder
                        params = out_nelder.params.copy()
        if method != 'leastsq':
            out = self._minimize(params, data_line, comp_name,
                method='leastsq', **kws)
        # --- mcmc ---
        if self.config['line_mcmc_tp']:
            emcee_kws = create_line_mcmc(self.config)
            emcee_params = out.params.copy()
            out_mcmc = self._minimize(emcee_params, data_line,
                comp_name, method='emcee', **emcee_kws)
            self.out_mcmc = out_mcmc
        return (out, method)

    def _minimize(self, params, data_line, comp_name, method='leastsq',
                  **kwargs):
        self.code = 'lmfit'
        if self.config['verbose_line']:
            start = timeit.default_timer()

        out = minimize(self._residual_line, params,
            args=(data_line,),
            method=method,
            nan_policy='omit', **kwargs)

        if self.config['verbose_line']:
            stop = timeit.default_timer()
            print('Line: {0} {1} {2}s, redchi2 = {3}'.format(comp_name, method,
                time_interval(start, stop), str(np.round(out.redchi, 2))))
        return out
        # todo: debug when redchi is large, emcee fails

    def _init_line_par(self, lines):
        # --- key function that is different from kmpfit ---
        config = self.config
        parf = Parameters()
        for k_tp, line in lines.iterrows():
            suffix = '_' + str(k_tp)
            parf.add('norm' + suffix, line['norm'], min=0.0, max=1e10)
            if line['type'] == 'broad':  # broad
                ln_sigma_min = 1.7e-8
                ln_sigma_max = 0.029
                ln_sigma_init = 0.0037
                # voff = 0.015
                fwhm_init = 4000
                fwhm_min = config['fwhm_broad_min']
                fwhm_max = config['fwhm_broad_max']
                gammar_init = 1.0
                gammar_max = config['gammar_max_broad']
            if line['type'] == 'mix':  # broad/narrow
                ln_sigma_min = 2.7e-9  # changed this for mix
                ln_sigma_max = 0.029
                ln_sigma_init = 0.0037
                # voff = 0.015
                fwhm_init = 4000
                fwhm_min = config['fwhm_mix_min']
                fwhm_max = config['fwhm_mix_max']
                gammar_init = 1.0
                gammar_max = config['gammar_max_mix']
            if line['type'] == 'narrow':  # narrow
                ln_sigma_min = 0.00015
                ln_sigma_max = 0.0017
                ln_sigma_init = 0.0005
                voff = 0.005
                fwhm_init = 500 # km/s
                fwhm_min = config['fwhm_narrow_min']
                fwhm_max = config['fwhm_narrow_max']
                gammar_init = 1e-2
                # gammar_max = 10.0  # not for wing
                gammar_max = config['gammar_max_narrow']
            voff = line['voff']
            parf.add('ln_shift' + suffix, 0.0, min=-voff, max=+voff)
            parf.add('ln_sigma' + suffix, ln_sigma_init,
                min=ln_sigma_min,
                max=ln_sigma_max)
            parf.add('skew' + suffix, 0.0, min=-abs(config['skew_limit']), 
                     max=abs(config['skew_limit']))
            parf.add('gamma_ratio' + suffix, gammar_init, min=0.0,
                max=gammar_max, vary=True, expr=None)
                # gamma ratio, so 0 < gamma < sigma,
                # free to be between gaussian and voigt
            parf.add('FWHM' + suffix, fwhm_init,
                min=fwhm_min, max=fwhm_max, expr=
                '(0.5346 * (2.0 * gamma_ratio{0} * ln_sigma{0}) + sqrt(0.2166 * (2.0 * gamma_ratio{0} * ln_sigma{0})**2 + (2.3548 * ln_sigma{0})**2)) * 2.99792458e5'.format(suffix))
            # NOTE: v = c * d(lambda) / lambda = c * d(ln_lambda); c in km/s
            # NOTE: it is smaller than the real FWHM when skew != 0
        parf = self._tie_line(parf, lines)
        # After tie, in case gamma was overwritten by _tie_line
        for k_tp, line in lines.iterrows():
            suffix = '_' + str(k_tp)
            if line['skew_var'] == 0:
                parf['skew' + suffix].set(value=0.0, vary=False) # not skewed
            if line['gamma_var'] == 0:
                parf['gamma_ratio' + suffix].set(value=1.0, vary=False) # 0 Gauss 1
        return parf

    def _residual_line(self, parf, data_line):
        wave, flux, fluxerr = data_line
        if self.code == 'lmfit': # parf
            par_value = self._get_par(parf)
        if self.code == 'kmpfit': # pp_conti
            # par_value = parf # change this for kmpfit
            par_value, parf = self.KL.update(parf)  # update format, and tie
        line_model = many_lines(wave, par_value,
            self.ln_lambda_temp)
        residual_line = (flux - line_model) / fluxerr
        residual_line = np.asarray(residual_line, dtype=np.float64)
        return residual_line

    def _get_par(self, parf, type='value'):
        """
        create p_value array in order of
        [norm, ln_shift, ln_sigma, skew, gamma_ratio]
        """
        keys = ['norm', 'ln_shift', 'ln_sigma', 'skew', 'gamma_ratio', 'FWHM']

        ngauss = int(len(parf) / 6.0)
        par_value = []
        for k in range(ngauss):
            suffix = '_' + str(k)
            for key in keys:
                par_value.append(getattr(parf[key + suffix], type))
        return par_value

    def _tie_line(self, parf, lines):
        # --- from self.lines_new ---
        if self.tie_width:
            # The width is controlled by sigma and gamma
            parf = self._tie_par(parf, 'ln_sigma', lines['windex'])
            parf = self._tie_par(parf, 'gamma_ratio', lines['windex'])
        # ---
        if self.tie_shift:
            parf = self._tie_par(parf, 'ln_shift', lines['vindex'])
            parf = self._tie_par(parf, 'skew', lines['vindex'])
        # ---
        if self.tie_flux:
            parf = self._tie_par(parf, 'norm', lines['findex'],
                tie_fvalue=lines['fvalue'])
        return parf

    def _tie_par(self, parf, key, tie, tie_fvalue=None):
        """
        For lmfit
        key = 'norm', 'ln_shift', 'ln_sigma', or 'gamma_ratio'
        tie = lines['windex'], or lines['vindex'], lines['findex']
        tie_fvalue = lines['fvalue'], or default as None.
        tie_factor = tie_fvalue, or default as 1.
        """
        # NOTE: tie_value = 1 or 2, and etc.
        # tie_value, ind_uniq = np.unique(tie, return_index=True)
        tie_value = np.unique(tie)
        # nt = len(tie_value)
        for value in tie_value:
            if value > 0:
                index_tie = np.where(tie == value)[0]
                ind0 = index_tie[0]
                suffix0 = '_' + str(ind0)
                for ind in index_tie:
                    if ind != ind0:
                        if tie_fvalue is None:
                            tie_factor = 1.0
                        else:
                            tie_factor = tie_fvalue[ind] / tie_fvalue[ind0]
                        # ---
                        suffix = '_' + str(ind)
                        parf[key + suffix].set(expr='{0}{1} * {2}'.format(key, suffix0, tie_factor))
        return parf

    def _get_number(self, lines):
        n_gauss = np.sum(lines['ngauss'])
        return n_gauss

    def _create_lines(self, line_list):
        """
        create repeat rows when n_gauss > 1
        """
        lines = None
        for k in range(len(line_list)):
            line = line_list[k: k + 1]
            n_gauss = int(line['ngauss'])
            if n_gauss > 1:
                line = pd.DataFrame(np.repeat(line.values, n_gauss,
                    axis=0), columns=line.columns)
            if lines is None:
                lines = line
            else:
                lines = pd.concat([lines, line], ignore_index=True)
        lines['ln_lambda'] = np.log(lines['lambda'].astype('float'))
        # reset index
        lines.reset_index(drop=True, inplace=True)
        return lines

    # NOTE: modified for lmfit
    def _update_res(self, lines, line_fit):
        par_value = self._get_par(line_fit)
        std_value = self._get_par(line_fit, type='stderr')
        # --- add columns ---
        lines['norm'], lines['ln_shift'], \
            lines['ln_sigma'], lines['skew'], \
            lines['gamma_ratio'], \
            lines['FWHM'] = sep_cols(par_value)
        # --- stderr ---
        lines['norm_stderr'], lines['ln_shift_stderr'], \
            lines['ln_sigma_stderr'], \
            lines['skew_stderr'], \
            lines['gamma_ratio_stderr'], \
            lines['FWHM_stderr'] \
            = sep_cols(std_value)
        if self.config['line_mcmc']:
            if self.n_line_vary > 1:
                out_mcmc = self.out_mcmc
                flatchain = out_mcmc.flatchain
                n_chain = len(flatchain)
                self.n_chain = n_chain
                # ---- add columns ----
                col_keys = ['norm', 'ln_shift', 'ln_sigma', 'skew', 'gamma_ratio', 'FWHM']
                for key in col_keys:
                    lines[key + '_chain'] = pd.Series(dtype='object')

                self._asteval = Interpreter() # init _asteval
                chain_keys = flatchain.keys()
                pars = line_fit.valuesdict()
                pars_keys = list(pars.keys())
                # for c in range(n_chain):
                    # chain_one = flatchain.iloc[c]
                for key in chain_keys:
                    self._asteval.symtable[key] = flatchain[key].values
                # ++++++ add keys not in flatchain ++++++
                for key in pars_keys:
                    if key not in chain_keys:
                        if 'FWHM' not in key:
                            if not line_fit[key].vary: # --- fixed keys ----
                                expr = getattr(line_fit[key], 'expr', None)
                                if expr is None:
                                    flatchain[key] = np.zeros(n_chain) + line_fit[key].value
                                else:
                                    flatchain[key] = self._asteval.eval(expr)
                                self._asteval.symtable[key] = flatchain[key].values
                for key in pars_keys:
                    if key not in chain_keys:
                        if 'FWHM' in key:
                            expr = getattr(line_fit[key], 'expr', None)
                            if expr is not None:
                                flatchain[key] = self._asteval.eval(expr)
                                self._asteval.symtable[key] = flatchain[key].values
                # ------ add keys not in flatchain ------
                chain_keys = flatchain.keys()
                
                ngauss = int(len(line_fit) / 6.0)
                for k in range(ngauss):
                    suffix = '_' + str(k)
                    for key in col_keys:
                        lines[key + '_chain'].iloc[k] = flatchain[key + suffix].values
        return lines

    def cal_line_prop(self, wave_ln, pp_one, ln_lambda):
        config = self.config
        wave_A = np.exp(wave_ln)
        line_one = many_lines(wave_ln, pp_one, ln_lambda)
        flux_int = integrate.trapz(line_one, wave_A)
        if flux_int > 0:
            logL = np.log10(flux_int * config['flux_lum_scale'] * 
                            config['flux_factor'])
        else:
            logL = 0.0
        if (len(pp_one) == 6) & (pp_one[3] == 0.0):
            FWHM = pp_one[5]
            # norm, ln_shift, ln_sigma, skew, gamma_ratio, fwhm = sep_cols(pp_one)
            # '(0.5346 * (2.0 * gamma_ratio{0} * ln_sigma{0}) + sqrt(0.2166 * (2.0 * gamma_ratio{0} * ln_sigma{0})**2 + (2.3548 * ln_sigma{0})**2)) * 2.99792458e5'
        else:
            FWHM_A = cal_fwhm_arr(wave_A, line_one)
            FWHM = FWHM_A / np.exp(ln_lambda) * units.C_KMS
        return (flux_int, logL, FWHM)

    def _cal_flux_each(self, wave_ln, lines):
        if self.config['verbose_line']:
            start = timeit.default_timer()
        linename = lines['linename']
        # names, idx = np.unique(linename, return_index=True)
        names = np.unique(linename)
        nl = len(names)
        line_info = self.line_info
        config = self.config
        for k in range(nl):
            info = {}
            name_one = names[k]
            idx = np.where(linename == name_one)
            line_this = lines.iloc[idx]
            ln_lambda = line_this['ln_lambda']
            ln_lambda_value = ln_lambda.values[0]
            # --- fit result ---
            pp_one = []
            for k, line in line_this.iterrows():
                pp_one += [line['norm'], line['ln_shift'], \
                line['ln_sigma'], line['skew'],
                line['gamma_ratio'], line['FWHM']]
            info['comp'] = self.comp_name
            info['linename'] = name_one
            info['n_gauss'] = len(line_this)
            info['flux'], info['logL'], info['FWHM'] = \
                self.cal_line_prop(wave_ln, pp_one, ln_lambda_value)
            if config['line_mcmc']:
                if self.n_line_vary > 1:
                    n_chain = self.n_chain
                    tparr_flux = np.zeros(n_chain)
                    tparr_logL = np.zeros(n_chain)
                    tparr_FWHM = np.zeros(n_chain)
                    for t in range(n_chain):
                        pp_one = []
                        for k, line in line_this.iterrows():
                            pp_one += [line['norm_chain'][t], \
                                    line['ln_shift_chain'][t], \
                                    line['ln_sigma_chain'][t], \
                                    line['skew_chain'][t], \
                                    line['gamma_ratio_chain'][t], \
                                    line['FWHM_chain'][t]]
                        tparr_flux[t], tparr_logL[t], tparr_FWHM[t] = \
                            self.cal_line_prop(wave_ln, pp_one, ln_lambda_value)
                    info['flux_err'] = cal_err_1sigma(tparr_flux)
                    info['logL_err'] = cal_err_1sigma(tparr_logL)
                    info['FWHM_err'] = cal_err_1sigma(tparr_FWHM)
                else:
                    info['flux_err'] = 0.0
                    info['logL_err'] = 0.0
                    info['FWHM_err'] = 0.0

            line_info.append(info)
        if self.config['verbose_line']:
            stop = timeit.default_timer()
            print('Line prop: {0}s'.format(time_interval(start, stop)))
        return(line_info)

##############################################################################
# Line Models
##############################################################################

def many_lines(xarr, pp, ln_lambda):
    """
    Multiple skewed Gaussian.

    Parameters
    ----------
    xarr : 1-d array or one value.
    pp: multiple skewed Gaussian parameters, in format of \
        [norm1, off1, sigma1, skew1, norm2, off2, sigma2, skew2, etc.]
    ln_lambda: the initial central wavelength

    Returns
    -------
    yarr: the output value at each input xarr value.
    """
    norm, ln_shift, ln_sigma, skew, gamma_ratio, fwhm = sep_cols(pp)
    ln_center = ln_lambda + ln_shift
    ngauss = len(norm)
    yarr = np.zeros_like(xarr)
    if ngauss > 0:
        for norm_one, ln_center_one, ln_sigma_one, skew_one, \
            gamma_ratio_one in zip(norm, ln_center, ln_sigma, skew,
            gamma_ratio):
            yarr += skewed_voigt(xarr, [norm_one, ln_center_one, ln_sigma_one,
                skew_one, gamma_ratio_one])
    return yarr

# ----------------------------

def cal_fwhm_arr(xarr, yarr):
    """
    Calucate the FWHM.

    Parameters
    ----------
    xarr : 1-d array.
    yarr : 1-d array.

    Returns
    -------
    One value: FWHM.
    """
    spline = interpolate.UnivariateSpline(xarr, yarr - np.max(yarr) / 2.0, s=0)
    root_values = spline.roots()
    # Use the min and max values, if there are more than two root points.
    if len(root_values) > 0:
        fwhm_left, fwhm_right = root_values.min(), root_values.max()
        fwhm = np.abs(fwhm_left - fwhm_right)
    else:
        fwhm = 0.0
    return fwhm

def cal_velocity(d_ln_lambda):
    """
    Return velocity in units of km/s
    """
    # velocity = Delta_lambda / lambda * C = d_ln_lambda * C
    velocity = d_ln_lambda * units.C_KMS
    return velocity

def cal_FWHM(log_sigma):
    """
    Return FWHM in units of km/s
    """
    # The lines are in log guass
    # FWHM = Delta_lambda / lambda * C
    # Delta_lambda / lambda = Delta(log_lambda) = 2 * log_sigma * sqrt(2 * ln(2))
    fwhm = units.FACTOR * log_sigma # d_ln_lambda
    fwhm_velocity = cal_velocity(fwhm)
    return fwhm_velocity

def voigt_fwhm(sigma, gamma):
    """
    Return FWHM in units of km/s
    The input parameters was fitted in log scale
    """
    # The lines are in log guass
    # FWHM = Delta_lambda / lambda * C
    # Delta_lambda / lambda = Delta(log_lambda) = width in log lambda
    width_gauss = 2.3548 * sigma
    width_lorentz = 2.0 * gamma
    width_voigt = 0.5346 * width_lorentz + np.sqrt(0.2166 * width_lorentz**2 + width_gauss**2)
    fwhm_velocity = cal_velocity(width_voigt)
    return fwhm_velocity
##############################################################################
# Functional
##############################################################################
def skewed_voigt(xarr, pp):
    mod = SkewedVoigtModel()
    params = mod.make_params(amplitude=pp[0], center=pp[1], sigma=pp[2],
        skew=pp[3])
    params['gamma'].set(value=pp[4] * pp[2], expr=None)
    yarr = mod.eval(params, x=xarr)
    return yarr

def get_ind(ind, *args):
    args = (arg[ind] for arg in args)
    return args

def sep_cols(array):
    """separate columns"""
    array = np.array(array)
    n_gauss = int(len(array) / 6)
    ind = np.arange(n_gauss) * 6
    col1, col2, col3, col4, col5, col6 = array[ind], array[ind + 1], \
        array[ind + 2], array[ind + 3], array[ind + 4], array[ind + 5]
    return (col1, col2, col3, col4, col5, col6)

def com_cols(col1, col2, col3, col4, col5, col6):
    """combine columns"""
    n_gauss = len(col1)
    array = []
    for c1, c2, c3, c4, c5, c6 in zip(col1, col2, col3, col4, col5, col6):
        array.extend([c1, c2, c3, c4, c5, c6])
    return array

def time_interval(start, stop, num=1):
    dt = np.round(stop - start, num)
    dt_str = str(dt)
    return dt_str

def create_conti(config):
    kws = dict(ftol=config['ftol'], xtol=config['xtol'])
    return kws

def create_line_mcmc(config):
    emcee_kws = dict(nwalkers=config['line_nwalkers'],
                     steps=config['line_steps'],
                     burn=config['line_burn'],
                     thin=config['line_thin'],
                     workers=config['line_workers'],
                     is_weighted=True,
                     progress=config['line_progress'],
                     run_mcmc_kwargs={'skip_initial_state_check':True}, 
                     float_behavior='chi2')
    return emcee_kws
