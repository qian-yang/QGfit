import numpy as np
import timeit
import pandas as pd
from lmfit import minimize
from kapteyn import kmpfit
from astropy.cosmology import FlatLambdaCDM

from . import configuration
from . import init_par
from .kmpfit_lmfit import Kmpfit_Lmfit
from .spec import SpecProcess
from .model import conti_model
from .line import LineFit
from .prop import cal_prop
from . import units
from . import write
from . import plot

##############################################################################
#  qfitobj
##############################################################################
# NOTE: model.py and spec.py, and plot.py are exactly the same for kmpfit and lmfit.

class qfitobj():
    """
    Pack multiple spectra into one object.
    Each spectrum will be processed by SpecProcess in spec.py.

    Example
    -------
    obj = [{'sdss_file': file1}, {'sdss_file': file2}]
    result = qfitobj(obj, z=z)
    All parameters in config can be updated by the kwargs in qfitobj
    """
    def __init__(self, spec_input, z=0.0,
            path='./result/', name='test', **kwargs):
        self.z = z
        # ====== config ======
        self.config = configuration.config_default()
        # --- update customized config ---
        config_new = configuration.config_customized()
        for key, value in config_new.items():
            self.config[key] = value
        # --- update input config ---
        for key, value in kwargs.items():
            self.config[key] = value
        # turn off host when z > 1.2
        if z > self.config['host_zcut']:
            self.config['conti_host'] = False
        # --- cosmo ---
        self._add_cosmo()
        # =====================
        # --- continuum fit window ---
        self.window = init_par.conti_window()
        # --- output attributes from lmfit ---
        self.config['out_attr'], self.config['out_types'] = init_par.out_attr()
        if self.config['conti_mcmc']:
            self.config['out_attr_mcmc_conti'], tp = \
                init_par.out_attr_mcmc(save_chains=
                self.config['save_conti_chains'])
        if self.config['line_mcmc']:
                self.config['out_attr_mcmc_line'], \
                self.config['out_types_mcmc_line'] = \
                    init_par.out_attr_mcmc(save_chains=
                    self.config['save_line_chains'])
        # --- prepare data ---
        spectra = self._process_spec(spec_input)
        self.spectra = spectra
        # --- continuum parameter name ---
        self.n_spec = len(spec_input)
        self.config['n_spec'] = self.n_spec # add n_spec to config
        self.input = spec_input
        self.path = path
        self.name = name

        # --- update host_factor = 1 fixed when only 1 spec ---
        if self.n_spec == 1:
            self.config['host_tie'] = 'same'
        # ---
        self.data_conti = self._prep_data(self.spectra)
        self.par_name, self.parf = init_par.get_parf_config(self.n_spec,
            self.config, self.data_conti)
        # for kmpfit
        self.KL = Kmpfit_Lmfit(self.parf)
        self.par_value, self.par_parinfo = self.KL.init_parinfo()

        # --- read line ----
        if self.config['line_fit']:
            self.config['line_list'] = pd.read_csv(self.config['line_file'])
            if isinstance(self.config['line_input'], pd.DataFrame):
                self.config['line_list'] = self._update_line_list(self.config['line_input'])


    def _process_spec(self, spec_input):
        spectra = []
        for input in spec_input:
            spectrum = SpecProcess(self.z, self.config, self.window,
                **input).run()
            spectra.append(spectrum)
        return spectra

    def _prep_data(self, spectra):
        data_conti = []
        for spec in spectra:
            # --- data ---
            data_rest_conti = (spec.wave_rest_conti, spec.flux_rest_conti,
                spec.fluxerr_rest_conti)
            data_conti.append(data_rest_conti)
        return data_conti

    def run_fit_conti(self):
        conti_redchi_limit = self.config['conti_redchi_limit']
        kws = create_conti(self.config)
        # out_leastsq = self._minimize(self.parf, method='leastsq', **kws)
        # --- test ---
        out_leastsq = self._minimize(self.parf, method='leastsq')
        if out_leastsq.redchi > conti_redchi_limit:
            out_leastsq = self._minimize(out_leastsq.params, method='leastsq', 
                                         **kws) # smaller tolerance
        # NOTE: if leastsq finds a local minima, fit with nelder
        # --- check chi2 ---
        redchi2 = out_leastsq.redchi
        params = out_leastsq.params.copy()
        method = 'leastsq'
        if (self.config['check_nelder']) | (self.config['check_kmpfit']):
            out = out_leastsq
            if self.config['check_kmpfit']:
                if redchi2 > conti_redchi_limit:
                    # ------ kmpfit here -----
                    out_kmpfit = self._run_kmpfit()
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
                if redchi2 > conti_redchi_limit:
                    out_nelder = self._minimize(self.parf, method='nelder')
                    if out_nelder.redchi < redchi2:
                        method = 'nelder'
                        out = out_nelder
                        params = out_nelder.params.copy()
        if method != 'leastsq':
            # params = out.params.copy()
            out = self._minimize(params, method='leastsq', **kws)
        # --- mcmc ---
        if self.config['conti_mcmc']:
            emcee_kws = create_conti_mcmc(self.config)
            emcee_params = out.params.copy()
            out_mcmc = self._minimize(emcee_params, method='emcee', **emcee_kws)
            self.out_mcmc = out_mcmc
        return out

    def _minimize(self, params, method='leastsq', **kwargs):
        if self.config['verbose_conti']:
            start = timeit.default_timer()

        self.code = 'lmfit'
        out = minimize(self._residual_conti, params,
            args=(self.data_conti,),
            method=method,
            nan_policy='omit', **kwargs)

        if self.config['verbose_conti']:
            stop = timeit.default_timer()
            print('Continuum: {0} {1}s, redchi = {2}'.format(method,
                time_interval(start, stop), str(np.round(out.redchi, 2))))
        return out

    def _run_kmpfit(self):
        if self.config['verbose_conti']:
            start = timeit.default_timer()

        self.code = 'kmpfit'
        out = kmpfit.Fitter(residuals=self._residual_conti,
                                  data=self.data_conti,
                                  params0=self.par_value,
                                  parinfo=self.par_parinfo,
                                  maxiter=200,
                                  ftol=1e-10,
                                  xtol=1e-10,
                                  gtol=1e-10)
        out.fit()
        if self.config['verbose_conti']:
            stop = timeit.default_timer()
            print('Continuum: {0} {1}s, redchi = {2}'.format('kmpfit',
                time_interval(start, stop), str(np.round(out.rchi2_min, 2))))
        return out

    def _add_cosmo(self):
        self.config['z'] = self.z
        cosmo = FlatLambdaCDM(H0=self.config['H0'], Om0=self.config['Om0'])
        self.config['dl_mpc'] = cosmo.luminosity_distance(self.z).value
        # in unit of Mpc
        self.config['dl_cm'] = units.mpc_cm(self.config['dl_mpc'])
        self.config['flux_lum'] = 4 * np.pi * self.config['dl_cm']**2
        # The same with Shen et al. 2011
        self.config['flux_lum_scale'] = self.config['flux_lum'] / \
            (1 + self.z)**2

    def run_fit(self):
        out = self.run_fit_conti()
        # ---
        self._update_res(out)
        # ---
        lines_result = []
        lines_self = []
        if self.config['line_fit']:
            self.spectra = self._update_spectra()
            for spec in self.spectra:
                line_fit = LineFit(self.config,
                    spec.wave_rest, spec.line_flux, spec.fluxerr_rest)
                lines_all, line_self = line_fit.run_fit_line()
                lines_result.append(lines_all)
                lines_self.append(line_self)
            self.out = out
            self.lines_result = lines_result
            self.lines_self = lines_self
        # --- pack results ---
        data = write.WriteData(self).pack_data()
        if self.config['prop']:
            data['conti']['prop'] = cal_prop(data)
        if self.config['write']:
            filew = self.path + self.name + '.pkl'
            write.write_data(data, filew)
        if self.config['plot']:
            filew = self.path + self.name + '.pdf'
            plot.PlotObj(data, filew).plot_obj()
        return data

    def _update_res(self, out):
        self.pars_set = init_par.get_pars_set(out.params, self.par_name,
            self.n_spec)
        self.stderr_set = init_par.get_pars_set(out.params, self.par_name,
            self.n_spec, type='stderr')
        for key in self.config['out_attr']:
            value = getattr(out, key)
            setattr(self, key, value)

    # the same for kmpfit and lmfit
    def _update_spectra(self):
        spectra = []
        for spec, pars in zip(self.spectra, self.pars_set):
            # --- data ---
            spec.flux_conti = conti_model(spec.wave_rest, pars, self.config)
            spec.line_flux = spec.flux_rest - spec.flux_conti
            spectra.append(spec)
        return spectra

    # for lmfit
    def _residual_conti(self, parf, data_conti):
        if self.code == 'lmfit': # parf
            CODE = init_par
        if self.code == 'kmpfit': # pp_conti
            CODE = self.KL
            par_value, parf = CODE.update(parf)  # update format, and tie
        # ----
        if self.n_spec == 1:
            pars_one = CODE.pick_pars(parf, self.par_name, -1)
            residual_conti = self._residual_one_conti(pars_one, data_conti[0])
        else:
            residual_conti = []
            for k in range(self.n_spec):
                data_one = data_conti[k]
                # pars_one = self.pars_set[k]
                pars_one = CODE.pick_pars(parf, self.par_name, k)
                residual_one = self._residual_one_conti(pars_one, data_one)
                residual_conti.extend(residual_one)
        residual_conti = np.asarray(residual_conti, dtype=np.float64)
        return residual_conti

    # exactly the same for kmpfit and lmfit
    def _residual_one_conti(self, pars, data):
        wave, flux, fluxerr = data
        flux_model = conti_model(wave, pars, self.config)
        residual = (flux - flux_model) / fluxerr
        return residual
    
    def _update_line_list(self, line_input):
        line_list = self.config['line_list'].copy()
        for k_tp, line in line_input.iterrows():
            index_line = np.where(line_list['linename'] == line['linename'])
            line_list.at[int(index_line[0]), 'norm'] = line['norm']
        return line_list


################################################################################
#  Functional
################################################################################
def time_interval(start, stop, num=1):
    dt = np.round(stop - start, num)
    dt_str = str(dt)
    return dt_str

def create_conti(config):
    kws = dict(ftol=config['ftol'], xtol=config['xtol'])
    return kws

def create_conti_mcmc(config):
    emcee_kws = dict(nwalkers=config['conti_nwalkers'] * config['n_spec'],
                     steps=config['conti_steps'],
                     burn=config['conti_burn'],
                     thin=config['conti_thin'],
                     workers=config['conti_workers'],
                     is_weighted=True,
                     progress=config['conti_progress'],
                     run_mcmc_kwargs={'skip_initial_state_check':True}, 
                     float_behavior='chi2')
    return emcee_kws
