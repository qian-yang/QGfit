import numpy as np
import timeit
from lmfit import minimize
from kapteyn import kmpfit

from .kmpfit_lmfit import Kmpfit_Lmfit

class Fits():
    """
    Function: Automatically fitting using the result with minimum redchi2 from
        1. lmfit leastsq
        2. kmpfit leastsq (optional)
        3. lmfit nelder (optional)
        4. lmfit mcmc (optional)
    Input:
        parf: initial parameters from lmfit (with needed constrains)
        data: (x, y, yerr)
        resi_func: residual function
    Optional input:
        kwargs:
            for config: verbose, kmpfit, nelder, mcmc;
            for mcmc: nwalkers, steps, burn, thin, workers, progress
    """
    def __init__(self, parf, data, resi_func, **kwargs):
        self.parf = parf
        self.data = data
        self.resi_func = resi_func
        self.config = self._config(**kwargs)
        # for kmpfit
        self._KL = Kmpfit_Lmfit(self.parf)
        self.par_value, self.par_parinfo = self._KL.init_parinfo()

    def _config(self, **kwargs):
        config = {}
        # default configuration
        config['redchi_limit'] = 1.5
        config['verbose'] = True
        config['kmpfit'] = True
        config['nelder'] = True
        config['mcmc'] = False
        # for leastsq
        config['ftol'] = 1e-10
        config['xtol'] = 1e-10
        config['gtol'] = 1e-10
        # for kmpfit
        config['maxiter'] = 200
        # for mcmc
        config['nwalkers'] = 100
        config['steps'] = 10
        config['burn'] = 10
        config['thin'] = 1
        config['workers'] = 1
        config['progress'] = False
        # update input kwargs
        for key, value in kwargs.items():
            config[key] = value
        return config

    def _minimize(self, parf, method='leastsq', **kwargs):
        self.code = 'lmfit'

        if self.config['verbose']:
            start = timeit.default_timer()

        out = minimize(self.resi_func, parf,
            args=(self.data,),
            method=method,
            nan_policy='omit', **kwargs)

        if self.config['verbose']:
            stop = timeit.default_timer()
            print('Continuum: {0} {1}s, redchi = {2}'.format(method,
                time_interval(start, stop), str(np.round(out.redchi, 2))))
        return out

    def _run_kmpfit(self):
        self.code = 'kmpfit'
        self.KL = Kmpfit_Lmfit(self.parf)
        self.par_value, self.par_parinfo = self.KL.init_parinfo()

        if self.config['verbose']:
            start = timeit.default_timer()

        out = kmpfit.Fitter(residuals=self.resi_func,
                                  data=self.data,
                                  params0=self.par_value,
                                  parinfo=self.par_parinfo,
                                  maxiter=self.config['maxiter'],
                                  ftol=self.config['ftol'],
                                  xtol=self.config['xtol'],
                                  gtol=self.config['gtol'])
        out.fit()
        if self.config['verbose']:
            stop = timeit.default_timer()
            print('Continuum: {0} {1}s, redchi = {2}'.format('kmpfit',
                time_interval(start, stop), str(np.round(out.rchi2_min, 2))))
        return out

    def run_fit(self):
        method = 'leastsq'
        leastsq_kws = create_leastsq(self.config)
        out_leastsq = self._minimize(self.parf, method='leastsq',
            **leastsq_kws)
        redchi_limit = self.config['redchi_limit']
        # NOTE: if leastsq finds a local minima, fit with nelder
        # --- check chi2 ---
        if (self.config['nelder']) | (self.config['kmpfit']):
            out = out_leastsq
            if self.config['kmpfit']:
                if out.redchi > redchi_limit:
                    # ------ kmpfit here -----
                    out_kmpfit = self._run_kmpfit()
                    rchi2_min = out_kmpfit.rchi2_min
                    if rchi2_min < out.redchi:
                        method = 'kmpfit'
                        out = out_kmpfit
                    # ------ finish kmpfit ------
            if self.config['nelder']:
                if out.redchi > redchi_limit:
                    out_nelder = self._minimize(self.parf, method='nelder')
                    if out_nelder.redchi < out.redchi:
                        method = 'nelder'
                        out = out_nelder
        if method != 'leastsq':
            params = out.params.copy()
            out = self._minimize(params, method='leastsq')
        # --- mcmc ---
        if self.config['mcmc']:
            emcee_kws = create_mcmc(self.config)
            emcee_params = out.params.copy()
            out_mcmc = self._minimize(emcee_params, method='emcee', **emcee_kws)
            self.out_mcmc = out_mcmc
        return out

################################################################################
#  Functional
################################################################################
def time_interval(start, stop, num=1):
    dt = np.round(stop - start, num)
    dt_str = str(dt)
    return dt_str

def create_leastsq(config):
    leastsq_kws = dict(xtol=config['xtol'], ftol=config['ftol'])
    return leastsq_kws

def create_mcmc(config):
    emcee_kws = dict(nwalkers=config['nwalkers'] * config['n_spec'],
                     steps=config['steps'],
                     burn=config['burn'],
                     thin=config['thin'],
                     workers=config['workers'],
                     is_weighted=True,
                     progress=config['progress'],
                     float_behavior='chi2')
    return emcee_kws
