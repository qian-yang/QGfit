import numpy as np
from asteval import Interpreter

# parf from lmfit
# parinfo for kmpfit
class Kmpfit_Lmfit():
    def __init__(self, parf, params=None):
        self.par_name = list(parf.keys())
        self.parf = parf
        self.params = self._get_p(params)

        self._asteval = Interpreter()

        self.par_value, self.parinfo = self.init_parinfo()

    def init_parinfo(self, params=None):
        params = self._get_p(params)
        # ----
        parinfo = []
        for key in self.par_name:
            par = self.parf[key]
            vary = par.vary
            fixed = not bool(vary)
            value = par.value
            # ----
            one = {'name': key,
                         'value': value,
                         'limits': (par.min, par.max),
                         'fixed': fixed,
                         'expr':par.expr}
            parinfo.append(one)
        par_value = self._get_pv(parinfo)
        return (par_value, parinfo)

    def update(self, params=None):
        params = self._get_p(params)
        # ----
        # load all parameters in
        for k, key in enumerate(self.par_name):
            self.parinfo[k]['value'] = params[k]
            self._asteval.symtable[key] = params[k]
        # tie pars
        for k, key in enumerate(self.par_name):
            expr = self.parinfo[k]['expr'] # par.expr
            if expr is not None:
                self.parinfo[k]['value'] = self._asteval.eval(expr)
        par_value = self._get_pv(self.parinfo)
        return (par_value, self.parinfo)

    def _get_pv(self, parinfo):
        par_value = [p['value'] for p in parinfo]
        return par_value

    def _get_p(self, params):
        if params is None:
            params = [self.parf[key].value for key in self.par_name]
        else:
            params = params
        return params

    def pick_pars(self, parinfo, par_name, k):
        """
        pars stands for simple parameters in format of, for example,
        {'pl_norm': 0.0,}
        """
        if k < 0:
            suffix = ''
        else:
            suffix = '_' + str(k)
        # ---
        names = np.array([p['name'] for p in parinfo])
        pars = {}
        # print(par_name)
        for key in par_name:
            key_one = key + suffix
            ind = np.where(names == key_one)[0]
            pars[key] = parinfo[ind[0]]['value']  # default as 0
        return pars
