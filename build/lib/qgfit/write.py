import pickle
import pandas as pd

from . import init_par
from .units import cal_err_1sigma

class WriteData():
    def __init__(self, obj):
        self.config = obj.config
        self.path = obj.path
        self.name = obj.name
        self.obj = obj

    def write(self):
        data = self.pack_data()
        filew = self.path + self.name + '.pkl'
        write_data(data, filew)

    def pack_data(self):
        obj = self.obj
        data = {}
        data['z'] = obj.z
        data['n_spec'] = obj.n_spec
        if self.config['save_input']:
            data['input'] = obj.input
        if self.config['save_config']:
            data['config'] = obj.config
        if self.config['save_window']:
            data['window'] = obj.window
        # --- default: save continuum fitting parameters ---
        conti = {}

        conti['par_name'] = obj.par_name

        conti['pars_set'] = obj.pars_set
        # conti['xerror_set'] = obj.xerror_set
        conti['stderr_set'] = obj.stderr_set

        for key in self.config['out_attr']:
            conti[key] = getattr(obj, key)

        if self.config['conti_mcmc']:
            mcmc = {}
            out_mcmc = obj.out_mcmc
            # conti['mcmc'] = obj.out_mcmc
            keys = self.config['out_attr_mcmc_conti']
            for key in keys:
                value = getattr(out_mcmc, key)
                mcmc[key] = value
            if self.config['save_conti_chains']:
                mcmc['values'] = [out_mcmc.params[key].value for key in
                    mcmc['var_names']]  # for corner plot

            mcmc['pars_set'] = init_par.get_pars_set(out_mcmc.params,
                obj.par_name, obj.n_spec)
            mcmc['stderr_set'] = init_par.get_pars_set(out_mcmc.params,
                obj.par_name, obj.n_spec, type='stderr')

            conti['mcmc'] = mcmc

            # --- quantile error ---
            if self.config['save_conti_chains']:
                error = {}
                flatchain = out_mcmc.flatchain
                par_name = mcmc['var_names']
                for key in par_name:
                    error[key + '_err'] = cal_err_1sigma(flatchain[key].values)
                mcmc['mcmc_err'] = error

        data['conti'] = conti
        # --- save line fitting parameters ---
        line = {}
        if self.config['line_fit']:
            line['result'] = obj.lines_result
            # ---
            lines_self = obj.lines_self
            comps = []
            line_info = []
            for j in range(obj.n_spec):
                comps.append(lines_self[j].comps)
                # --- line_info ---
                line_info_one = lines_self[j].line_info
                info_one = line_info_one[0]
                keysList = list(info_one.keys())
                line_info_df = pd.DataFrame(line_info_one, columns=keysList)
                line_info.append(line_info_df)
            line['comps'] = comps
            line['line_info'] = line_info
            # ---
            # if self.config['line_mcmc']:
            #     comps_mcmc = []
            #     for j in range(obj.n_spec):
            #         comps_mcmc.append(lines_self[j].comps_mcmc)
            #     line['comps_mcmc'] = comps_mcmc
            # ---
            # if self.config['line_mcmc']:
            #     comps_mcmc = []
            #     for j in range(obj.n_spec):
            #         comps_mcmc.append(lines_self[j].mcmc)
            #     line['comps_mcmc'] = comps_mcmc

        data['line'] = line
        # --- save data ---
        if self.config['save_data']:
            data_rest = []
            for spec in obj.spectra:
                data_rest.append(
                    (spec.wave_rest, spec.flux_rest, spec.fluxerr_rest))
            data['data_rest'] = data_rest
        return data

################################################################################
# Functional
################################################################################

def write_data(data, filew):
    with open(filew, 'wb') as file:
        pickle.dump(data, file)

def read_data(filer):
    with open(filer, 'rb') as file:
        data = pickle.load(file)
    return data
