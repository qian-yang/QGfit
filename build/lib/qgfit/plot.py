import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner
from matplotlib.backends.backend_pdf import PdfPages

from .model import conti_model_all_config, host_sed_all_out
from .line import many_lines, com_cols

class PlotObj():
    def __init__(self, obj, filew):
        self.n_spec = obj['n_spec']
        self.filew = filew
        self.obj = obj
        self.z = obj['z']
        self.config = obj['config']
        self.comp_plot = self.config['comp_plot']
        if (self.config['line_fit'] & self.config['plot_line']):
            self.plot_line = True
        else:
            self.plot_line = False

    def plot_obj(self):
        obj = self.obj
        window = obj['window']
        conti = obj['conti']
        line = obj['line']
        pars_set = conti['pars_set']

        if self.config['plot_conti_mcmc_corner']:
            filew_corner = self.filew.replace('.pdf', '_corner.jpg')
            mcmc = conti['mcmc']
            emcee_plot = corner.corner(mcmc['flatchain'], labels=mcmc['var_names'], truths=mcmc['values'])
            emcee_plot.savefig(filew_corner)

        xlabel = r'$\rm Rest-frame \, Wavelength$ ($\rm \AA$)'
        ylabel = r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
        pdf_pages = PdfPages(self.filew)
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)

        if self.config['line_fit']:
            line_result = line['result']
            line_comps = line['comps']
        for j, data in enumerate(obj['data_rest']):
            wave, flux, fluxerr = data
            # --- wave for plot ---
            edge = 100  # 100 A
            minx, maxx = wave.min() + edge, wave.max() - edge

            pars_one = pars_set[j]
            if self.config['line_fit']:
                lines = line_result[j]
                comps = line_comps[j]
            flux_conti, flux_pl, flux_fe_uv, flux_fe_op, flux_balmer, \
                flux_host = conti_model_all_config(wave,
                pars_one, self.config)
            if self.config['plot_host_type']:
                if self.config['conti_host']:
                    f_host, f_young, f_old = host_sed_all_out(wave,
                        pars_one, self.config)
            nr = 1
            nc = 1
            # --- comps ---
            yy = 10
            if self.plot_line:
                nr = 2
                flux_resi = flux - flux_conti
                compname = list(comps['compname'])
                ind = []
                for c, comp in enumerate(compname):
                    if comp in self.comp_plot:
                        ind.append(c)
                comps_part = comps.iloc[ind]
                comps_part.sort_values(by=['minwav'], ascending=True,
                    inplace=True)
                nc = len(comps_part)
                if (nc > 3):
                    nr = 1 + int(np.ceil(nc / 3))
                    yy = nr * 5
                    nc = 3
                # ---
                # fwhm = lines['FWHM']
                # norm = lines['norm']
                # lines_broad = lines[(fwhm > 1200) & (norm > 0)]
                # lines_narrow = lines[(fwhm <= 1200) & (norm > 0)]
                xval = np.log(wave)
                flux_line = get_line_flux(xval, lines)
            # --- start plot ---
            minv, maxv = get_minmax(smooth(flux, 2), smooth(fluxerr, 2))
            fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize =(15, yy))
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.15)
            # --- conti ---
            ax_conti = plt.subplot(nr, 1, 1)
            if self.config['min_x']:
                minx = self.config['min_x']
            if self.config['max_x']:
                maxx = self.config['max_x']
            if self.config['min_y']:
                minv = self.config['min_y']
            if self.config['max_y']:
                maxv = self.config['max_y']
            ax_conti.set_ylim(minv, maxv)
            ax_conti.set_xlim(minx, maxx)
            ymin = 0.98
            for win in window:
                ax_conti.axvspan(win[0], win[1], ymin=ymin, alpha=0.2,
                    color='blue')
            ax_conti.axvspan(3500, 3686, ymin=ymin, alpha=0.2, color='red')
            # --- plot the continuum panel ---

            # QY add plot line
            if self.plot_line & self.config['plot_vertical_line']:
                dy = np.abs(maxv - minv)
                # dx = np.abs(maxw - minw)
                for k, line_one in lines.iterrows():
                    ax_conti.axvline(x=line_one['lambda'],
                        color='orange', ls='--', lw=0.5)
                    ax_conti.text(line_one['lambda'],
                        maxv + dy * 0.02,
                        line_one['linename'],
                        fontsize=5, rotation=90, ha='center')
                if self.config['color_line_SNR'] > 0:
                    ind = np.where((lines['norm_stderr'] > 0) &
                        # (lines['ln_shift_stderr'] > 0) &
                        (lines['ln_sigma_stderr'] > 0))[0]
                    lines_SNR = lines.iloc[ind]
                    # -------------- narrow --------------
                    ind = np.where(
                        (lines_SNR['SNR'] > self.config['color_line_SNR']) &
                        (lines_SNR['FWHM'] < 1200))[0]
                    if (len(ind) > 0):
                        lines_SNR_narrow = lines_SNR.iloc[ind]
                        for k, line_one in lines_SNR_narrow.iterrows():
                            ax_conti.axvline(x=line_one['lambda'],
                                color='magenta', ls=':', lw=0.5)
                    # -------------- broad --------------
                    ind = np.where(
                        (lines_SNR['SNR'] > self.config['color_line_SNR']) &
                        (lines_SNR['FWHM'] > 1200))[0]
                    if (len(ind) > 0):
                        lines_SNR_broad = lines_SNR.iloc[ind]
                        for k, line_one in lines_SNR_broad.iterrows():
                            ax_conti.axvline(x=line_one['lambda'],
                                color='dodgerblue', ls='--', lw=1.0)
                    ind = np.where(lines['broad'] > 0)[0]
                    if (len(ind) > 0):
                        lines_broad_all = lines.iloc[ind]
                        for k, line_one in lines_broad_all.iterrows():
                            ax_conti.text(line_one['lambda'],
                                maxv + dy * 0.15,
                                str(np.round(line_one['SNR'], 1)),
                                fontsize=10, ha='center')
                            ax_conti.text(line_one['lambda'],
                                maxv + dy * 0.2,
                                str(np.round(line_one['flux'], 1)),
                                fontsize=10, ha='center')

            ax_conti.step(wave, flux - fluxerr, '-', color='gray', alpha=0.3)
            ax_conti.step(wave, flux + fluxerr, '-', color='gray', alpha=0.3)
            ax_conti.step(wave, flux, '-k')  # , label='spectrum'

            if self.config['conti_pl']:
                ax_conti.plot(wave, flux_pl, '--m', label='power law')
            if self.config['conti_fe_uv'] | self.config['conti_fe_op']:
                ax_conti.plot(wave, flux_fe_uv + flux_fe_op, '--g',
                    label='Fe')

            if self.config['conti_host']:
                if self.config['plot_host_type']:
                    if self.config['host_young']:
                        ax_conti.plot(wave, f_young, '-', color='orange',
                            lw=1.0, label='young stellar')
                    if self.config['host_old']:
                        ax_conti.plot(wave, f_old, '-y', lw=1.0,
                            label='old stellar')
                else:
                    ax_conti.plot(wave, flux_host, '-y', lw=1.0,
                        label='stellar')

            ax_conti.plot(wave, flux_conti, ':c', label='continuum')

            if self.plot_line:
                ax_conti.plot(wave, flux_conti + flux_line, '--r',
                    lw=1.0, label='continuum + lines')

            chi2 = str(np.round(float(conti['redchi']), 2))
            ax_conti.text(0.02, 0.9, r'$z=$' + str(self.z), fontsize=16,
                transform=ax_conti.transAxes)
            ax_conti.text(0.02, 0.8, r'$\chi ^2_r=$' + chi2, fontsize=16,
                transform=ax_conti.transAxes)

            ax_conti.legend(loc='best', frameon=False, ncol=2, fontsize=16,
                bbox_to_anchor=(0.15, 0, 0.85, 0.99))

            # --- plot line ---
            if self.plot_line:
                tp = 0
                for c, comp in comps_part.iterrows():
                    if c < len(comps_part):
                        row = int(np.floor(c / 3)) + 1
                        col = int(c % 3)
                        ax_line = axs[row]
                        ax_one = ax_line[col]
                        compname = comp['compname']
                        lines_comp = lines[lines['compname'] == compname]
                        if len(lines_comp) > 0:
                            minx = comp['minwav']
                            maxx = comp['maxwav']

                            ind = (wave > minx) & (wave < maxx)
                            wave_one, flux_one, fluxerr_one = get_ind(ind,
                                wave, flux_resi, fluxerr)
                            xval = np.log(wave_one)
                            miny, maxy = get_minmax(smooth(flux_one, 2), 
                                                    smooth(fluxerr_one, 2))

                            ax_one.set_xlim(minx, maxx)
                            ax_one.set_ylim(miny, maxy)

                            # fwhm = lines_comp['FWHM']
                            # lines_broad = lines_comp[fwhm >= 1200]
                            # lines_narrow = lines_comp[fwhm < 1200]
                            fwhm = lines_comp['FWHM']
                            norm = lines_comp['norm']
                            lines_broad = lines_comp[(fwhm > 1200) & (norm > 0)]
                            lines_narrow = lines_comp[(fwhm <= 1200) & (norm > 0)]

                            flux_line = get_line_flux(xval, lines_comp)

                            ax_one.step(wave_one, flux_one - fluxerr_one, '-',
                                color='gray', alpha=0.3)
                            ax_one.step(wave_one, flux_one + fluxerr_one, '-',
                                color='gray', alpha=0.3)
                            ax_one.step(wave_one, flux_one, '-k', lw=1.7)
                            # , label='spec - conti'
                            ax_one.axhline(y=0, color='gray', linestyle='--')

                            # --- broad ---
                            my_label = 'broad lines'
                            line_broad = np.zeros_like(wave_one)
                            for k, line in lines_broad.iterrows():
                                pp_one = [line['norm'], line['ln_shift'], \
                                    line['ln_sigma'], line['skew'],
                                    line['gamma_ratio'], line['FWHM']]
                                ln_lambda = line['ln_lambda']
                                line_one = many_lines(xval, pp_one, ln_lambda)
                                line_broad += line_one
                                if max(line_one) > 0:
                                    ax_one.plot(wave_one, line_one, '-b', lw=1.5, \
                                        label=my_label)
                                my_label = '_nolegend_'
                            if max(line_broad) > 0:
                                ax_one.plot(wave_one, line_broad, '--c', lw=1.5, \
                                label=my_label)
                            # --- narrow ---
                            my_label = 'narrow lines'
                            for k, line in lines_narrow.iterrows():
                                pp_one = [line['norm'], line['ln_shift'], \
                                    line['ln_sigma'], line['skew'],
                                    line['gamma_ratio'], line['FWHM']]
                                ln_lambda = line['ln_lambda']
                                line_one = many_lines(xval, pp_one, ln_lambda)
                                if max(line_one) > 0:
                                    ax_one.plot(wave_one, line_one, '-g', lw=1.5, \
                                    label=my_label)
                                my_label = '_nolegend_'

                            chi2 = str(np.round(float(comp['redchi']), 2))
                            ax_one.text(0.02, 0.9, r'{}'.format(compname),
                                fontsize=16, transform=ax_one.transAxes)
                            ax_one.text(0.02, 0.8, r'$\chi ^2_r=$'+chi2,
                                fontsize=16, transform=ax_one.transAxes)

                            ax_one.plot(wave_one, flux_line, '--r', lw=1.5,
                                label='broad + narrow')
                            # if (tp == 0):
                            #     ax_one.legend(loc='best', frameon=False,
                            #         ncol=1, fontsize=16,
                            #         bbox_to_anchor=(0.15, 0, 0.85, 0.99))
                            tp += 1
                
                # for c in range(tp, (nr - 1) * nc):
                #     row = int(np.floor(c / 3)) + 1
                #     col = int(c % 3)
                #     ax_line = axs[row]
                #     ax_one = ax_line[col]
                #     ax_one.axis('off')
            # ---
            plt.text(0.5, 0.04, xlabel, fontsize=20, ha='center', transform=plt.gcf().transFigure)
            plt.text(0.05, 0.5, ylabel, fontsize=20, rotation=90, ha='center', rotation_mode='anchor', transform=plt.gcf().transFigure)
            # ---
            pdf_pages.savefig(fig)
        # ---
        pdf_pages.close()

################################################################################
# Functional
################################################################################

def get_line_flux(xval, lines):
    pp_line = com_cols(lines['norm'], lines['ln_shift'], lines['ln_sigma'],
        lines['skew'], lines['gamma_ratio'], lines['FWHM'])
    ln_lambda = np.array(lines['ln_lambda'])
    flux_line = many_lines(xval, pp_line, ln_lambda)
    return flux_line

def get_minmax(arr, err):
    ratio_low = 0.01
    ratio_high = 0.997
    minv = np.quantile(arr - err, ratio_low)
    maxv = np.quantile(arr + err, ratio_high)
    dff = np.abs(maxv - minv)
    minv -= 0.1 * dff
    maxv += 0.2 * dff
    minv = np.min([minv, -1])
    # dv = np.abs(maxv - minv)
    return (minv, maxv)  # , dv

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_ind(ind, *args):
    args = (arg[ind] for arg in args)
    return args
