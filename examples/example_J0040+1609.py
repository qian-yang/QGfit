from astropy.io import fits
from qgfit.fitter import QGfitObj

z = 0.29379
ra = 10.160011
dec = 16.16387

file1 = './J0040+1609/J0040+1609_2000-12-06_SDSS.fits'
file2 = './J0040+1609/J0040+1609_2023-08-25_MMT.fits'
hdu1 = fits.open(file1)
hdu2 = fits.open(file2)
spec1 = hdu1[1].data
spec2 = hdu2[1].data
obj = [{'wave':spec1['wave'], 'flux':spec1['flux'], 'fluxerr':spec1['fluxerr'], 
            'ra':ra, 'dec':dec}, 
        {'wave':spec2['wave'], 'flux':spec2['flux'], 'fluxerr':spec2['fluxerr'], 
            'ra':ra, 'dec':dec}]
fitter = QGfitObj(obj, z=z, path='./J0040+1609/', name='J0040+1609', 
                    conti_mcmc=False, line_fit=True, 
                    comp_plot=['Ha', 'Hb'], host_tie='same')
result = fitter.run_fit()