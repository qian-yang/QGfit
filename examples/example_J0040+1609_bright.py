from astropy.io import fits
from qgfit.fitter import QGfitObj

z = 0.29379
ra = 10.160011
dec = 16.16387

file = './J0040+1609/J0040+1609_2023-08-25_MMT.fits'
hdu = fits.open(file)
spec = hdu[1].data
obj = [{'wave':spec['wave'], 'flux':spec['flux'], 'fluxerr':spec['fluxerr'], 
            'ra':ra, 'dec':dec}]
fitter = QGfitObj(obj, z=z, path='./J0040+1609/', name='J0040+1609_bright', 
                    conti_mcmc=False, line_fit=True, 
                    comp_plot=['Ha', 'Hb'])
result = fitter.run_fit()