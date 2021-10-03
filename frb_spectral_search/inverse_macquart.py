import math
import numpy as np
import scipy.integrate as integrate
from astropy.cosmology import Planck18 as cosmo

def macquart(z):
    #outputs: Dm_IGM in pc cm^-3, uncertainty is +/- 200 pc cm^-3
    omega_m = 0.3
    omega_lambda = 0.7
    f_IGM = 0.75 #ionization fraction, technically z dependent
    k_IGM = 933 #pc cm^-3 from page 13 https://arxiv.org/pdf/1904.07947.pdf
    return integrate.quad(lambda z: (k_IGM*(1+z)*f_IGM/np.sqrt(omega_m*(1+z)**3+omega_lambda)),0,z)[0]
#second number is error in the result but it's much smaller than +/- 200 (it's like .001)

#def inverse_macquart(dm):
#    #returns z
#    nearest = find_nearest(mc_list, dm)
#    return inverse_m_d(nearest)

def inverse_macquart(dm_eg,  dm_host=116):
    """
    Macquart relation (or its inverse, really) used to cheaply infer z from DM excess
    Parameters
    ----------
    dm_eg : float
    Dispersion measure after subtracting off MW contribution
    cosmo : astorpy.cosmology
    A cosmology for calculating Hubble parameter.
    Y : float
    Helium ionization
    dm_host : float
    Average dispersion measure to assign to each FRB host galaxy. Best fit estimate from James (2021).
    https://www.wolframalpha.com/input/?i=%281+cm%5E3+%2F+1+pc%29+*+%28c+%2F+70+km+%2F+s+%2F+Mpc%29+*+7%2F8+*+1+*+0.045+*+3+*+%2870+km+%2F+s+%2F+Mpc%29%5E2+%2F+%288+*+pi+*+G+*+proton+mass%29
    """
    z_grid = np.linspace(0, 5, num=1000)
    z = 0.5 * (z_grid[1:] + z_grid[:-1])
    dz = np.diff(z_grid)
    # Riemann sum
    DM_lookup = 927.9 * np.cumsum(dz * (1 + z) / cosmo.efunc(z)) + dm_host / (1 + z)
    return z[np.argmin(np.abs(dm_eg - DM_lookup))]
