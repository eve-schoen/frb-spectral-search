import math
import numpy as np
import scipy.integrate as integrate

def macquart(z):
    #outputs: Dm_IGM in pc cm^-3, uncertainty is +/- 200 pc cm^-3
    omega_m = 0.3 
    omega_lambda = 0.7 
    f_IGM = 0.75 #ionization fraction, technically z dependent
    k_IGM = 933 #pc cm^-3 from page 13 https://arxiv.org/pdf/1904.07947.pdf
    return integrate.quad(lambda z: (k_IGM*(1+z)*f_IGM/np.sqrt(omega_m*(1+z)**3+omega_lambda)),0,z)[0] 
#second number is error in the result but it's much smaller than +/- 200 (it's like .001)

inverse_m_d = {}
i = 0
while i < 3:
    inverse_m_d[macquart(i)] = i
    i += .01
mc_list = np.asarray(sorted(inverse_m_d.keys()))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
def inverse_macquart(dm):
    #returns z
    nearest = find_nearest(mc_list, dm)
    return inverse_m_d(nearest)
