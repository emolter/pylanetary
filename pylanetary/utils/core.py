# Licensed under a ??? style license - see LICENSE.rst
import numpy as np


## TO DO: make these accept astropy units


def jybm_to_jysr(I, beamx, beamy):
    '''
    Parameters
    ----------
    I: numpy array, int, or float in Jy/beam
    beamx, beamy: FWHM of Gaussian beam in arcsec
    '''
    beamA = (np.pi / (4 * np.log(2))) * self.beamx * self.beamy #in arcsec
    return 4.25e10*x/beamA
    

def jybm_to_tb(I, freq, beamx, beamy):
    '''
    from science.nrao.edu/facilities/vla/proposing/TBconv
        I in Jy/bm, freq in GHz, returns tb in Kelvin
    
    Parameters
    ----------
    I: numpy array, int, or float in Jy/beam
    freq: float or int, units GHz
    beamx, beamy: FWHM of Gaussian beam in arcsec
    '''
    return (1e3 * 1.22e3) * I / (freq**2 * beamx * beamy)
    

def tb_to_jybm(Tb, freq, beamx, beamy):
    '''
    from science.nrao.edu/facilities/vla/proposing/TBconv
        I in Jy/bm, freq in GHz, returns tb in Kelvin
    
    Parameters
    ----------
    Tb: numpy array, int, or float in K
    freq: float or int, units GHz
    beamx, beamy: FWHM of Gaussian beam in arcsec
    '''
    return Tb * (freq**2 * beamx * beamy) / (1e3 * 1.22e3)


def planck(tb, freq_ghz):
    '''given temp in K and freq in GHz, returns flux density in Jy sr-1'''
    freq = freq_ghz*1.0e9 #GHz to Hz
    h = 6.626e-34 #SI
    c = 2.9979e8 #SI
    kb = 1.3806e-23 #SI
    I = (2.0 * h * freq**3 / c**2) * (1 / (np.exp(h * freq / (kb * tb)) - 1)) #SI
    return I*1e26 #Jy sr-1


def inverse_planck(B, freq):
    '''given flux density in Jy sr-1 and freq in GHz, returns temp in K'''
    f = freq*1.0e9
    I = B/1e26
    h = 6.626e-34 #SI
    c = 2.9979e8 #SI
    kb = 1.3806e-23 #SI
    tb = (h*f/kb) * (1/ np.log(2.0 * h * f**3 / (c**2 * I) + 1))
    return tb #K

    
def inverse_rj(B, freq):
    '''given flux density in Jy sr-1 and freq in GHz, returns temp in K by Rayleigh-Jeans approx'''
    f = freq*1.0e9
    I = B/1e26
    h = 6.626e-34 #SI
    c = 2.9979e8 #SI
    kb = 1.3806e-23 #SI
    tb = (c**2/(2*kb*f**2)) * I
    return tb    
    

def beam_area(fwhmx, fwhmy):
    return (np.pi / (4 * np.log(2))) * fwhmx * fwhmy