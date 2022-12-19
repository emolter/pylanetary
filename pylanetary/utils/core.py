# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
from astropy import convolution
from astropy.coordinates import Angle
import astropy.units as u

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
    
    
def jysr_to_jybm():
    
    return
    

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


def planck(tb, freq):
    '''given temp in K and freq in GHz, returns flux density in Jy sr-1'''
    f = freq*1.0e9 #GHz to Hz
    h = 6.626e-34 #SI
    c = 2.9979e8 #SI
    kb = 1.3806e-23 #SI
    I = (2.0 * h * f**3 / c**2) * (1 / (np.exp(h * f / (kb * tb)) - 1)) #SI
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
    
    
def rj(tb, freq):
    '''given temp in K and freq in GHz, returns flux density in Jy sr-1 by Rayleigh-Jeans approx'''
    return  
    

def beam_area(fwhmx, fwhmy):
    return (np.pi / (4 * np.log(2))) * fwhmx * fwhmy
    
    
def convolve_with_beam(data, beam):
    '''
    Parameters
    ----------
    data : 2-D numpy array, required.
    beam : float/int or 3-element array-like, required.
        if float/int, circular beam assumed, and this sets the fwhm in pixels
        if 3-element array-like, those are (fwhm_x, fwhm_y, theta_deg) 
        in units (pixels, pixels, degrees)
    '''
    
    if np.array(beam).size == 1:
        fwhm_x = beam
        fwhm_y = beam
        theta = 0.0
    else:
        (fwhm_x, fwhm_y, theta) = beam
    psf = convolution.Gaussian2DKernel(fwhm_x / 2.35482004503,
                                        fwhm_y / 2.35482004503,
                                        Angle(theta, unit=u.deg))
    return convolution.convolve_fft(data, psf)