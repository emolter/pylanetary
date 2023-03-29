# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
from astropy import convolution
from astropy.coordinates import Angle
import astropy.units as u
import importlib
from scipy.interpolate import interp1d

'''
To do:
* write tests for convolve_with_beam (although this is somewhat tested in test_planetnav)
* make these accept astropy units
* add I/F calculations... or maybe that goes somewhere else
'''

def beam_area(beamx, beamy):
    '''
    beamx, beamy: FWHM of Gaussian beam in arcsec
    '''
    return (np.pi / (4 * np.log(2))) * beamx * beamy
    

def jybm_to_jysr(I, beamx, beamy):
    '''
    Parameters
    ----------
    I: numpy array, int, or float in Jy/beam
    beamx, beamy: FWHM of Gaussian beam in arcsec
    '''
    beamA = beam_area(beamx, beamy) #in arcsec
    return 4.25e10*I/beamA
    
    
def jysr_to_jybm(I, beamx, beamy):
    '''
    Parameters
    ----------
    I: numpy array, int, or float in Jy/sr
    beamx, beamy: FWHM of Gaussian beam in arcsec
    '''
    beamA = beam_area(beamx, beamy) #in arcsec
    return I*beamA/4.25e10
    

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

    
def inverse_rayleigh_jeans(B, freq):
    '''given flux density in Jy sr-1 and freq in GHz, returns temp in K by Rayleigh-Jeans approx'''
    f = freq*1.0e9
    I = B/1e26
    h = 6.626e-34 #SI
    c = 2.9979e8 #SI
    kb = 1.3806e-23 #SI
    tb = (c**2/(2*kb*f**2)) * I
    return tb  

    
def rayleigh_jeans(tb, freq):
    '''given temp in K and freq in GHz, returns flux density in Jy sr-1 by Rayleigh-Jeans approx'''
    f = freq*1.0e9
    h = 6.626e-34 #SI
    c = 2.9979e8 #SI
    kb = 1.3806e-23 #SI  
    I = tb / (c**2/(2*kb*f**2))
    return I*1e26
    
    
def solar_spectrum():
    '''
    Load and return Gueymard solar standard spectrum
        from 0 to 1000 um
    Returns
    -------
    wl: array of astropy Quantities, wavelength in microns
    flux: array of astropy Quantities, solar flux in erg s-1 cm-2 um-1
    '''
    infile = importlib.resources.open_binary('pylanetary.utils.data', 'newguey2003.txt')
    wl, flux = np.loadtxt(infile, skiprows=8).T
    wl=wl*1e-3*u.micron
    flux = flux*u.Watt*u.m**(-2)*u.nm**(-1)
    flux = flux.to(u.erg*u.second**(-1)*u.cm**(-2)*u.micron**(-1))
    
    return wl, flux
    
    
def I_over_F(observed_flux, bp, target_sun_dist, target_omega):
    '''
    see Hammel et al 1989, DOI:10.1016/0019-1035(89)90149-8
    
    Parameters
    ----------
    observed_flux: float, required. flux from target. units erg s-1 cm-2 um-1
    bp: np.array([wls, trans]). the filter transmission function. units of wls is um
    target_sun_dist: float, required. distance between sun and target in AU
    target_omega: float, required. solid angle of target in sr
    
    Returns
    -------
    wl_eff: effective filter wavelength in um
    I/F: the I/F
    
    To do
    -----
    handling of astropy units
    '''
    wl_sun, flux_sun = solar_spectrum()
    
    # observe sun through the filter bandpass
    wl_filt, trans = bp[0], bp[1]
    trans = trans/np.nanmax(trans)
    wl_eff = np.average(wl_filt, weights = trans)
    interp_f = interp1d(wl_filt, trans, bounds_error = False, fill_value = 0.0)
    trans_sun = interp_f(wl_sun)
    sun_flux_earth = np.sum(flux_sun * trans_sun)/np.nansum(trans_sun)
    
    # compute I/F
    sun_flux = (sun_flux_earth)*(1.0/target_sun_dist)**2 
    expected_flux = sun_flux.value * target_omega
    
    return wl_eff, observed_flux * np.pi / expected_flux #factor of pi for pi*Fsun in I/F definition
    
    
def rebin(arr, z):
    '''
    assumes z < 1
    use this instead of ndimage.zoom for z < 1
    '''
    new_shape = (int(arr.shape[0]*z), int(arr.shape[1]*z))
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return (1/z**2)*arr.reshape(shape).mean(-1).mean(1)
    
    
def convolve_with_beam(data, beam):
    '''
    Parameters
    ----------
    data : 2-D numpy array, required.
    beam : float/int, 3-element array-like, or array representing psf required.
        if float/int, circular beam assumed, and this sets the fwhm in pixels
        if 3-element array-like, those are (fwhm_x, fwhm_y, theta_deg) 
        in units (pixels, pixels, degrees)
    '''
    
    if np.array(beam).size == 1:
        fwhm_x = beam
        fwhm_y = beam
        theta = 0.0
        psf = convolution.Gaussian2DKernel(fwhm_x / 2.35482004503,
                                            fwhm_y / 2.35482004503,
                                            Angle(theta, unit=u.deg))
    elif np.array(beam).size == 3:
        (fwhm_x, fwhm_y, theta) = beam
        psf = convolution.Gaussian2DKernel(fwhm_x / 2.35482004503,
                                            fwhm_y / 2.35482004503,
                                            Angle(theta, unit=u.deg))
    else:
        psf = beam
    return convolution.convolve_fft(data, psf)
    