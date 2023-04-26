# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
from astropy import convolution
from astropy.coordinates import Angle
import astropy.units as u
from astropy.units import Quantity
import importlib, yaml, importlib.resources
from scipy.interpolate import interp1d
import pylanetary.utils.data as body_info
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from datetime import datetime, timedelta

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


class Body:
    """
    Class will instantiate object with attributes from yaml file of matching
    string name.

    Example usage:
    from pylanetary.utils import Body
    jup = Body('Jupiter')

    jup.app_dia_max, jup.app_dia_min
    output> (<Quantity 50.1 arcsec>, <Quantity 30.5 arcsec>)

    print('Jupiter is currently', jup.distance, ' AU from the Earth with apparent size ', jup.app_dia, ' today = ',
    jup.epoch_datetime)
    output> Jupiter is currently 5.93796102228318 AU  AU from the Earth with apparent size  33.2009 arcsec
    today = 2023-04-26 16:32:54
    """
    def __init__(self, name, epoch=None):
        """
        Input string instantiates object using file name of string to
        populate attributes. Astropy quantity objects utilized for data
        with astropy units.

        Parameters
        ----------
        name: str
            Body name string, example "Jupiter" will load Jupiter.yaml
        epoch: astropy.time.Time or None
            The epoch at which to retrieve the ephemeris of the body. If None,
            the current time will be used with a delta time of 1 minute.
        """
        self.name = name
        with importlib.resources.open_binary(body_info, f"{self.name}.yaml") as file:
            yaml_bytes = file.read()
            data = yaml.safe_load(yaml_bytes)

            # basic information and rewrite name
            self.name = data['body']['name']
            self.jpl_hor_id = data['body']['jpl_hor_id']
            self.mass = Quantity(data['body']['mass'], unit=u.kg)
            self.r_eq = Quantity(data['body']['r_eq'], unit=u.km)
            self.r_pol = Quantity(data['body']['r_pol'], unit=u.km)
            self.r_avg = (self.r_eq + self.r_pol) / 2
            self.r_vol = Quantity(data['body']['r_vol'], unit=u.km)
            self.accel_g = Quantity(data['body']['accel_g'], unit=u.m / u.s**2)
            self.num_moons = data['body']['num_moons']

            # orbital information
            self.semi_major_axis = Quantity(data['orbit']['semi_major_axis'], unit=u.au)
            self.t_rot_hrs = Quantity(data['orbit']['t_rot'], unit=u.hr)

            # static observational information
            self.app_dia_max = Quantity(data['observational']['app_dia_max'], unit=u.arcsec)
            self.app_dia_min = Quantity(data['observational']['app_dia_min'], unit=u.arcsec)

            # use datetime Time.now() or epoch to astroquery for ephemeris
            if epoch is None:
                start_time = Time.now()
            else:
                start_time = Time(epoch)
            end_time = start_time + timedelta(minutes=1)
            epochs = {'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                      'stop': end_time.strftime('%Y-%m-%d %H:%M:%S'), 'step': '1m'}
            self.epoch_datetime = start_time.strftime('%Y-%m-%d %H:%M:%S')
            obj = Horizons(id=self.jpl_hor_id, location='500', epochs=epochs)

            self.eph = obj.ephemerides()
            # see self.eph.columns for all columns available in astropy table
            # dynamic observational information
            self.ra = self.eph['RA'][0] * u.deg
            self.dec = self.eph['DEC'][0] * u.deg
            self.distance = self.eph['delta'][0] * u.au
            self.app_dia = self.eph['ang_width'][0] * u.arcsec
