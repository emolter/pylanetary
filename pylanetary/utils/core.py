# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
from datetime import timedelta
import importlib
import yaml
import importlib.resources
from scipy.interpolate import interp1d
from scipy import fftpack

from astropy import convolution
from astropy.coordinates import Angle
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from astroquery.jplhorizons import Horizons

import pylanetary.utils.data as body_info

'''
To do:
* docstrings
* write Jupyter notebook showing off these functions
* test I/F function against literature values for multiple observations!
* write tests for convolve_with_beam, especially for arbitrary PSF image
* make convolve_with_beam accept Astropy PSFs and super-resolution PSFs as the kernel
* make these accept astropy units
'''


def solid_angle(dS, d):
    '''
    Approximate solid angle of small facet
    https://en.wikipedia.org/wiki/Solid_angle

    Parameters
    ----------
    dS : float, required
        [dist^2] flat area of facet
    d : float, required
        [dist] distance to facet

    Returns
    -------
    float
        [sr] solid angle of facet

    Example
    -------
    a = 210 #km, Proteus average radius
    d = 30 * 1.496e8 #distance to Neptune in km
    solid_angle(3.14*a**2, d)
    '''
    return (dS / d**2)


def beam_area(beamx, beamy):
    '''
    Parameters
    ----------
    beamx : float, required
        [arcsec]
    beamy : float, required
        [arcsec]

    Returns
    -------
    float
        area of a Gaussian beam
    '''
    return (np.pi / (4 * np.log(2))) * beamx * beamy


def jybm_to_jysr(I, beamx, beamy):
    '''
    Parameters
    ----------
    I : float, required
        [Jy/beam] flux density
    beamx : float, required
        [arcsec]
    beamy : float, required
        [arcsec]

    Returns
    -------
    float
        [Jy/sr] flux density
    '''
    beamA = beam_area(beamx, beamy)  # in arcsec
    return 4.25e10 * I / beamA


def jysr_to_jybm(I, beamx, beamy):
    '''
    Parameters
    ----------
    I: float
        [Jy/sr] flux density
    beamx: float, required
        [arcsec]
    beamy: float, required
        [arcsec]

    Returns
    -------
    [Jy/beam] flux density
    '''
    beamA = beam_area(beamx, beamy)  # in arcsec
    return I * beamA / 4.25e10


def jybm_to_tb(I, freq, beamx, beamy):
    '''
    Parameters
    ----------
    I : float, required
        [Jy/beam] flux density
    freq : float, required
        [GHz] frequency
    beamx : float, required
        [arcsec]
    beamy : float, required
        [arcsec]

    Returns
    -------
    float
        [K] brightness temperature

    References
    ----------
    science.nrao.edu/facilities/vla/proposing/TBconv

    Notes
    -----
    uses the Rayleigh-Jeans approximation
    '''
    return (1e3 * 1.22e3) * I / (freq**2 * beamx * beamy)


def tb_to_jybm(Tb, freq, beamx, beamy):
    '''
    from science.nrao.edu/facilities/vla/proposing/TBconv
        I in Jy/bm, freq in GHz, returns tb in Kelvin

    Parameters
    ----------
    Tb : float, required
        [K] float or array-like, brightness temperature
    freq : float, required
        [GHz] float or array-like, frequency
    beamx : float, required
        [arcsec] float, FWHM of Gaussian beam in x direction
    beamy : float, required
        [arcsec] float, FWHM of Gaussian beam in y direction
    '''
    return Tb * (freq**2 * beamx * beamy) / (1e3 * 1.22e3)


def planck(tb, freq):
    '''
    Parameters
    ----------
    tb :  float, required
        [K] brightness temperature
    freq :  float, required
        [GHz] frequency

    Returns
    -------
    float
        [Jy/sr] flux density
    '''
    f = freq * 1.0e9  # GHz to Hz
    h = 6.626e-34  # SI
    c = 2.9979e8  # SI
    kb = 1.3806e-23  # SI
    I = (2.0 * h * f**3 / c**2) * (1 / (np.exp(h * f / (kb * tb)) - 1))  # SI
    return I * 1e26  # Jy sr-1


def inverse_planck(B, freq):
    '''
    Parameters
    ----------
    B : float, required
        [Jy/sr] flux density
    freq : float, required
        [GHz] frequency

    Returns
    -------
    float
        [K] brightness temperature
    '''
    f = freq * 1.0e9
    I = B / 1e26
    h = 6.626e-34  # SI
    c = 2.9979e8  # SI
    kb = 1.3806e-23  # SI
    tb = (h * f / kb) * (1 / np.log(2.0 * h * f**3 / (c**2 * I) + 1))
    return tb  # K


def inverse_rayleigh_jeans(B, freq):
    '''
    Rayleigh-Jeans approximation to inverse_planck()

    Parameters
    ----------
    B : float, required
        [Jy/sr] flux density
    freq :  float, required
        [GHz] frequency

    Returns
    -------
    float
        [K] brightness temperature
    '''
    f = freq * 1.0e9
    I = B / 1e26
    h = 6.626e-34  # SI
    c = 2.9979e8  # SI
    kb = 1.3806e-23  # SI
    tb = (c**2 / (2 * kb * f**2)) * I
    return tb


def rayleigh_jeans(tb, freq):
    '''
    Rayleigh-Jeans approximation to planck()

    Parameters
    ----------
    tb : float, required
        [K] brightness temperature
    freq : float, required
        [GHz] frequency

    Returns
    -------
    float
        [Jy/sr] flux density
    '''
    f = freq * 1.0e9
    h = 6.626e-34  # SI
    c = 2.9979e8  # SI
    kb = 1.3806e-23  # SI
    I = tb / (c**2 / (2 * kb * f**2))
    return I * 1e26


def rayleigh_criterion(wl, d):
    '''
    Parameters
    ----------
    wl :  float, required
        [m] wavelength
    d : float, required
        [m] diameter of telescope

    Returns
    -------
    float
        [arcsec] Diffraction limit of a circular aperture
    '''
    return np.rad2deg(1.22 * wl / d) * 3600


def solar_spectrum():
    '''
    Load and return Gueymard solar standard spectrum from 0 to 1000 um

    References
    ~~~~~~~~~~
    https://doi.org/10.1016/j.solener.2003.08.039
    accessed at https://www.nrel.gov/grid/solar-resource/spectra.html

    Returns
    -------
    np.array
        array of astropy Quantities.
        [um] wavelength
    np.array
        array of astropy Quantities.
        solar flux
        [erg s-1 cm-2 um-1]
    '''
    infile = importlib.resources.open_binary(
        'pylanetary.utils.data', 'newguey2003.txt')
    wl, flux = np.loadtxt(infile, skiprows=8).T
    wl = wl * 1e-3 * u.micron
    flux = flux * u.Watt * u.m**(-2) * u.nm**(-1)
    flux = flux.to(u.erg * u.second**(-1) * u.cm**(-2) * u.micron**(-1))

    return wl, flux


def I_over_F(observed_flux, bp, target_sun_dist, target_omega):
    '''
    definition from Hammel et al 1989, DOI:10.1016/0019-1035(89)90149-8

    Parameters
    ----------
    observed_flux : float, required.
        [erg s-1 cm-2 um-1] flux of target
    bp : np.array, required
         ([wls, trans])
        [[um], [-]] the filter transmission function. does not matter if normalized or not.
    target_sun_dist : float, required.
        [AU] distance between sun and target
    target_omega : float, required.
        [sr] for unresolved object, solid angle of that object
        for resolved object, solid angle of one pixel

    Returns
    -------
    float
        [um] effective filter wavelength
    float
        I/F

    Notes
    -----
    * sun_flux_earth agrees with Arvesen 1969 in H band, doi:10.1364/AO.8.002215
    * needs handling of astropy units
    '''
    wl_sun, flux_sun = solar_spectrum()

    # observe sun through the filter bandpass
    wl_filt, trans = bp[0], bp[1]
    trans = trans / np.nanmax(trans)
    wl_eff = np.average(wl_filt, weights=trans)
    interp_f = interp1d(wl_filt, trans, bounds_error=False, fill_value=0.0)
    trans_sun = interp_f(wl_sun)
    sun_flux_earth = np.sum(flux_sun * trans_sun) / np.nansum(trans_sun)

    # compute I/F. add factor of 1/pi from definition of I/F
    sun_flux = (1 / np.pi) * (sun_flux_earth) * (1.0 / target_sun_dist)**2
    expected_flux = sun_flux.value * target_omega

    return wl_eff, observed_flux / expected_flux


def rebin(arr, z):
    '''
    simple integer binning of numpy array in two dimensions

    Parameters
    ----------
    arr : np.array, required
    z : int, required.
        factor to multiply array size by. typically z<1
        i.e., this is ndimage.zoom() for z<1

    Returns
    -------
    np.array
        2-D numpy array of shape arr.shape * z

    Notes
    -----
    use this instead of ndimage.zoom for z < 1
    '''
    new_shape = (int(arr.shape[0] * z), int(arr.shape[1] * z))
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return (1 / z**2) * arr.reshape(shape).mean(-1).mean(1)


def convolve_with_beam(data, beam, mode='gaussian'):
    '''
    Convolves input 2-D image with a Gaussian beam or an input PSF image

    Parameters
    ----------
    data : np.array, required.
    beam : float/int, 3-element array-like, np.array, or None, required.
        If float/int, this sets the fwhm [pixels] of either Airy disk or
        circular Gaussian beam, depending on "mode" parameter.
        If 3-element array-like, those are (fwhm_x, fwhm_y, theta_deg) for a 2-D Gaussian
        [pixels, pixels, degrees]. In this case, "Airy" mode not supported.
        if np.array of size > 3, assumes input PSF image
        if None or 0.0, simply returns original data
    mode : str, optional, default "gaussian"
        options "gaussian", "airy"; what beam shape to use. Case insensitive.
        "airy" only accepts beam of dtype float
        or 1-element array-like (i.e., beam must be circular).
        this parameter has no effect if beam is a 2-D array.

    Returns
    -------
    np.array
        beam-convolved data
    '''
    # allow pass through for beam of zero size
    if (beam is None):
        return data
    if np.array(beam).size == 1:
        if beam == 0.0:
            return data

    # check inputs
    mode = mode.lower()
    if mode not in ['gaussian', 'airy']:
        raise ValueError(
            f'mode {mode} not recognized; supported options are "gaussian", "airy"')
    if (mode == 'airy') and (np.array(beam).size != 1):
        raise ValueError(
            f'mode "airy" only accepts a single value for the "beam" parameter (distance to first null)')

    if mode == 'airy':
        # convert FWHM to first-null distance
        null = 0.5 * (2.44 / 1.02) * beam
        psf = convolution.AiryDisk2DKernel(radius=null)
    elif (mode == 'gaussian') and (np.array(beam).size == 1):
        fwhm_x = beam
        fwhm_y = beam
        theta = 0.0
        psf = convolution.Gaussian2DKernel(fwhm_x / 2.35482004503,
                                           fwhm_y / 2.35482004503,
                                           Angle(theta, unit=u.deg))
    elif (mode == 'gaussian') and (np.array(beam).size == 3):
        (fwhm_x, fwhm_y, theta) = beam
        psf = convolution.Gaussian2DKernel(fwhm_x / 2.35482004503,
                                           fwhm_y / 2.35482004503,
                                           Angle(theta, unit=u.deg))
    else:
        psf = beam

    return convolution.convolve_fft(data, psf)


def fourier_deconvolve(data, psf, psf_desired, gamma=3e-4):
    '''
    Reproducing `Pat Fry's Fourier deconvolution method <https://doi.org/10.1016/j.icarus.2022.115224>`_.
    see also `Cunningham and Anthony <https://doi.org/10.1006/icar.1993>`_.

    Parameters
    ----------
    data : np.array, required.
        must be square. needs odd shape
    psf : np.array, required.
        must be square. needs odd shape
    psf_desired: np.array, required.
        must have same shape as psf
    gamma: float, optional, default 3e-4
        Tikhonov regularization parameter. prevents zeros in the denominator.

    Returns
    -------
    np.array
        deconvolved data
    '''
    # pad psf to equal data
    w = int((data.shape[0] - psf.shape[0]) / 2)
    psf = np.pad(psf, w)
    psf_desired = np.pad(psf_desired, w)

    I = fftpack.fftshift(fftpack.fftn(data))
    P = fftpack.fftshift(fftpack.fftn(psf))
    G = fftpack.fftshift(fftpack.fftn(psf_desired))
    O = (np.conjugate(P) * I * G) / (np.absolute(P)**2 + gamma)

    #psf_deconvolved = fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(O)))
    psf_deconvolved = fftpack.ifftn(fftpack.ifftshift(O))
    return np.real(psf_deconvolved)


class Body:
    """
    instantiate object with attributes from yaml file of matching
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

    def __init__(self, name, epoch=None, location=None):
        """
        Input string instantiates object using file name of string to
        populate attributes. Astropy quantity objects utilized for data
        with astropy units.

        Parameters
        ----------
        name : str, required.
            Body name string, example "Jupiter" will load Jupiter.yaml
        epoch : astropy.time.Time, optional.
            The epoch at which to retrieve the ephemeris of the body.
            If not set, the current time will be used.
        location : str, optional.
            JPL Horizons observatory code.
            If not set, center of Earth will be used.
        """
        self.name = name.lower().capitalize()
        with importlib.resources.open_binary(body_info, f"{self.name}.yaml") as file:
            yaml_bytes = file.read()
            data = yaml.safe_load(yaml_bytes)

        # basic information and rewrite name
        self.name = data['body']['name']
        self.jpl_hor_id = data['body']['jpl_hor_id']
        self.mass = Quantity(data['body']['mass'], unit=u.kg)
        self.req = Quantity(data['body']['req'], unit=u.km)
        self.rpol = Quantity(data['body']['rpol'], unit=u.km)
        self.ravg = (self.req + self.rpol) / 2
        self.rvol = Quantity(data['body']['rvol'], unit=u.km)
        self.accel_g = Quantity(data['body']['accel_g'], unit=u.m / u.s**2)
        self.longitude_convention = data['body']['longitude_convention']

        # orbital information
        self.semi_major_axis = Quantity(
            data['orbit']['semi_major_axis'],
            unit=u.km).to(
            u.AU)
        self.t_rot = Quantity(data['orbit']['t_rot'], unit=u.hr)

        # static observational information
        self.app_dia_max = Quantity(
            data['observational']['app_dia_max'],
            unit=u.arcsec)
        self.app_dia_min = Quantity(
            data['observational']['app_dia_min'],
            unit=u.arcsec)

        # astroquery Horizons ephemeris
        if location is None:
            location = '500'
        if epoch is None:
            start_time = Time.now()
        else:
            start_time = Time(epoch)
        self.location = location
        self.epoch = epoch

        end_time = start_time + timedelta(minutes=1)
        epochs = {'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                  'stop': end_time.strftime('%Y-%m-%d %H:%M:%S'), 'step': '1m'}
        self.epoch_datetime = start_time.strftime('%Y-%m-%d %H:%M:%S')

        obj = Horizons(id=self.jpl_hor_id, location=location, epochs=epochs)

        self.ephem = obj.ephemerides()[0]

    def __str__(self):
        return f'pylantary.utils.Body instance; {self.name}, Horizons ID {self.jpl_hor_id}'
