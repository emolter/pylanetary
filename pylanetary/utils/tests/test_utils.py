
import pytest
from pytest import fixture
import os
from ... import utils
import numpy as np
import astropy.units as u
from astropy.io import fits

@fixture
def datadir(request,tmpdir):
    rootdir = request.config.rootdir
    path = os.path.join(rootdir, 'pylanetary', 'navigation', 'tests', 'data')
    return path

@fixture
def expecteddir(request,tmpdir):
    rootdir = request.config.rootdir
    path = os.path.join(rootdir, 'pylanetary', 'utils', 'tests', 'data')
    return path

def test_jybm_to_jysr_and_back():
    jybm = 99.
    beamx = 1
    beamy = 2
    jysr = utils.jybm_to_jysr(jybm, beamx, beamy)
    assert np.isclose(utils.jysr_to_jybm(jysr, beamx, beamy), jybm, rtol=1e-5)

def test_jybm_to_tb_and_back():
    I = 100.
    freq = 300.
    beamx = 0.5
    beamy = 0.3
    tb = utils.jybm_to_tb(I, freq, beamx, beamy)
    assert np.isclose(utils.tb_to_jybm(tb, freq, beamx, beamy), I, rtol=1e-5)

def test_planck_and_back():
    tb = 100
    freq = 301.
    assert np.isclose(utils.inverse_planck(utils.planck(tb, freq), freq), tb, rtol=1e-5)

def test_rj_and_back():
    tb = 100
    freq = 301.
    B = utils.rayleigh_jeans(tb, freq)
    assert np.isclose(utils.inverse_rayleigh_jeans(B, freq), tb, rtol=1e-5)
    
def test_rayleigh_criterion():
    # Test case 1
    wavelength = 5e-6
    diameter = 1.0
    expected_result = 3600*np.rad2deg(1.22 * (wavelength / diameter))
    assert np.isclose(utils.rayleigh_criterion(wavelength, diameter), expected_result, rtol=1e-5)

    # Test case 2
    wavelength = 700e-9
    diameter = 0.5
    expected_result = 3600*np.rad2deg(1.22 * (wavelength / diameter))
    assert np.isclose(utils.rayleigh_criterion(wavelength, diameter), expected_result, rtol=1e-5)

def test_I_over_F():
    # also tests loading solar spectrum
    flux = 6e-16
    wl = np.linspace(1, 2, 500)
    trans = np.ones(500)
    trans[wl < 1.5] = 0
    trans[wl > 1.7] = 0
    bp = np.array([wl, trans])
    target_sun_dist = 20
    target_omega = 1e-19
    (eff_wl, result) = utils.I_over_F(flux, bp, target_sun_dist, target_omega)

    expected_eff_wl = 1.6
    expected_result = 30.601950279 # approximate Mab integrated I/F from Molter+23
    assert np.isclose(eff_wl, expected_eff_wl, rtol=1e-3)
    assert np.isclose(result, expected_result, rtol=1e-5)

def test_rebin():
    
    x = np.arange(10)
    y = np.arange(10)
    xx, yy = np.meshgrid(x, y)
    z = xx + yy
    expected_result = np.array([[100, 225], [225, 350]])
    assert np.allclose(utils.rebin(z, 1/5), expected_result, rtol=1e-8)
    assert np.isclose(np.sum(utils.rebin(z, 1/5)), np.sum(z))
    
def test_convolve_with_beam(datadir, expecteddir):
    
    data = fits.open(os.path.join(datadir, 'nepk99_IF.fits'))[0].data
    
    # Test case 1: no beam
    assert np.allclose(utils.convolve_with_beam(data, None), data, rtol = 1e-5)
    assert np.allclose(utils.convolve_with_beam(data, 0), data, rtol = 1e-5)
    
    # Test case 2: Airy
    beam = 20.0
    result = utils.convolve_with_beam(data, beam, 'airy')
    expected_result = np.load(os.path.join(expecteddir, 'nepk99_IF_airy.npy'))
    assert np.allclose(result, expected_result, rtol = 1e-5)
    
    # Test case 3: circular Gaussian
    beam = 20.0
    result = utils.convolve_with_beam(data, beam, 'gaussian')
    expected_result = np.load(os.path.join(expecteddir, 'nepk99_IF_gauss.npy'))
    assert np.allclose(result, expected_result, rtol = 1e-5)
    
    # Test case 4: ellipsoidal Gaussian
    beam = (20.0, 20.0, 30.0)
    result = utils.convolve_with_beam(data, beam, 'gaussian')
    expected_result = np.load(os.path.join(expecteddir, 'nepk99_IF_gauss.npy'))
    assert np.allclose(result, expected_result, rtol = 1e-5)
    
    # Test case 5: custom 2-D beam
    beam = np.ones((20,20))
    result = utils.convolve_with_beam(data, beam)
    expected_result = np.load(os.path.join(expecteddir, 'nepk99_IF_custom.npy'))
    assert np.allclose(result, expected_result, rtol = 1e-5)
    
    # Test raises
    with pytest.raises(ValueError) as e:
        utils.convolve_with_beam(data, 1.0, 'badinput')
    with pytest.raises(ValueError) as e:
        utils.convolve_with_beam(data, (1.0, 1.0, 30.), 'airy')
    
def test_fourier_deconvolve(datadir, expecteddir):
    
    data = np.load(os.path.join(expecteddir, 'nepk99_IF_custom.npy'))
    data = data[:, 200:669] #must be square
    psf = np.ones((21,21))
    psf_desired = np.pad(np.ones((5,5)), 8, mode='constant') #must have same size as psf
    result = utils.fourier_deconvolve(data, psf, psf_desired, gamma=3e-4)
    expected_result = np.load(os.path.join(expecteddir, 'nepk99_IF_deconvolved.npy'))
    
    assert np.allclose(result, expected_result, rtol = 1e-5)