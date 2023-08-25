import pytest
from pytest import fixture
import importlib
from ... import navigation
import numpy as np
from distutils import dir_util
import os, shutil
from ...utils import Body
from astropy.io import fits


@fixture
def datadir(request,tmpdir):
    rootdir = request.config.rootdir
    path = os.path.join(rootdir, 'pylanetary', 'navigation', 'tests', 'data')
    return path


def test_navigation(datadir):
    
    # load data, ephem, lat_g, lon_w, mu
    obs_time = '2019-10-28 08:50:50'
    obs_code=568 #Keck Horizons observatory code
    pixscale_arcsec = 0.009971 #arcsec, keck
    flux=2000
    a=0.1
    beam=0.5
    keck_uranus = np.load(os.path.join(datadir, 'keck_uranus.npy'))
    
    ura = Body('Uranus', epoch=obs_time, location=obs_code)
    nav = navigation.Nav(keck_uranus, ura, pixscale_arcsec)
       
    ldmodel_expected = np.load(os.path.join(datadir, 'ldmodel.npy'))
    ldmodel = nav.ldmodel(flux, a, beam = beam, law='exp')
    
    assert np.allclose(ldmodel, ldmodel_expected, rtol=1e-5, equal_nan=True)
    
    # test co-location algorithms
    (dx, dy, dxerr, dyerr) = nav.colocate(
                        tb=flux, 
                        a=a, 
                        mode='convolution', 
                        diagnostic_plot=False,
                        beam=beam)
    assert np.allclose([dx, dy], [137.794921875, -10.744140625], atol = 1.0) #absolute 1-pixel tolerance
    (dx_canny, dy_canny, dxerr_canny, dyerr_canny) = nav.colocate(mode='canny', diagnostic_plot=False, tb=flux, a=a, low_thresh=1e-5, high_thresh=0.01, sigma=5)
    assert np.allclose([dx_canny, dy_canny], [130.048828125, -9.201171875], atol = 1.0) #absolute 1-pixel tolerance
    
    # test shifting of model to same loc as data
    nav.xy_shift_model(dx, dy)
    shifted_lat_expected = np.load(os.path.join(datadir, 'lat_g_keck.npy'))
    assert np.allclose(nav.lat_g, shifted_lat_expected, rtol=1e-3, equal_nan=True)
    
    # test re-projection onto lat-lon grid
    projected, mu_projected = nav.reproject()
    projected_expected = np.load(os.path.join(datadir, 'projected.npy'))
    mu_projected_expected = np.load(os.path.join(datadir, 'mu_projected.npy'))
    assert np.allclose(projected, projected_expected, rtol = 1e-2, equal_nan=True)
    assert np.allclose(mu_projected, mu_projected_expected, rtol = 1e-2, equal_nan=True)
    
    
def test_nav_nonsquare(datadir):
    '''
    test for issue where non-square and/or odd-sided
    nav.colocate fails
    also represents a Neptune test case
    '''
    obs_code=568 #Keck Horizons observatory code
    pixscale_arcsec = 0.009971 #arcsec, keck
    hdul = fits.open(os.path.join(datadir, 'nepk99_IF.fits'))
    obs_time = hdul[0].header['DATE-OBS'] + ' ' + hdul[0].header['EXPSTART'][:-4]
    
    nep = Body('Neptune', epoch=obs_time, location=obs_code)
    nep.ephem['NPole_ang'] = 0.0
    nav = navigation.Nav(hdul[0].data, nep, pixscale_arcsec)
    (dx, dy, dxerr, dyerr) = nav.colocate(
                        tb=1.5e-4, 
                        a=0.01, 
                        mode='disk', 
                        diagnostic_plot=False,
                        beam=0.5)
    
    assert dx == -1.5
    assert dy == 6.5
    
    
    
    
    