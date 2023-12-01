import pytest
from pytest import fixture
import importlib
from ... import navigation
import numpy as np
from astropy import table
from distutils import dir_util
import os, shutil

'''
to do:
- test what happens with different lat-lon coordinate reference systems
- make tests with other/different ephemerides
'''
    

@fixture
def datadir(request,tmpdir):
    rootdir = request.config.rootdir
    path = os.path.join(rootdir, 'pylanetary', 'navigation', 'tests', 'data')
    return path


def test_datafiles_working(datadir):
    
    assert os.path.isfile(os.path.join(datadir, 'lat_g.npy'))
    assert os.path.isfile(os.path.join(datadir, 'lon_w.npy'))


def test_lat_lon(datadir):
    
    shape = (119,100)
    ob_lon = 37.1 #degrees
    ob_lat = -65 #degrees
    pixscale_km = 600
    np_ang = 285 #degrees
    req = 25560 #km
    rpol = 24970 #km
    
    lat_g, lat_c, lon_w, _, _ = navigation.lat_lon(shape, pixscale_km, ob_lon, ob_lat, np_ang, req, rpol)
    
    lat_g_expected = np.load(os.path.join(datadir, 'lat_g.npy'))
    assert np.allclose(lat_g, lat_g_expected, rtol = 1e-5, equal_nan=True)
    
    lon_w_expected = np.load(os.path.join(datadir, 'lon_w.npy'))      
    assert np.allclose(lon_w, lon_w_expected, rtol = 1e-5, equal_nan=True)
    
    
def test_surface_normal(datadir):
    
    ob_lon = 37.1 #degrees
    lat_g = np.load(os.path.join(datadir, 'lat_g.npy'))
    lon_w = np.load(os.path.join(datadir, 'lon_w.npy'))
    surf_n = navigation.surface_normal(lat_g, lon_w, ob_lon)

    surf_n_expected = np.load(os.path.join(datadir, 'surf_n.npy'))  
    assert np.allclose(surf_n, surf_n_expected, rtol = 1e-5, equal_nan=True)
    
    
#def test_sun_normal():
#    
#    assert whatever
    
    
def test_emission_angle(datadir):
    
    ob_lat = -65 #degrees
    surf_n= np.load(os.path.join(datadir, 'surf_n.npy'))
    mu = navigation.emission_angle(ob_lat, surf_n)
    
    mu_expected = np.load(os.path.join(datadir, 'mu.npy')) 
    assert np.allclose(mu, mu_expected, rtol=1e-5, equal_nan=True)
    
    
def test_ld():
    
    mu = 0.7
    a = 0.3
    ld_linear = navigation.limb_darkening(mu, a, law='linear')
    assert np.isclose(ld_linear, np.array([0.91]), rtol=1e-5)
    
    ld_exp = navigation.limb_darkening(mu, a, law='exp')
    assert np.isclose(ld_exp, np.array([0.8985234417906397]), rtol=1e-5)
    
    ld_quadratic = navigation.limb_darkening(mu, [a, a], law='quadratic')
    assert np.isclose(ld_quadratic, np.array([0.883]), rtol=1e-3)
        
    mu0 = 0.95
    ld_minnaert = navigation.limb_darkening(mu, a, law='minnaert', mu0=mu0)
    assert np.isclose(ld_minnaert, np.array([1.2640040153756316]), rtol=1e-5)
    
    
def test_model_planet_ellipsoid(datadir):
    
    shape = (119,100)
    ob_lon = 37.1 #degrees
    ob_lat = -65 #degrees
    pixscale_km = 600
    np_ang = 285 #degrees
    req = 25560 #km
    rpol = 24970 #km
    
    ell = navigation.ModelEllipsoid( 
                ob_lon, ob_lat, 
                pixscale_km, 
                np_ang, 
                req, rpol, shape=shape)  
    
    lat_g = np.load(os.path.join(datadir, 'lat_g.npy'))
    lon_w = np.load(os.path.join(datadir, 'lon_w.npy'))
    mu = np.load(os.path.join(datadir, 'mu.npy'))
    
    assert np.allclose(ell.lat_g, lat_g, rtol=1e-5, equal_nan=True)
    assert np.allclose(ell.lon_w, lon_w, rtol=1e-5, equal_nan=True)
    assert np.allclose(ell.mu, mu, rtol=1e-5, equal_nan=True)
    