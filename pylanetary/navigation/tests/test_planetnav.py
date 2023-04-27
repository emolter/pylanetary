import pytest
from pytest import fixture
import importlib
from ... import planetnav
import numpy as np
from astropy import table
from distutils import dir_util
import os, shutil

'''
to do:
- refactor lat-lon to be more easily chunked into small tests
- test what happens with different lat-lon coordinate reference systems
- make tests with other/different ephemerides
'''
    

@fixture
def datadir(request,tmpdir):
    rootdir = request.config.rootdir
    path = os.path.join(rootdir, 'pylanetary', 'planetnav', 'tests', 'data')
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
    
    xcen, ycen = int(shape[0]/2), int(shape[1]/2) #pixels at center of planet
    xx = np.arange(shape[0]) - xcen
    yy = np.arange(shape[1]) - ycen
    x,y = np.meshgrid(xx,yy)
    lat_g, lat_c, lon_w = planetnav.lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol)
    
    lat_g_expected = np.load(os.path.join(datadir, 'lat_g.npy'))
    assert np.allclose(lat_g, lat_g_expected, rtol = 1e-5, equal_nan=True)
    
    lon_w_expected = np.load(os.path.join(datadir, 'lon_w.npy'))      
    assert np.allclose(lon_w, lon_w_expected, rtol = 1e-5, equal_nan=True)
    
    
def test_surface_normal(datadir):
    
    ob_lon = 37.1 #degrees
    lat_g = np.load(os.path.join(datadir, 'lat_g.npy'))
    lon_w = np.load(os.path.join(datadir, 'lon_w.npy'))
    surf_n = planetnav.surface_normal(lat_g, lon_w, ob_lon)
    surf_n_expected = np.load(os.path.join(datadir, 'surf_n.npy'))
            
    assert np.allclose(surf_n, surf_n_expected, rtol = 1e-5, equal_nan=True)
    
    
#def test_sun_normal():
#    
#    assert whatever
    
    
def test_emission_angle(datadir):
    
    ob_lat = -65 #degrees
    surf_n= np.load(os.path.join(datadir, 'surf_n.npy'))
    mu = planetnav.emission_angle(ob_lat, surf_n)
    mu_expected = np.load(os.path.join(datadir, 'mu.npy'))   
    assert np.allclose(mu, mu_expected, rtol=1e-5, equal_nan=True)
    
    
def test_ld():
    
    mu = 0.7
    a = 0.3
    ld_linear = planetnav.limb_darkening(mu, a, law='linear')
    assert np.isclose(ld_linear, np.array([0.91]), rtol=1e-5)
    
    ld_exp = planetnav.limb_darkening(mu, a, law='exp')
    assert np.isclose(ld_exp, np.array([0.8985234417906397]), rtol=1e-5)
    
    with pytest.raises(NotImplementedError):
        ld_quadratic = planetnav.limb_darkening(mu, a, law='quadratic')
        
    mu0 = 0.95
    ld_minnaert = planetnav.limb_darkening(mu, a, law='minnaert', mu0=mu0)
    assert np.isclose(ld_minnaert, np.array([1.2640040153756316]), rtol=1e-5)
    
    
def test_model_planet_ellipsoid(datadir):
    
    shape = (119,100)
    ob_lon = 37.1 #degrees
    ob_lat = -65 #degrees
    pixscale_km = 600
    np_ang = 285 #degrees
    req = 25560 #km
    rpol = 24970 #km
    
    ell = planetnav.ModelEllipsoid(
                shape, 
                ob_lon, ob_lat, 
                pixscale_km, 
                np_ang, 
                req, rpol)    
    
    lat_g = np.load(os.path.join(datadir, 'lat_g.npy'))
    lon_w = np.load(os.path.join(datadir, 'lon_w.npy'))
    mu = np.load(os.path.join(datadir, 'mu.npy'))
    
    assert np.allclose(ell.lat_g, lat_g, rtol=1e-5, equal_nan=True)
    assert np.allclose(ell.lon_w, lon_w, rtol=1e-5, equal_nan=True)
    assert np.allclose(ell.mu, mu, rtol=1e-5, equal_nan=True)
    
    with pytest.raises(NotImplementedError):
        ell.write('whatever')
    

def test_planetnav(datadir):
    
    # load data, ephem, lat_g, lon_w, mu
    ephem = table.Table.read(os.path.join(datadir,'ephem.hdf5'), format='hdf5')[0]
    req = 25560 #km
    rpol = 24970 #km
    pixscale_arcsec = 0.009942 #arcsec, keck
    flux=2000
    a=0.1
    beamsize=0.05
    keck_uranus = np.load(os.path.join(datadir, 'keck_uranus.npy'))
    nav = planetnav.Nav(keck_uranus, ephem, req, rpol, pixscale_arcsec)
       
    ldmodel_expected = np.load(os.path.join(datadir, 'ldmodel.npy'))
    ldmodel = nav.ldmodel(flux, a, beamsize = beamsize, law='exp')
    assert np.allclose(ldmodel, ldmodel_expected, rtol=1e-5, equal_nan=True)
    
    # test co-location algorithms
    (dx, dy, dxerr, dyerr) = nav.colocate(
                        tb=flux, 
                        a=a, 
                        mode='convolution', 
                        diagnostic_plot=False,
                        beamsize=beamsize)
    assert np.allclose([dx, dy], [137.794921875, -10.744140625], atol = 1.0) #absolute 1-pixel tolerance
    (dx_canny, dy_canny, dxerr_canny, dyerr_canny) = nav.colocate(mode='canny', diagnostic_plot=False, low_thresh=1e-5, high_thresh=0.01, sigma=5)
    assert np.allclose([dx_canny, dy_canny], [130.048828125, -9.201171875], atol = 1.0) #absolute 1-pixel tolerance
    
    # test shifting of model to same loc as data
    nav.xy_shift_model(dx, dy)
    shifted_lat_expected = np.load(os.path.join(datadir, 'lat_g_keck.npy'))
    assert np.allclose(nav.lat_g, shifted_lat_expected, rtol=1e-2, equal_nan=True)
    
    # test re-projection onto lat-lon grid
    projected, mu_projected = nav.reproject()
    projected_expected = np.load(os.path.join(datadir, 'projected.npy'))
    mu_projected_expected = np.load(os.path.join(datadir, 'mu_projected.npy'))
    assert np.allclose(projected, projected_expected, rtol = 1e-2, equal_nan=True)
    assert np.allclose(mu_projected, mu_projected_expected, rtol = 1e-2, equal_nan=True)
    
    
    
    