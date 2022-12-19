import pytest
import importlib
from ... import planetnav
import numpy as np
from astropy import table

'''
to do:
- refactor lat-lon to be more easily chunked into small tests
- test what happens with different lat-lon coordinate reference systems
- make tests with other/different ephemerides
'''

def test_lat_lon():
    
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
    
    lat_g_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lat_g.npy')
    lon_w_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lon_w.npy')
        
    assert np.allclose(lat_g, np.load(lat_g_data_source), rtol = 1e-5)
    assert np.allclose(lon_w, np.load(lon_w_data_source), rtol = 1e-5)
    assert np.allclose(mu, np.load(mu_data_source), rtol = 1e-5)
    
    
def test_surface_normal():
    
    ob_lon = 37.1 #degrees
    lat_g_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lat_g.npy')
    lon_w_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lon_w.npy')
    lat_g = np.load(lat_g_data_source)
    lon_w = np.load(lon_w_data_source)
    suf_n = planetnav.surface_normal(lat_g, lon_w, ob_lon)
    surf_n_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'surf_n.npy')
            
    assert np.allclose(surf_n, np.load(surf_n_data_source), rtol = 1e-5)
    
    
def test_sun_normal():
    
    assert whatever
    
    
def test_emission_angle():
    
    ob_lat = -65 #degrees
    surf_n_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'surf_n.npy')
    surf_n = np.load(surf_n_data_source)
    mu = planetnav.emission_angle(ob_lat, surf_n)
    mu_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'mu.npy')
        
    assert np.allclose(mu, np.load(mu_data_source), rtol=1e-5)
    
    
def test_ld():
    
    mu = 0.7
    a = 0.3
    ld_linear = planetnav.limb_darkening(mu, a, law='linear')
    assert np.isclose(ld_linear, np.array([0.91]), rtol=1e-5)
    
    ld_exp = limb_darkening(mu, a, law='exp')
    assert np.isclose(ld_exp, np.array([0.8985234417906397]), rtol=1e-5)
    
    with pytest.raises(NotImplementedError):
        ld_quadratic = limb_darkening(mu, a, law='quadratic')
        
    mu0 = 0.95
    ld_minnaert = planetnav.limb_darkening(mu, a, law='minnaert', mu0=mu0)
    assert np.isclose(ld_minnaert, np.array([1.2640040153756316]), rtol=1e-5)
    
    
def test_model_planet_ellipsoid():
    
    shape = (119,100)
    ob_lon = 37.1 #degrees
    ob_lat = -65 #degrees
    pixscale_km = 600
    np_ang = 285 #degrees
    req = 25560 #km
    rpol = 24970 #km
    
    ell = planetnav.ModelPlanetEllipsoid(
                shape, 
                ob_lon, ob_lat, 
                pixscale_km, 
                np_ang, 
                req, rpol)    
    
    lat_g_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lat_g.npy')
    lon_w_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lon_w.npy')
    mu_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'mu.npy')
    lat_g = np.load(lat_g_data_source)
    lon_w = np.load(lon_w_data_source)
    mu = np.load(mu_data_source)
    
    assert np.isclose(ell.lat_g, lat_g, rtol=1e-5)
    assert np.isclose(ell.lon_w, lon_w, rtol=1e-5)
    assert np.isclose(ell.mu, mu, rtol=1e-5)
    
    with pytest.raises(NotImplementedError):
        ell.write('whatever')
    

def test_planetnav():
    
    # load data, ephem, lat_g, lon_w, mu
    ephem_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'ephem.hdf5')
    ephem = table.Table.read(ephem_data_source, format='hdf5')
    req = 25560 #km
    rpol = 24970 #km
    pixscale_arcsec = 0.009942 #arcsec, keck
    nav = planetnav.PlanetNav(data, ephem, req, rpol, pixscale_arcsec)
       
    ldmodel_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'ldmodel.npy')
    ldmodel = nav.ldmodel(2000, 0.1, beamsize = 0.05, law='exp')
    assert np.allclose(ldmodel, np.load(ldmodel_data_source))
    
    # test co-location algorithms
    (dx, dy, dxerr, dyerr) = nav.colocate(mode='convolution', diagnostic_plot=False)
    assert np.allclose([dx, dy], [137.794921875, -10.744140625], tol = 1.0) #absolute 1-pixel tolerance
    (dx_canny, dy_canny, dxerr_canny, dyerr_canny) = nav.colocate(mode='canny', diagnostic_plot=False)
    assert np.allclose([dx_canny, dy_canny], [137.794921875, -10.744140625], tol = 1.0) #absolute 1-pixel tolerance
    
    # test shifting of model to same loc as data
    nav.xy_shift_model(dx, dy)
    shifted_lat_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'lat_g_keck.npy')
    assert np.allclose(nav.lat_g, np.load(shifted_lat_data_source), rtol=1e-2)
    
    # test re-projection onto lat-lon grid
    projected, mu_projected = nav.reproject()
    projected_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'projected.npy')
    mu_projected_data_source = importlib.resources.open_binary(
        'pylanetary.pylanetary.planetnav.tests.data', 'mu_projected.npy')
    assert np.allclose(projected, np.load(projected_data_source), rtol = 1e-3)
    assert np.allclose(mu_projected, np.load(mu_projected_data_source), rtol = 1e-3)
    
    
    
    