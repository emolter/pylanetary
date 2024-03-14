import pytest
import os
import numpy as np
from astroquery.solarsystem import pds
import astropy.units as u
from ...rings import *


# files in data/ for different planets
DATA_FILES = {'Uranus': 'uranus_ephemeris.html',
              'Neptune': 'neptune_ephemeris.html',
              'Saturn': 'saturn_ephemeris.html',
              }


def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_dir, filename)


# monkeypatch replacement request function
def nonremote_request(self, request_type, url, **kwargs):

    planet_name = kwargs['params']['center_body']
    with open(data_path(DATA_FILES[planet_name.capitalize()]), "rb") as f:
        response = MockResponse(content=f.read(), url=url)

    return response


# use a pytest fixture to create a dummy 'requests.get' function,
# that mocks(monkeypatches) the actual 'requests.get' function:
@pytest.fixture
def patch_request(request):
    mp = request.getfixturevalue("monkeypatch")

    mp.setattr(pds.core.RingNodeClass, "_request", nonremote_request)
    return mp


# --------------------------------- actual test functions

def test_ring_as_array():
    '''
    this also implicitly tests as_elliptical_annulus()
    but should probably write an explicit test eventually
    '''

    a = 51149  # km
    e = 0.4
    i = 80.0
    omega = 60.0
    w = 30.
    imsize = 300  # px
    pixscale = 500  # km/px
    ringmodel2 = Ring(a, 0.4, w, i, omega, flux=0.001, width=10000 * u.km)
    arr = ringmodel2.as_2d_array((imsize, imsize), pixscale, beam=(10, 6, 30))
    arr_expected = np.load(data_path('as_2d_array_testarr.npy'))
    assert np.allclose(arr, arr_expected, rtol=1e-3)


def test_change_params_preserve_flux():
    '''
    discovered a bug in testing that if e is changed dynamically like this,
    then b will not be re-computed, and this ruins the elliptical annulus
    solved by removing self.b and self.c inside __init__
    '''
    a = 51149  # km
    e = 0.4
    i = 80.0
    omega = 60.0
    w = 30.
    imsize = 300  # px
    pixscale = 500  # km/px
    ringmodel = Ring(a, 0.4, w, i, omega, flux=0.001, width=10000 * u.km)
    arr = ringmodel.as_2d_array((imsize, imsize), pixscale, beam=(10, 6, 30))

    ringmodel2 = Ring(a, 0.01, w, i, omega, flux=0.001, width=10000 * u.km)
    ringmodel2.e = 0.4
    arr2 = ringmodel2.as_2d_array((imsize, imsize), pixscale, beam=(10, 6, 30))

    assert np.allclose(arr, arr2, rtol=1e-3)


def test_ring_as_wedges():

    a = 51149  # km
    e = 0.4
    i = 80.0
    omega = 60.0
    w = 30.
    imsize = 300  # px
    pixscale = 500  # km/px
    width = 10000 * u.km
    nwedges = 12
    ringmodel = Ring(a, e, omega, i, w)

    thetas, wedges = ringmodel.as_azimuthal_wedges(
        [imsize, imsize],
        pixscale,
        nwedges=nwedges,
        z=1,
        width=width)
    wedges = np.array(wedges)
    wedges_expected = np.load(data_path('wedges_testarr.npy'))

    assert np.allclose(thetas, np.linspace(
        0, 2 * np.pi, nwedges + 1)[:-1], rtol=1e-3)
    # rounding to deal with machine precision
    assert np.round(np.max(np.sum(wedges, axis=0)), 6) == 1.0
    assert np.allclose(wedges, wedges_expected, rtol=1e-3)
