import pytest
from astroquery.solarsystem import pds

from ... import rings


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

def test_ring_as_annulus():
    
    model_ring = rings.Ring(a, e, omega, i, w)
    ann = model_ring.as_elliptical_annulus(shape, pixscale, width, center = None)
    
    # actually use annulus on some data?
    
    assert whatever
    
    
def test_ring_as_wedges():
    
    model_ring = rings.Ring(a, e, omega, i, w)
    ann_list = model_ring.as_keplers3rd_wedges(width, n)
    
    assert whatever
    
    
def test_ring_as_orbit():
    
    model_ring = rings.Ring(a, e, omega, i, w)
    ke = model_ring.as_orbit(T, tau)
    
    assert whatever
    
    
def test_ring_as_array():
    
    model_ring = rings.Ring(a, e, omega, i, w)
    arr = model_ring.as_2d_array(shape, pixscale, width=None, flux=None, beamsize=None)
    
    assert whatever
    
    
def test_model_system_Uranus(patch_request):
    
    assert whatever
    
    
def test_model_system_Saturn(patch_request):
    
    assert whatever
    
    
    
    
    
    
    
    