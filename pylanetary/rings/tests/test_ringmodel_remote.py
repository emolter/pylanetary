import pytest
from astropy.time import Time
import astropy.units as u
import numpy as np
import os

from ...rings import *
from ...utils import Body


def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_dir, filename)
    

def test_uranus():
    
    imsize = 300 #px
    pixscale = 500 #km/px
    alma_coords = ( -67.755 * u.deg, -23.029 * u.deg, 5000 * u.m) #lon, lat, alt(m)
    epoch = '2020-01-30 00:00'
    
    epoch_astropy = Time(epoch, format = 'iso', scale = 'utc')
    ura = Body('uranus', epoch=epoch_astropy, location='399')
    uranus_rings = RingSystemModelObservation(ura, alma_coords, ringnames = None)
    arr = uranus_rings.as_2d_array((imsize, imsize), pixscale, beam = (5,4,30)) 
    
    arr_expected = np.load(data_path('ura_system_testarr.npy'))
    
    assert np.allclose(arr, arr_expected, rtol=1e-3)