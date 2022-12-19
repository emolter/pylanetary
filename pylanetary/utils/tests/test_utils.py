
import pytest
from ... import utils
import numpy as np

def test_jybm_to_jysr_and_back():
    
    assert x


def test_jybm_to_tb_and_back():
    
    I = 100.
    freq = 300.
    beamx = 0.5
    beamy = 0.3
    tb = jybm_to_tb(I, freq, beamx, beamy)
    assert np.isclose(tb_to_jybm(tb, freq, beamx, beamy), I, rtol=1e-5)
    

def test_planck_and_back():
    
    tb = 100
    freq = 301.
    assert np.isclose(inverse_planck(planck(tb, freq), freq), tb, rtol=1e-5)
    

def test_rj_and_back():
    
    tb = 100
    freq = 301.
    assert np.isclose(inverse_rj(rj(tb, freq), freq), tb, rtol=1e-5)
    
    
def test_convolve_with_beam():
    
    assert
    





