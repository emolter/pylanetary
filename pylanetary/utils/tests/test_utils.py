
import pytest
from ... import utils
import numpy as np

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
    





