from ... import utils
import numpy as np
import astropy.units as u
from astropy.time import Time


def test_body():

    epoch = Time('2018-01-01T00:00:00', format='isot', scale='utc')
    jup = utils.Body('Jupiter', epoch=epoch, location='T17')  # Keck 2

    # basic data
    assert jup.name == 'Jupiter'
    assert jup.mass == 1.89813e27 * u.kg
    assert jup.req == 71492 * u.km
    assert jup.rpol == 66854 * u.km
    assert jup.ravg == 69173 * u.km
    assert jup.rvol == 69911 * u.km
    assert jup.accel_g == 24.79 * u.m / u.s**2
    assert jup.semi_major_axis == 5.203810698356417 * u.AU
    assert jup.t_rot == 9.925 * u.h
    assert jup.app_dia_max == 50.1 * u.arcsec
    assert jup.app_dia_min == 30.5 * u.arcsec
    assert jup.location == 'T17'
    assert jup.epoch == epoch

    # ephemeris
    assert np.isclose(jup.ephem['r'], 5.432247809300, atol=1e-6)
    assert np.isclose(jup.ephem['delta'], 5.95806215806774, atol=1e-6)


def test_body_print(capsys):

    epoch = Time('2018-01-01T00:00:00', format='isot', scale='utc')
    jup = utils.Body('Jupiter', epoch=epoch, location='T17')  # Keck 2
    print(jup)
    out, err = capsys.readouterr()
    assert out == 'pylantary.utils.Body instance; Jupiter, Horizons ID 599\n'
