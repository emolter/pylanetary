import pytest
from ... import rings
import numpy as np


def test_b_from_ae():

    assert np.isclose(rings.b_from_ae(10, np.sqrt(2) / 2), 5 * np.sqrt(2))


def test_vector_ellipse():

    origin = np.array([10, 10, 10])
    n = 4
    a = np.array([1, 0, 0])
    b = np.array([0, 0.5, 0])
    result = rings.vector_ellipse(a, b, n, origin=origin)
    expected_result = np.array([[11, 10, 10],
                                [10, 10.5, 10],
                                [9, 10, 10],
                                [10, 9.5, 10]])
    assert np.allclose(result, expected_result, rtol=1e-3)


def test_project_ellipse():
    '''
    parameters for Uranus's epsilon ring on 2013-08-12 00:00
    as in ellipse-projection-play.ipynb
    this has been qualitatively tested against the Planetary Ring Node
    '''
    a = 51149
    e = 0.0079
    params_ring = [350.078, 0.0, 0.0]
    params_sys = [90., 90 + 24.04446, 255.1941]
    origin = np.array([0, 0, 0])
    ell = rings.project_ellipse_double(a, e, params_ring, params_sys,
                                       origin=origin,
                                       n=50,
                                       proj_plane=[0, 0, 1])

    expected_a = np.array([-22099.31625149, -3274.70117637, 0.])
    expected_b = np.array([9403.36772809, 49627.18523954, -0.])
    expected_f0 = np.array([-174.58459839, -25.87013929, 0.])
    expected_f1 = np.array([174.58459839, 25.87013929, -0.])
    expected_ell5 = np.array([-12351.56153907, 26520.8386934, 0.])
    assert np.allclose(expected_a, ell['a'], rtol=1e-3)
    assert np.allclose(expected_b, ell['b'], rtol=1e-3)
    assert np.allclose(expected_f0, ell['f0'], rtol=1e-3)
    assert np.allclose(expected_f1, ell['f1'], rtol=1e-3)
    assert np.allclose(expected_ell5, ell['ell'][5], rtol=1e-3)


def test_calc_abtheta():
    '''
    This has been tested graphically in ellipse_projection_play.ipynb
    '''
    a = np.array([1.2, 0.0, 0.0])
    b = np.array([0.0, 0.8, 0.0])
    n = 100
    ell = rings.vector_ellipse(a, b, n, origin=np.array([0, 0, 0]))
    ai, bi, thetai = rings.calc_abtheta(ell)

    assert np.isclose(ai, a[0], rtol=1e-2)
    assert np.isclose(bi, b[1], rtol=1e-2)
    assert np.isclose(thetai, 0.0, rtol=1e-2)


def test_ring_area():
    '''
    epsilon ring area from Molter et al 2019 doi:10.3847/1538-3881/ab258c
    using B, d_au at 1.3 mm from that paper
    '''
    a = 51149  # km
    e = 0.0079
    wp = 19.7
    wa = 96.4
    width = (wp + wa) / 2
    delta_width = wa - wp
    B = 43.7
    d_au = 19.1
    d_km = d_au * 1.496e8
    A = rings.ring_area(a, e, width, delta_width=delta_width, B=B)

    published_A = 1.57e-12 * d_km**2  # publication lists only steradians

    assert np.isclose(A, published_A, rtol=1e-2)
