import pytest
from ... import rings
import numpy as np

# look in ellipse projection play Jupyter notebook

def test_vector_normalize():
    '''also tests vector_magnitude'''
    
    # n vectors
    v = np.array([[3, 0, 0],
                    [0, 5, 5],
                    [1, 0, 9],
                    [0, 0, 0]])
    expected_result = np.array([[1, 0, 0],
                        [0, np.sqrt(2)/2, np.sqrt(2)/2],
                        [1/np.sqrt(82), 0, 9/np.sqrt(82)],
                        [0, 0, 0]])
    
    assert np.allclose(rings.vector_normalize(v), expected_result, rtol=1e-3)
    
    # one vector
    v2 = np.array([0, 0, 0])
    assert np.all(rings.vector_normalize(v2) == v2)
    
    v3 = np.array([2,0,2])
    expected_result_3 = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2])
    assert np.allclose(rings.vector_normalize(v3), expected_result_3, rtol = 1e-3)


def test_plane_project():
    
    X = np.array([[2, 0, 0],
        [0, 3, 0],
        [0, 0, 1.1],
        [3, 4, 2]])
    Z = np.array([3, 0, 0])
    expected_result = np.array([[0, 0, 0],
                    [0, 3, 0],
                    [0, 0, 1.1],
                    [0, 4, 2]])
    
    assert np.allclose(rings.plane_project(X, Z), expected_result, rtol=1e-3)


def test_double_rotate():
    
    params_ring = [90, 90, 90]
    params_sys = [45, 45, 45]
    v = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]])
    result = rings.double_rotate(v, params_ring, params_sys)
    expected_result = np.asarray([[1/2, -1/2, np.sqrt(2)/2],
                    [1/2 + np.sqrt(2)/4, 1/2 - np.sqrt(2)/4, -1/2],
                    [1/2 - np.sqrt(2)/4, 1/2 + np.sqrt(2)/4, 1/2],
                    [0, 0, 0]])
    
    assert np.allclose(result, expected_result, rtol=1e-3)
