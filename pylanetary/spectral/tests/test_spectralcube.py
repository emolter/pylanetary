import numpy as np
import pytest
from astropy.io import fits

from pylanetary.spectral.core import SpectralCube


def create_test_fits(path):
    """Create a minimal valid FITS cube for testing."""

    data = np.zeros((1, 5, 4, 3))  # (stokes?, freq, y, x)

    hdu = fits.PrimaryHDU(data)

    hdr = hdu.header
    hdr["NAXIS"] = 4
    hdr["NAXIS1"] = 3
    hdr["NAXIS2"] = 4
    hdr["NAXIS3"] = 5

    hdr["CRPIX3"] = 1
    hdr["CDELT3"] = 1.0
    hdr["CRVAL3"] = 100.0

    hdr["DATE-OBS"] = "2020-01-01T00:00:00"

    hdr["BMAJ"] = 1.0
    hdr["BMIN"] = 1.0
    hdr["BPA"] = 0.0

    hdr["CDELT1"] = 0.0001
    hdr["CDELT2"] = 0.0001

    hdu.writeto(path)


def test_spectralcube_init(tmp_path):
    """Test SpectralCube initializes from FITS."""

    fits_path = tmp_path / "test_cube.fits"
    create_test_fits(fits_path)

    cube = SpectralCube(str(fits_path), nu_rest_str="CO10")

    # Basic attributes
    assert cube.n_chan == 5
    assert cube.ni == 3
    assert cube.nj == 4

    # Data shape
    assert cube.data.shape == (5, 4, 3)

    # Frequency array exists
    assert len(cube.nu_arr) == 5

    # Rest frequency set
    assert cube.nu_rest is not None


def test_spectralcube_mask_default(tmp_path):
    """Mask should default to all False."""

    fits_path = tmp_path / "test_cube.fits"
    create_test_fits(fits_path)

    cube = SpectralCube(str(fits_path), nu_rest_str="CO10")

    assert cube.mask.shape == (cube.ni, cube.nj)
    assert np.all(cube.mask == False)


def test_spectralcube_invalid_rest_frequency(tmp_path):
    """Invalid rest frequency should raise ValueError."""

    fits_path = tmp_path / "test_cube.fits"
    create_test_fits(fits_path)

    with pytest.raises(ValueError):
        SpectralCube(str(fits_path), nu_rest_str="NOT_A_LINE")