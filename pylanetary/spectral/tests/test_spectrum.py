import numpy as np
import pytest

from pylanetary.spectral.core import Spectrum, SpectrumData


def test_spectrum_basic_init():
    """Test that Spectrum initializes with minimal valid input."""

    intensity = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    frequency = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

    data = SpectrumData(
        intensity=intensity,
        frequency=frequency,
        nu_rest=102.0,
        i=0,
        j=0
    )

    spec = Spectrum(data)

    # Basic attribute checks
    assert spec.I_nu_arr is intensity
    assert spec.nu_arr is frequency
    assert spec.nu_rest == 102.0
    assert spec.i == 0
    assert spec.j == 0

    # Derived quantities
    assert spec.num_chan == len(frequency)
    assert spec.del_nu == pytest.approx(1.0)

    # Stats exist
    assert spec.rms is not None
    assert spec.std is not None


def test_spectrum_defaults():
    """Test default values are set correctly."""

    intensity = np.ones(10)
    frequency = np.linspace(100, 110, 10)

    data = SpectrumData(intensity=intensity, frequency=frequency)

    spec = Spectrum(data)

    # Defaults
    assert spec.snr == 3.0
    assert spec.signal_flag is False
    assert spec.fit_result is not None


def test_spectrum_weights_none_by_default():
    """Weights should be None unless explicitly set."""

    intensity = np.ones(5)
    frequency = np.arange(5)

    data = SpectrumData(intensity=intensity, frequency=frequency)

    spec = Spectrum(data)

    assert spec.weights is None
    