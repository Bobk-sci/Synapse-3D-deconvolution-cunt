"""Correctness checks on the Richardson-Lucy implementation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from synapse_deconv.config import OpticsConfig, PSFConfig
from synapse_deconv.deconvolution import richardson_lucy
from synapse_deconv.psf import compute_psf

VOXEL = (0.30, 0.095, 0.095)


@pytest.fixture(scope="module")
def psf():
    return compute_psf(
        voxel_size_um=VOXEL, emission_nm=519.0, optics=OpticsConfig(),
        psf_cfg=PSFConfig(cache_dir=None, xy_size=21, z_size=11),
    ).data


@pytest.fixture(scope="module")
def scene(psf):
    """Two puncta 6 pixels apart plus one isolated, blurred and Poisson-noised."""
    rng = np.random.default_rng(7)
    truth = np.zeros((20, 80, 80), dtype=np.float64)
    truth[10, 40, 37] = 4.0e5
    truth[10, 40, 43] = 4.0e5
    truth[6, 20, 20] = 3.0e5
    blurred = np.maximum(fftconvolve(truth, psf, mode="same"), 0.0)
    observed = rng.poisson(blurred).astype(np.float32)
    return truth, blurred, observed


def test_output_is_non_negative_and_finite(psf, scene):
    _, _, observed = scene
    result = richardson_lucy(observed, psf, iterations=15)
    assert result.data.min() >= 0.0
    assert np.all(np.isfinite(result.data))
    assert result.data.shape == observed.shape


def test_total_intensity_is_conserved(psf, scene):
    """Richardson-Lucy redistributes photons; it must not invent or lose them."""
    _, _, observed = scene
    result = richardson_lucy(observed, psf, iterations=25)
    assert result.diagnostics["intensity_ratio"] == pytest.approx(1.0, abs=0.02)


def test_deconvolution_sharpens_towards_the_truth(psf, scene):
    """The deconvolved punctum must be narrower than the blurred one."""
    _, blurred, observed = scene
    result = richardson_lucy(observed, psf, iterations=25)

    def width(volume):
        profile = volume[6, 20, 12:29].astype(float)
        return float((profile >= profile.max() / 2).sum())

    assert width(result.data) < width(blurred)


def test_close_puncta_are_separated(psf, scene):
    """The dip between two puncta 0.57 um apart must deepen after deconvolution."""
    _, blurred, observed = scene
    result = richardson_lucy(observed, psf, iterations=25)

    def contrast(volume):
        left = float(volume[10, 40, 37])
        right = float(volume[10, 40, 43])
        valley = float(volume[10, 40, 40])
        return (min(left, right) - valley) / min(left, right)

    assert contrast(result.data) > contrast(blurred)
    assert contrast(result.data) > 0.5


def test_more_iterations_sharpen_further(psf, scene):
    _, _, observed = scene
    peaks = [
        float(richardson_lucy(observed, psf, iterations=n).data[10, 40, 37])
        for n in (5, 15, 30)
    ]
    assert peaks[0] < peaks[1] < peaks[2]


def test_convergence_decreases(psf, scene):
    _, _, observed = scene
    result = richardson_lucy(observed, psf, iterations=20)
    assert len(result.convergence) == 20
    # Not monotone iteration by iteration, but the trend must be downwards.
    assert result.convergence[-1] < result.convergence[0]


def test_deterministic(psf, scene):
    _, _, observed = scene
    a = richardson_lucy(observed, psf, iterations=10)
    b = richardson_lucy(observed, psf, iterations=10)
    np.testing.assert_array_equal(a.data, b.data)


def test_uniform_image_stays_uniform(psf):
    """A flat field has nothing to deconvolve; it must survive unchanged."""
    flat = np.full((16, 48, 48), 500.0, dtype=np.float32)
    result = richardson_lucy(flat, psf, iterations=10, pad_mode="edge")
    interior = result.data[4:-4, 8:-8, 8:-8]
    np.testing.assert_allclose(interior, 500.0, rtol=0.02)


def test_tv_regularization_reduces_noise(psf):
    """On a pure-noise field, TV must give a smoother result than plain RL."""
    rng = np.random.default_rng(3)
    noise = rng.poisson(200, size=(16, 48, 48)).astype(np.float32)
    plain = richardson_lucy(noise, psf, iterations=30, regularization="none")
    regularised = richardson_lucy(noise, psf, iterations=30,
                                  regularization="tv", tv_lambda=0.01)
    assert regularised.data.std() < plain.data.std()
    assert regularised.diagnostics["regularization"] == "tv"


def test_rejects_a_psf_larger_than_the_image(psf):
    small = np.ones((4, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="larger than the image"):
        richardson_lucy(small, psf, iterations=1)


def test_rejects_non_finite_input(psf):
    volume = np.ones((16, 48, 48), dtype=np.float32)
    volume[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        richardson_lucy(volume, psf, iterations=1)


def test_rejects_negative_input(psf):
    volume = np.ones((16, 48, 48), dtype=np.float32)
    volume[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        richardson_lucy(volume, psf, iterations=1)


def test_rejects_degenerate_psf():
    volume = np.ones((16, 48, 48), dtype=np.float32)
    with pytest.raises(ValueError, match="finite positive sum"):
        richardson_lucy(volume, np.zeros((5, 5, 5)), iterations=1)


def test_float64_matches_float32_closely(psf, scene):
    _, _, observed = scene
    single = richardson_lucy(observed, psf, iterations=10, dtype="float32")
    double = richardson_lucy(observed, psf, iterations=10, dtype="float64")
    scale = float(double.data.max())
    np.testing.assert_allclose(single.data, double.data, atol=scale * 1e-3)
