"""Physics checks on the theoretical PSF."""

from __future__ import annotations

import numpy as np
import pytest

from synapse_deconv.config import OpticsConfig, PSFConfig
from synapse_deconv.psf import compute_psf, measure_fwhm, theoretical_resolution

FINE_VOXEL = (0.05, 0.05, 0.05)
ACQUISITION_VOXEL = (0.30, 0.095, 0.095)


def build(voxel, emission=519.0, **optics_kwargs):
    optics = OpticsConfig(**optics_kwargs)
    cfg = PSFConfig(cache_dir=None, xy_size=61, z_size=61, oversample_xy=2)
    return compute_psf(voxel_size_um=voxel, emission_nm=emission,
                       optics=optics, psf_cfg=cfg)


def test_psf_is_normalised_and_non_negative():
    psf = build(ACQUISITION_VOXEL).data
    assert psf.min() >= 0.0
    assert psf.sum() == pytest.approx(1.0, rel=1e-9)
    assert np.all(np.isfinite(psf))


def test_psf_peaks_at_the_centre_without_aberration():
    psf = build(ACQUISITION_VOXEL, sample_ri=1.515, particle_depth_um=0.0).data
    assert np.unravel_index(int(np.argmax(psf)), psf.shape) == tuple(s // 2 for s in psf.shape)


def test_lateral_fwhm_matches_the_diffraction_limit():
    """FWHM should land on 0.51 lambda / NA within the sampling error."""
    psf = build(FINE_VOXEL, emission=519.0, sample_ri=1.515, particle_depth_um=0.0)
    measured = measure_fwhm(psf.data, FINE_VOXEL)["fwhm_x_um"]
    expected = 0.51 * 0.519 / 1.40
    assert measured == pytest.approx(expected, abs=0.02)


def test_axial_fwhm_matches_the_non_paraxial_prediction():
    """At NA 1.4 the exact width is 4 * 1.3916 * lambda / (8 pi n sin^2(alpha/2))."""
    na, ni, lam = 1.40, 1.515, 0.519
    sin_alpha = na / ni
    cos_alpha = np.sqrt(1 - sin_alpha**2)
    expected = 11.13 * lam / (8 * np.pi * ni * (1 - cos_alpha) / 2)

    psf = build(FINE_VOXEL, emission=519.0, sample_ri=1.515, particle_depth_um=0.0)
    measured = measure_fwhm(psf.data, FINE_VOXEL)["fwhm_z_um"]
    assert measured == pytest.approx(expected, rel=0.10)


def test_psf_width_scales_with_wavelength():
    widths = [
        measure_fwhm(build(ACQUISITION_VOXEL, emission=em).data, ACQUISITION_VOXEL)["fwhm_x_um"]
        for em in (421.0, 519.0, 617.0)
    ]
    assert widths[0] < widths[1] < widths[2], widths


def test_index_mismatch_broadens_the_psf():
    """Oil objective into an aqueous sample: deeper means worse."""
    matched = build(ACQUISITION_VOXEL, sample_ri=1.515, particle_depth_um=0.0)
    mismatched = build(ACQUISITION_VOXEL, sample_ri=1.33, particle_depth_um=8.0)

    matched_fwhm = measure_fwhm(matched.data, ACQUISITION_VOXEL)
    mismatched_fwhm = measure_fwhm(mismatched.data, ACQUISITION_VOXEL)
    assert mismatched_fwhm["fwhm_z_um"] > matched_fwhm["fwhm_z_um"]
    # The Strehl ratio must drop: the peak of a normalised PSF falls when the
    # energy spreads into the aberration skirt.
    assert mismatched.data.max() < matched.data.max()


def test_born_wolf_ignores_the_sample_index():
    """born_wolf collapses the layered model, so sample_ri must not matter."""
    cfg = PSFConfig(model="born_wolf", cache_dir=None, xy_size=31, z_size=21)
    a = compute_psf(voxel_size_um=ACQUISITION_VOXEL, emission_nm=519.0,
                    optics=OpticsConfig(sample_ri=1.33, particle_depth_um=5.0), psf_cfg=cfg)
    b = compute_psf(voxel_size_um=ACQUISITION_VOXEL, emission_nm=519.0,
                    optics=OpticsConfig(sample_ri=1.47, particle_depth_um=0.0), psf_cfg=cfg)
    np.testing.assert_allclose(a.data, b.data, rtol=1e-10, atol=1e-12)


def test_gibson_lanni_differs_from_born_wolf_under_mismatch():
    optics = OpticsConfig(sample_ri=1.33, particle_depth_um=8.0)
    gl = compute_psf(voxel_size_um=ACQUISITION_VOXEL, emission_nm=519.0, optics=optics,
                     psf_cfg=PSFConfig(model="gibson_lanni", cache_dir=None,
                                       xy_size=31, z_size=21))
    bw = compute_psf(voxel_size_um=ACQUISITION_VOXEL, emission_nm=519.0, optics=optics,
                     psf_cfg=PSFConfig(model="born_wolf", cache_dir=None,
                                       xy_size=31, z_size=21))
    assert not np.allclose(gl.data, bw.data, atol=1e-6)


def test_anisotropic_voxel_is_honoured():
    """A PSF on an anisotropic grid must not be the isotropic one rescaled."""
    aniso = build((0.30, 0.095, 0.095), sample_ri=1.515, particle_depth_um=0.0)
    iso = build((0.095, 0.095, 0.095), sample_ri=1.515, particle_depth_um=0.0)
    # Same physical width, so the anisotropic kernel needs ~3x fewer z samples
    # to cover it: the axial profiles cannot be equal element-wise.
    centre = aniso.data.shape[1] // 2
    aniso_profile = aniso.data[:, centre, centre]
    iso_profile = iso.data[:, iso.data.shape[1] // 2, iso.data.shape[2] // 2]
    aniso_width = measure_fwhm(aniso.data, (0.30, 0.095, 0.095))["fwhm_z_um"]
    iso_width = measure_fwhm(iso.data, (0.095, 0.095, 0.095))["fwhm_z_um"]
    assert aniso_width == pytest.approx(iso_width, rel=0.25)
    assert len(aniso_profile) == len(iso_profile)   # same kernel size in voxels
    assert not np.allclose(aniso_profile, iso_profile, rtol=0.05)


def test_confocal_psf_is_narrower_than_emission_only():
    optics = OpticsConfig(sample_ri=1.515, particle_depth_um=0.0)
    common = dict(cache_dir=None, xy_size=41, z_size=31)
    emission = compute_psf(voxel_size_um=FINE_VOXEL, emission_nm=519.0, optics=optics,
                           psf_cfg=PSFConfig(mode="emission", **common))
    confocal = compute_psf(voxel_size_um=FINE_VOXEL, emission_nm=519.0, excitation_nm=488.0,
                           optics=optics,
                           psf_cfg=PSFConfig(mode="confocal", pinhole_airy_units=1.0, **common))

    emission_fwhm = measure_fwhm(emission.data, FINE_VOXEL)
    confocal_fwhm = measure_fwhm(confocal.data, FINE_VOXEL)
    assert confocal_fwhm["fwhm_x_um"] < emission_fwhm["fwhm_x_um"]
    assert confocal_fwhm["fwhm_z_um"] < emission_fwhm["fwhm_z_um"]
    assert confocal.data.sum() == pytest.approx(1.0, rel=1e-9)


def test_theoretical_resolution_formulas():
    lateral, axial = theoretical_resolution(519.0, 1.40, 1.515)
    assert lateral == pytest.approx(0.61 * 0.519 / 1.40)
    assert axial == pytest.approx(2 * 0.519 * 1.515 / 1.4**2)


def test_cache_round_trip(tmp_path):
    cfg = PSFConfig(cache_dir=str(tmp_path), xy_size=21, z_size=11)
    kwargs = dict(voxel_size_um=ACQUISITION_VOXEL, emission_nm=519.0,
                  optics=OpticsConfig(), psf_cfg=cfg)
    first = compute_psf(**kwargs)
    assert list(tmp_path.glob("psf_*.npy"))
    second = compute_psf(**kwargs)
    np.testing.assert_array_equal(first.data, second.data)
    assert first.cache_key == second.cache_key


def test_kernel_size_is_capped_by_the_stack():
    psf = compute_psf(
        voxel_size_um=ACQUISITION_VOXEL, emission_nm=617.0, optics=OpticsConfig(),
        psf_cfg=PSFConfig(cache_dir=None), stack_shape=(9, 64, 64),
    )
    assert psf.shape[0] <= 9
    assert all(s % 2 == 1 for s in psf.shape)
