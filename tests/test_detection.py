"""Punctum detection: anisotropy handling, thresholding, and size measurement."""

from __future__ import annotations

import numpy as np
import pytest

from synapse_deconv.detection import detect_puncta, estimate_noise_mad, subtract_background

VOXEL = (0.30, 0.095, 0.095)          # strongly anisotropic, as acquired
SHAPE_UM = (6.0, 12.0, 12.0)


def make_volume(points_um, voxel=VOXEL, shape_um=SHAPE_UM, amplitude=2000.0,
                sigma_um=0.11, offset=20.0, noise=5.0, seed=0):
    """Gaussian blobs at physical positions, with Poisson + read noise."""
    rng = np.random.default_rng(seed)
    dz, dy, dx = voxel
    shape = (int(shape_um[0] / dz), int(shape_um[1] / dy), int(shape_um[2] / dx))
    volume = np.zeros(shape, dtype=np.float64)
    zz = np.arange(shape[0])[:, None, None] * dz
    yy = np.arange(shape[1])[None, :, None] * dy
    xx = np.arange(shape[2])[None, None, :] * dx
    for cz, cy, cx in points_um:
        volume += amplitude * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_um ** 2)
        )
    noisy = rng.poisson(volume) + offset + rng.normal(0, noise, size=shape)
    return np.clip(noisy, 0, 65535).astype(np.uint16)


def grid_points(n_per_axis=4, shape_um=SHAPE_UM):
    zs = np.linspace(1.5, shape_um[0] - 1.5, 2)
    ys = np.linspace(2.0, shape_um[1] - 2.0, n_per_axis)
    xs = np.linspace(2.0, shape_um[2] - 2.0, n_per_axis)
    return [(z, y, x) for z in zs for y in ys for x in xs]


def test_finds_the_expected_number_of_puncta():
    points = grid_points()
    volume = make_volume(points)
    result = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    assert result.count == pytest.approx(len(points), abs=2)


def test_centroids_land_on_the_true_positions():
    points = grid_points(3)
    volume = make_volume(points)
    result = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)

    from scipy.spatial import cKDTree

    distances, _ = cKDTree(result.centroids_um).query(np.asarray(points))
    # Within one z-step: the axial sampling is the limiting term.
    assert float(np.median(distances)) < VOXEL[0]


def test_empty_volume_yields_no_puncta_and_says_so():
    volume = make_volume([], noise=5.0)
    result = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=8.0)
    assert result.count == 0
    assert result.centroids_um.shape == (0, 3)
    assert result.diagnostics["n_seeds"] == 0


def test_threshold_is_monotone_in_sigma():
    volume = make_volume(grid_points(), amplitude=600.0)
    counts = [
        detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35],
                      threshold_sigma=s).count
        for s in (3.0, 5.0, 12.0, 30.0)
    ]
    assert counts == sorted(counts, reverse=True), counts


def test_threshold_is_calibrated_on_the_local_maxima():
    """A threshold taken from the bulk lets every noise maximum through."""
    volume = make_volume(grid_points())
    result = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    diagnostics = result.diagnostics
    # Local maxima vastly outnumber the real puncta; that gap is exactly what
    # the calibration has to absorb, and a bulk-derived threshold would not.
    assert diagnostics["n_local_maxima"] > 10 * result.count
    assert diagnostics["log_threshold"] > diagnostics["log_peak_median"]


def test_anisotropy_is_respected():
    """The same physical blob must be found whatever the z-step."""
    points = grid_points(3)
    fine = (0.10, 0.095, 0.095)
    coarse = (0.30, 0.095, 0.095)
    n_fine = detect_puncta(make_volume(points, voxel=fine), fine,
                           punctum_diameters_um=[0.35], threshold_sigma=5.0).count
    n_coarse = detect_puncta(make_volume(points, voxel=coarse), coarse,
                             punctum_diameters_um=[0.35], threshold_sigma=5.0).count
    # Both must land on the truth. They are not required to be identical: a
    # 3x finer z-step means 3x more voxels, hence 3x more independent
    # neighbourhoods competing for the threshold, so a few more false positives
    # slip through at the same sigma. Sampling must stay constant within a study.
    assert abs(n_fine - len(points)) <= 5, n_fine
    assert abs(n_coarse - len(points)) <= 5, n_coarse


def test_volumes_are_physical_not_voxel_counts():
    volume = make_volume(grid_points(3))
    result = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    voxel_volume = VOXEL[0] * VOXEL[1] * VOXEL[2]
    assert np.all(result.volumes_um3 >= voxel_volume * 0.999)
    # A 0.11 um sigma blob is well under a cubic micrometre.
    assert float(np.median(result.volumes_um3)) < 0.5


def test_size_gate_rejects_out_of_range_regions():
    volume = make_volume(grid_points(3))
    loose = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    tight = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0,
                          min_volume_um3=10.0)
    assert loose.count > 0
    assert tight.count == 0
    assert tight.diagnostics["n_rejected_by_size"] == loose.count


def test_border_artefacts_do_not_become_puncta():
    """A signal running off the edge leaves a rim the top-hat cannot handle."""
    points = grid_points(3)
    volume = make_volume(points).astype(np.float32)
    ramp = np.linspace(0, 400, volume.shape[2])[None, None, :]

    found = detect_puncta(volume + ramp, VOXEL, punctum_diameters_um=[0.35],
                          threshold_sigma=5.0, background_radius_um=1.0)
    assert found.count == pytest.approx(len(points), abs=3), found.count
    # Nothing survives inside the excluded rim.
    bz, by, bx = found.diagnostics["border_excluded_voxels"]
    xs = found.centroids_um[:, 2] / VOXEL[2]
    assert xs.min() >= bx - 0.5
    assert xs.max() <= volume.shape[2] - bx - 0.5


def test_top_hat_removes_a_broad_gradient_but_keeps_puncta():
    points = grid_points(3)
    volume = make_volume(points).astype(np.float32)
    dz, dy, dx = VOXEL
    ramp = np.linspace(0, 400, volume.shape[2])[None, None, :]
    with_veil = volume + ramp

    clean = subtract_background(with_veil, VOXEL, radius_um=1.0)
    # The veil is gone: the residual baseline no longer tracks x.
    left = float(np.median(clean[..., : clean.shape[2] // 4]))
    right = float(np.median(clean[..., -clean.shape[2] // 4:]))
    assert abs(left - right) < 20

    found = detect_puncta(with_veil, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    assert found.count == pytest.approx(len(points), abs=2)


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match="expected a 3D volume"):
        detect_puncta(np.zeros((10, 10)), VOXEL, punctum_diameters_um=[0.35])


def test_requires_a_scale():
    with pytest.raises(ValueError, match="at least one punctum diameter"):
        detect_puncta(np.zeros((8, 16, 16)), VOXEL, punctum_diameters_um=[])


def test_mad_is_not_dragged_by_outliers():
    values = np.concatenate([np.random.default_rng(0).normal(0, 1, 10_000),
                             np.full(200, 1000.0)])
    assert estimate_noise_mad(values) == pytest.approx(1.0, abs=0.1)
    assert float(values.std()) > 5


def test_detection_is_deterministic():
    volume = make_volume(grid_points(3))
    a = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    b = detect_puncta(volume, VOXEL, punctum_diameters_um=[0.35], threshold_sigma=5.0)
    np.testing.assert_array_equal(a.centroids_um, b.centroids_um)
