"""Reader/writer round-trips, with the anisotropic voxel size as the focus."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from synapse_deconv.readers import ReadError, _normalise_axes, _to_micrometres, read_stack
from synapse_deconv.writers import to_uint16, write_ome_tiff

VOXEL = (0.30, 0.095, 0.095)


@pytest.fixture
def stack_file(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=(3, 12, 32, 32), dtype=np.uint16)
    path = tmp_path / "stack.ome.tif"
    write_ome_tiff(path, data, voxel_size_um=VOXEL,
                   channel_names=["a", "b", "c"])
    return path, data


def test_ome_tiff_round_trip_preserves_data_and_calibration(stack_file):
    path, original = stack_file
    stack = read_stack(path)
    np.testing.assert_array_equal(stack.data, original)
    assert stack.data.dtype == np.uint16
    assert stack.voxel_size_um == pytest.approx(VOXEL)
    assert stack.voxel_size_source == "metadata"


def test_anisotropy_is_not_flattened(stack_file):
    """The whole point: dz must never be replaced by dxy."""
    path, _ = stack_file
    stack = read_stack(path)
    dz, dy, dx = stack.voxel_size_um
    assert dz != dx
    assert dz / dx == pytest.approx(0.30 / 0.095, rel=1e-6)


def test_channel_names_survive_the_round_trip(stack_file):
    path, _ = stack_file
    stack = read_stack(path)
    assert [c.name for c in stack.channels] == ["a", "b", "c"]


def test_fallback_is_used_when_calibration_is_absent(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), np.zeros((10, 16, 16), dtype=np.uint16))
    stack = read_stack(path, fallback_xy_um=0.095, fallback_z_um=0.30)
    assert stack.voxel_size_um == pytest.approx(VOXEL)
    assert stack.voxel_size_source == "fallback"
    assert any("fallback" in w for w in stack.warnings)


def test_config_source_overrides_the_file(stack_file):
    path, _ = stack_file
    stack = read_stack(path, voxel_size_source="config",
                       fallback_xy_um=0.2, fallback_z_um=0.5)
    assert stack.voxel_size_um == pytest.approx((0.5, 0.2, 0.2))
    assert stack.voxel_size_source == "config"


def test_unexpected_voxel_size_warns(tmp_path):
    path = tmp_path / "coarse.ome.tif"
    write_ome_tiff(path, np.zeros((1, 8, 16, 16), dtype=np.uint16),
                   voxel_size_um=(0.30, 0.4, 0.4))
    stack = read_stack(path, fallback_xy_um=0.095, fallback_z_um=0.30)
    assert any("differs from the expected" in w for w in stack.warnings)


def test_imagej_calibration_is_read(tmp_path):
    """A Fiji-saved hyperstack carries its scale in the ImageJ header, not OME."""
    path = tmp_path / "hyperstack.tif"
    tifffile.imwrite(
        str(path),
        np.zeros((12, 3, 32, 32), dtype=np.uint16),
        imagej=True,
        resolution=(1 / 0.095, 1 / 0.095),
        metadata={"axes": "ZCYX", "spacing": 0.30, "unit": "um",
                  "slices": 12, "channels": 3},
    )
    stack = read_stack(path)
    assert stack.data.shape == (3, 12, 32, 32)
    assert stack.voxel_size_um == pytest.approx(VOXEL, rel=1e-6)
    assert stack.voxel_size_source == "metadata"


def test_plain_tiff_resolution_tags_are_not_trusted(tmp_path):
    """XResolution=1 with ResolutionUnit=NONE means 'uncalibrated', not 1 um."""
    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), np.zeros((10, 16, 16), dtype=np.uint16))
    stack = read_stack(path, fallback_xy_um=0.095, fallback_z_um=0.30)
    assert stack.voxel_size_um[1] == pytest.approx(0.095)
    assert stack.voxel_size_source == "fallback"


def test_single_channel_stack_gets_a_channel_axis(tmp_path):
    path = tmp_path / "one.tif"
    tifffile.imwrite(str(path), np.zeros((10, 16, 16), dtype=np.uint16))
    stack = read_stack(path)
    assert stack.data.shape == (1, 10, 16, 16)


def test_missing_file_raises():
    with pytest.raises(ReadError, match="file not found"):
        read_stack("no/such/file.oib")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ReadError, match="unsupported file extension"):
        read_stack(path)


def test_corrupt_tiff_raises(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00garbage")
    with pytest.raises(ReadError, match="could not open TIFF"):
        read_stack(path)


# -- axis normalisation ----------------------------------------------------

@pytest.mark.parametrize("axes,shape,expected", [
    ("CZYX", (3, 10, 8, 8), (3, 10, 8, 8)),
    ("ZCYX", (10, 3, 8, 8), (3, 10, 8, 8)),
    ("ZYX", (10, 8, 8), (1, 10, 8, 8)),
    ("TZCYX", (1, 10, 3, 8, 8), (3, 10, 8, 8)),
])
def test_axis_orders_normalise_to_czyx(axes, shape, expected):
    data, _ = _normalise_axes(np.zeros(shape, dtype=np.uint16), axes, __import__("pathlib").Path("x"))
    assert data.shape == expected


def test_extra_timepoints_are_dropped_with_a_warning():
    from pathlib import Path

    data = np.zeros((4, 10, 3, 8, 8), dtype=np.uint16)
    data[2] = 7
    out, warnings = _normalise_axes(data, "TZCYX", Path("x"))
    assert out.shape == (3, 10, 8, 8)
    assert out.max() == 0            # kept the first timepoint
    assert any("only the first is processed" in w for w in warnings)


def test_stack_without_z_is_rejected():
    from pathlib import Path

    with pytest.raises(ReadError, match="no Z axis"):
        _normalise_axes(np.zeros((3, 8, 8), dtype=np.uint16), "CYX", Path("x"))


# -- units -----------------------------------------------------------------

@pytest.mark.parametrize("value,unit,expected", [
    (300.0, "nm", 0.3),
    (0.3, "um", 0.3),
    (0.3, "µm", 0.3),
    (0.0003, "mm", 0.3),
    (0.3, None, 0.3),
    (0.3, "unknown", 0.3),
])
def test_unit_conversion(value, unit, expected):
    assert _to_micrometres(value, unit) == pytest.approx(expected)


# -- 16-bit conversion -----------------------------------------------------

def test_clip_policy_preserves_absolute_values():
    volume = np.array([[[0.0, 10.4, 10.6, 70000.0]]])
    out, report = to_uint16(volume, "clip")
    assert out.tolist() == [[[0, 10, 11, 65535]]]
    assert report.clipped_voxels == 1
    assert report.scale_factor == 1.0


def test_intensity_scale_prevents_clipping():
    volume = np.array([[[0.0, 100000.0]]])
    out, report = to_uint16(volume, "clip", intensity_scale=0.5)
    assert out.max() == 50000
    assert report.clipped_voxels == 0


def test_rescale_policy_fills_the_range():
    volume = np.array([[[0.0, 1.0, 2.0]]])
    out, _ = to_uint16(volume, "rescale_per_stack")
    assert out.max() == 65535


def test_non_finite_values_are_zeroed():
    volume = np.array([[[np.nan, np.inf, -np.inf, 5.0]]])
    out, _ = to_uint16(volume, "clip")
    assert out.tolist() == [[[0, 65535, 0, 5]]]


def test_writer_refuses_non_uint16(tmp_path):
    with pytest.raises(ValueError, match="stays 16-bit"):
        write_ome_tiff(tmp_path / "x.ome.tif",
                       np.zeros((1, 4, 8, 8), dtype=np.float32), voxel_size_um=VOXEL)


def test_writer_requires_four_dimensions(tmp_path):
    with pytest.raises(ValueError, match=r"\(C, Z, Y, X\)"):
        write_ome_tiff(tmp_path / "x.ome.tif",
                       np.zeros((4, 8, 8), dtype=np.uint16), voxel_size_um=VOXEL)


# -- background estimation -------------------------------------------------

def test_modal_background_finds_the_detector_offset():
    """The offset must be recovered from a punctate volume, not the noise tail."""
    from synapse_deconv.qc import estimate_background

    rng = np.random.default_rng(4)
    volume = rng.normal(100.0, 8.0, size=(20, 64, 64))
    volume[5, 30, 30] = 5000        # a few bright puncta must not shift it
    volume[8, 40, 20] = 4000
    volume = np.clip(volume, 0, 65535).astype(np.uint16)
    assert estimate_background(volume) == pytest.approx(100, abs=3)


def test_modal_background_beats_a_low_percentile():
    """A low percentile sits on the lower noise tail and under-estimates."""
    from synapse_deconv.qc import estimate_background

    rng = np.random.default_rng(5)
    volume = np.clip(rng.normal(200.0, 15.0, size=(10, 64, 64)), 0, 65535).astype(np.uint16)
    modal = estimate_background(volume)
    percentile = float(np.percentile(volume, 1.0))
    assert abs(modal - 200) < abs(percentile - 200)


def test_modal_background_handles_float_and_empty():
    from synapse_deconv.qc import estimate_background

    # Float volumes go through a 512-bin histogram, so the answer is only
    # accurate to one bin width.
    assert estimate_background(np.full((4, 8, 8), 3.5, dtype=np.float32)) == pytest.approx(
        3.5, abs=0.01
    )
    assert estimate_background(np.array([], dtype=np.uint16)) == 0.0
