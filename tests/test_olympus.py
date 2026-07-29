"""Olympus FluoView (.oif) metadata parsing, on a synthetic FV1000 dataset."""

from __future__ import annotations

import numpy as np
import pytest

from fv1000_fixture import write_oif
from synapse_deconv.config import Config
from synapse_deconv.pipeline import process_stack
from synapse_deconv.readers import read_stack


def test_reads_data_and_anisotropic_calibration(tmp_path):
    path = write_oif(tmp_path, shape=(2, 8, 32, 32), xy_um=0.095, z_um=0.30)
    stack = read_stack(path)

    assert stack.data.shape == (2, 8, 32, 32)
    assert stack.data.dtype == np.uint16
    assert stack.voxel_size_um == pytest.approx((0.30, 0.095, 0.095), rel=1e-6)
    assert stack.voxel_size_source == "metadata"
    assert stack.warnings == []


def test_z_positions_in_nanometres_are_converted(tmp_path):
    """The FV1000 stores Z in nm on some configurations; um and nm must agree."""
    in_nm = read_stack(write_oif(tmp_path / "nm", z_um=0.30, z_unit="nm"))
    in_um = read_stack(write_oif(tmp_path / "um", z_um=0.30, z_unit="um"))
    assert in_nm.voxel_size_um[0] == pytest.approx(0.30, rel=1e-6)
    assert in_nm.voxel_size_um == pytest.approx(in_um.voxel_size_um)


def test_per_channel_wavelengths_and_pinhole_are_read(tmp_path):
    path = write_oif(tmp_path, emission_nm=(421.0, 519.0), excitation_nm=(405.0, 488.0),
                     channel_names=("Syn1", "PSD95"), pinhole_nm=100000.0)
    stack = read_stack(path)

    assert [c.name for c in stack.channels] == ["Syn1", "PSD95"]
    assert [c.emission_nm for c in stack.channels] == [421.0, 519.0]
    assert [c.excitation_nm for c in stack.channels] == [405.0, 488.0]
    # PinholeDiameter is nanometres in the file: 100 um back-projected.
    assert stack.channels[0].pinhole_um == pytest.approx(100.0)


def test_axis_spacing_divisor_convention(tmp_path):
    """n_minus_1 treats the positions as sample centres; n as bin edges."""
    path = write_oif(tmp_path, shape=(1, 11, 16, 16), z_um=0.30)
    centres = read_stack(path, spacing_divisor="n_minus_1")
    edges = read_stack(path, spacing_divisor="n")

    assert centres.voxel_size_um[0] == pytest.approx(0.30, rel=1e-6)
    # Same span shared over one more interval.
    assert edges.voxel_size_um[0] == pytest.approx(0.30 * 10 / 11, rel=1e-6)


def test_unexpected_calibration_is_flagged(tmp_path):
    """A 10x coarser Z than configured is more likely a unit slip than reality."""
    path = write_oif(tmp_path, z_um=3.0)
    stack = read_stack(path, fallback_z_um=0.30, tolerance_warn_ratio=0.25)
    assert any("differs from the expected" in w for w in stack.warnings)


def test_config_can_override_the_file_calibration(tmp_path):
    path = write_oif(tmp_path, xy_um=0.2, z_um=0.5)
    stack = read_stack(path, voxel_size_source="config",
                       fallback_xy_um=0.095, fallback_z_um=0.30)
    assert stack.voxel_size_um == pytest.approx((0.30, 0.095, 0.095))


def test_olympus_stack_runs_through_the_pipeline(tmp_path):
    path = write_oif(tmp_path / "raw", shape=(2, 8, 48, 48))
    cfg = Config.from_dict({
        "input": {"directory": str(tmp_path / "raw")},
        "output": {"directory": str(tmp_path / "out"), "save_psf": False},
        "channels": [
            {"name": "Alexa405", "emission_nm": 421, "excitation_nm": 405},
            {"name": "Alexa488", "emission_nm": 519, "excitation_nm": 488},
        ],
        "psf": {"xy_size": 11, "z_size": 7, "cache_dir": None, "oversample_xy": 1},
        "deconvolution": {"iterations": 2},
        "qc": {"enabled": False},
        "logging": {"file": ""},
    })
    result = process_stack(cfg, path)

    assert result.status == "ok"
    assert result.output.exists()
    assert result.voxel_size_um == pytest.approx((0.30, 0.095, 0.095), rel=1e-6)
    assert result.voxel_size_source == "metadata"


def test_config_wavelength_disagreeing_with_the_file_is_reported(tmp_path):
    """The config stays authoritative, but the discrepancy must be logged."""
    path = write_oif(tmp_path / "raw", emission_nm=(421.0, 519.0))
    cfg = Config.from_dict({
        "input": {"directory": str(tmp_path / "raw")},
        "output": {"directory": str(tmp_path / "out"), "save_psf": False},
        "channels": [
            {"name": "Alexa405", "emission_nm": 421, "excitation_nm": 405},
            {"name": "wrong", "emission_nm": 680, "excitation_nm": 640},
        ],
        "psf": {"xy_size": 11, "z_size": 7, "cache_dir": None, "oversample_xy": 1},
        "deconvolution": {"iterations": 2},
        "qc": {"enabled": False},
        "logging": {"file": ""},
    })
    result = process_stack(cfg, path)

    assert any("file reports emission 519" in w for w in result.warnings)
    assert result.channels[1]["emission_nm"] == 680


def test_missing_storage_directory_is_a_read_error(tmp_path):
    from synapse_deconv.readers import ReadError

    path = write_oif(tmp_path)
    for entry in (tmp_path / "scan.oif.files").iterdir():
        entry.unlink()
    (tmp_path / "scan.oif.files").rmdir()

    with pytest.raises(ReadError, match="could not open Olympus file"):
        read_stack(path)


def test_inspect_flags_a_reversed_channel_order(tmp_path, capsys):
    """The classic FV1000 trap: sequential acquisition stored in the other order."""
    import yaml

    from synapse_deconv.cli import main

    write_oif(tmp_path / "raw", stem="scan", shape=(3, 8, 32, 32),
              emission_nm=(617.0, 519.0, 421.0), excitation_nm=(594.0, 488.0, 405.0),
              channel_names=("CH1", "CH2", "CH3"))
    cfg = {
        "input": {"directory": str(tmp_path / "raw")},
        "output": {"directory": str(tmp_path / "out")},
        "channels": [
            {"name": "Alexa405", "emission_nm": 421, "excitation_nm": 405},
            {"name": "Alexa488", "emission_nm": 519, "excitation_nm": 488},
            {"name": "Alexa594", "emission_nm": 617, "excitation_nm": 594},
        ],
        "logging": {"file": ""},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("CHECK THE CHANNEL ORDER") == 2
    assert "pinhole in file" in out


def test_inspect_reports_a_channel_count_mismatch(tmp_path, capsys):
    import yaml

    from synapse_deconv.cli import main

    write_oif(tmp_path / "raw", stem="scan", shape=(2, 8, 32, 32))
    cfg = {
        "input": {"directory": str(tmp_path / "raw")},
        "output": {"directory": str(tmp_path / "out")},
        "channels": [{"name": "a", "emission_nm": 421},
                     {"name": "b", "emission_nm": 519},
                     {"name": "c", "emission_nm": 617}],
        "logging": {"file": ""},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    assert main(["inspect", str(path)]) == 1
    assert "CHANNEL COUNT MISMATCH" in capsys.readouterr().out
