"""End-to-end pipeline behaviour, including batch consistency and error handling."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import yaml

from synapse_deconv.config import Config
from synapse_deconv.pipeline import discover_inputs, process_stack, run_batch
from synapse_deconv.writers import write_ome_tiff

VOXEL = (0.30, 0.095, 0.095)


def make_config(tmp_path, **overrides) -> Config:
    data = {
        "input": {"directory": str(tmp_path / "raw")},
        "output": {"directory": str(tmp_path / "out"), "save_psf": False},
        "channels": [
            {"name": "Alexa405", "emission_nm": 421, "excitation_nm": 405},
            {"name": "Alexa488", "emission_nm": 519, "excitation_nm": 488},
        ],
        # Small kernels and few iterations keep the test fast; the physics is
        # covered by test_psf.py and test_deconvolution.py.
        "psf": {"xy_size": 11, "z_size": 7, "cache_dir": str(tmp_path / "psfcache"),
                "oversample_xy": 1},
        "deconvolution": {"iterations": 3},
        "qc": {"enabled": False},
        "logging": {"file": ""},
    }
    for section, payload in overrides.items():
        if isinstance(payload, dict) and isinstance(data.get(section), dict):
            data[section] = {**data[section], **payload}
        else:
            data[section] = payload
    return Config.from_dict(data)


@pytest.fixture
def raw_dir(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    rng = np.random.default_rng(11)
    for i in range(2):
        volume = np.full((2, 10, 48, 48), 100, dtype=np.uint16)
        volume[:, 5, 24, 24] = 6000
        volume[:, 4, 12, 30] = 4000
        volume += rng.integers(0, 30, size=volume.shape, dtype=np.uint16)
        write_ome_tiff(directory / f"stack_{i:02d}.ome.tif", volume,
                       voxel_size_um=VOXEL, channel_names=["Alexa405", "Alexa488"])
    return directory


def test_single_stack_produces_a_calibrated_16bit_output(tmp_path, raw_dir):
    cfg = make_config(tmp_path)
    result = process_stack(cfg, raw_dir / "stack_00.ome.tif")

    assert result.status == "ok"
    assert result.output.exists()
    assert result.voxel_size_um == pytest.approx(VOXEL)

    with tifffile.TiffFile(result.output) as tif:
        series = tif.series[0]
        assert series.dtype == np.uint16
        assert tif.is_ome
        assert "PhysicalSizeZ=\"0.3\"" in tif.ome_metadata.replace("'", '"')
        data = series.asarray()
    assert data.shape == (10, 2, 48, 48)      # ZCYX


def test_stack_stays_16bit_end_to_end(tmp_path, raw_dir):
    cfg = make_config(tmp_path)
    result = process_stack(cfg, raw_dir / "stack_00.ome.tif")
    for entry in result.channels:
        assert entry["stats_after"]["max"] <= 65535
        assert entry["stats_after"]["min"] >= 0
        assert entry["stats_after"]["nan_voxels"] == 0


def test_batch_processes_every_file_and_writes_a_manifest(tmp_path, raw_dir):
    cfg = make_config(tmp_path)
    summary = run_batch(cfg)

    assert summary.n_ok == 2
    assert summary.n_failed == 0
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["config_fingerprint"] == cfg.fingerprint()
    assert len(manifest["run"]["stacks"]) == 2
    assert manifest["config"]["deconvolution"]["iterations"] == 3
    assert "numpy" in manifest["environment"]


def test_batch_applies_identical_parameters_to_every_stack(tmp_path, raw_dir):
    """The reproducibility requirement: same PSF for the same channel everywhere."""
    cfg = make_config(tmp_path)
    summary = run_batch(cfg)

    keys_per_stack = [
        [entry["psf"]["cache_key"] for entry in result.channels]
        for result in summary.results
    ]
    assert keys_per_stack[0] == keys_per_stack[1]
    # And the two channels must NOT share a PSF: different wavelengths.
    assert keys_per_stack[0][0] != keys_per_stack[0][1]


def test_rerunning_is_deterministic(tmp_path, raw_dir):
    cfg = make_config(tmp_path)
    first = process_stack(cfg, raw_dir / "stack_00.ome.tif")
    data_first = tifffile.imread(first.output)

    cfg.output.overwrite = True
    second = process_stack(cfg, raw_dir / "stack_00.ome.tif")
    np.testing.assert_array_equal(data_first, tifffile.imread(second.output))


def test_existing_output_is_skipped_then_overwritten(tmp_path, raw_dir):
    cfg = make_config(tmp_path)
    assert process_stack(cfg, raw_dir / "stack_00.ome.tif").status == "ok"

    skipped = process_stack(cfg, raw_dir / "stack_00.ome.tif")
    assert skipped.status == "skipped"

    cfg.output.overwrite = True
    assert process_stack(cfg, raw_dir / "stack_00.ome.tif").status == "ok"


def test_channel_count_mismatch_is_reported(tmp_path, raw_dir):
    cfg = make_config(tmp_path, channels=[{"name": "only_one", "emission_nm": 519}])
    with pytest.raises(ValueError, match="config declares 1 channel"):
        process_stack(cfg, raw_dir / "stack_00.ome.tif")


def test_unreadable_file_fails_the_stack_not_the_batch(tmp_path, raw_dir):
    (raw_dir / "broken.ome.tif").write_bytes(b"II*\x00not a tiff")
    cfg = make_config(tmp_path)
    summary = run_batch(cfg)

    assert summary.n_ok == 2
    assert summary.n_failed == 1
    failed = next(r for r in summary.results if r.status == "failed")
    assert "ReadError" in failed.message


def test_fail_fast_stops_the_batch(tmp_path, raw_dir):
    (raw_dir / "broken.ome.tif").write_bytes(b"II*\x00not a tiff")
    cfg = make_config(tmp_path, processing={"fail_fast": True, "max_workers": 1})
    with pytest.raises(Exception):
        run_batch(cfg)


def test_saturated_input_is_flagged(tmp_path, raw_dir):
    volume = np.full((2, 10, 48, 48), 65535, dtype=np.uint16)
    write_ome_tiff(raw_dir / "saturated.ome.tif", volume, voxel_size_um=VOXEL,
                   channel_names=["Alexa405", "Alexa488"])
    cfg = make_config(tmp_path)
    result = process_stack(cfg, raw_dir / "saturated.ome.tif")
    assert any("65535" in w for w in result.warnings)
    assert result.channels[0]["stats_before"]["saturated_fraction"] == 1.0


def test_qc_figure_is_written(tmp_path, raw_dir):
    cfg = make_config(tmp_path, qc={"enabled": True, "dpi": 50, "include_xz": True,
                                    "subdirectory": "qc"})
    result = process_stack(cfg, raw_dir / "stack_00.ome.tif")
    assert result.qc_figure is not None
    assert result.qc_figure.exists()
    assert result.qc_figure.stat().st_size > 1000


def test_intensity_scale_is_applied(tmp_path, raw_dir):
    full = process_stack(make_config(tmp_path), raw_dir / "stack_00.ome.tif")
    halved = process_stack(
        make_config(tmp_path, output={"directory": str(tmp_path / "out2"),
                                      "intensity_scale": 0.5, "save_psf": False}),
        raw_dir / "stack_00.ome.tif",
    )
    ratio = (halved.channels[0]["stats_after"]["total_intensity"]
             / full.channels[0]["stats_after"]["total_intensity"])
    assert ratio == pytest.approx(0.5, rel=0.02)


def test_discover_inputs_skips_oif_companion_directories(tmp_path):
    raw = tmp_path / "raw"
    (raw / "scan.oif.files").mkdir(parents=True)
    (raw / "scan.oif.files" / "s_C001Z001.tif").write_bytes(b"")
    (raw / "real.ome.tif").write_bytes(b"")
    cfg = make_config(tmp_path, input={"directory": str(raw), "recursive": True,
                                       "patterns": ["*.tif", "*.oif"]})
    found = [p.name for p in discover_inputs(cfg)]
    assert found == ["real.ome.tif"]


def test_background_subtraction_lowers_the_floor(tmp_path, raw_dir):
    plain = process_stack(make_config(tmp_path), raw_dir / "stack_00.ome.tif")
    subtracted = process_stack(
        make_config(
            tmp_path,
            output={"directory": str(tmp_path / "out_bg"), "save_psf": False},
            background={"method": "constant", "constant_value": 100.0},
        ),
        raw_dir / "stack_00.ome.tif",
    )
    assert (subtracted.channels[0]["stats_after"]["median"]
            < plain.channels[0]["stats_after"]["median"])
    assert subtracted.channels[0]["background_subtracted"] == 100.0


def test_config_file_round_trip_drives_the_batch(tmp_path, raw_dir):
    """The documented workflow: one YAML file, nothing else."""
    cfg = make_config(tmp_path)
    path = tmp_path / "run.yaml"
    payload = cfg.to_dict()
    payload.pop("source_path")
    path.write_text(yaml.safe_dump(payload))

    from synapse_deconv.cli import main

    assert main(["run", str(path)]) == 0
    assert len(list((tmp_path / "out").glob("*_decon.ome.tif"))) == 2


def test_depth_command_runs(tmp_path, raw_dir, capsys):
    """The decision-support command must work on a real config file."""
    cfg = make_config(tmp_path)
    path = tmp_path / "run.yaml"
    payload = cfg.to_dict()
    payload.pop("source_path")
    path.write_text(yaml.safe_dump(payload))

    from synapse_deconv.cli import main

    assert main(["depth", str(path), "--depths", "0", "10",
                 "--true-depth", "10", "--psf-size", "13"]) == 0
    out = capsys.readouterr().out
    assert "Depth sensitivity" in out
    assert "axial concentration" in out


def test_depth_command_psf_only(tmp_path, raw_dir, capsys):
    cfg = make_config(tmp_path)
    path = tmp_path / "run.yaml"
    payload = cfg.to_dict()
    payload.pop("source_path")
    path.write_text(yaml.safe_dump(payload))

    from synapse_deconv.cli import main

    assert main(["depth", str(path), "--depths", "0", "5",
                 "--psf-size", "13", "--no-restoration"]) == 0
    out = capsys.readouterr().out
    assert "FWHM axial" in out
    assert "axial concentration" not in out


def test_init_writes_a_usable_config(tmp_path, capsys):
    """'init' is the documented starting point; its output must run as-is."""
    from synapse_deconv.cli import main
    from synapse_deconv.config import load_config

    destination = tmp_path / "config" / "my_study.yaml"
    assert main(["init", str(destination)]) == 0
    assert destination.is_file()
    assert load_config(destination).optics.numerical_aperture == 1.40
    assert "Wrote" in capsys.readouterr().out


def test_init_refuses_to_overwrite_without_force(tmp_path, capsys):
    from synapse_deconv.cli import main

    destination = tmp_path / "cfg.yaml"
    assert main(["init", str(destination)]) == 0
    assert main(["init", str(destination)]) == 2
    assert main(["init", str(destination), "--force"]) == 0
