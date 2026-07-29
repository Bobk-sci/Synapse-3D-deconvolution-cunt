"""End-to-end synapse counting: CSV contents, densities, ROI and sanity checks."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from synapse_deconv.config import Config, ConfigError
from synapse_deconv.counting import count_stack, run_counting
from synapse_deconv.writers import write_ome_tiff

VOXEL = (0.30, 0.095, 0.095)
SHAPE_UM = (6.0, 14.0, 14.0)


def make_stack(tmp_path, n_exc=12, n_inh=6, n_orphan=8, seed=0, name="stack_00.ome.tif"):
    """Triple-labelled stack with a known number of apposed pairs."""
    rng = np.random.default_rng(seed)
    dz, dy, dx = VOXEL
    shape = (int(SHAPE_UM[0] / dz), int(SHAPE_UM[1] / dy), int(SHAPE_UM[2] / dx))
    volumes = [np.zeros(shape, dtype=np.float64) for _ in range(3)]
    zz = np.arange(shape[0])[:, None, None] * dz
    yy = np.arange(shape[1])[None, :, None] * dy
    xx = np.arange(shape[2])[None, None, :] * dx

    def blob(volume, centre, amplitude=2500.0, sigma=0.11):
        volume += amplitude * np.exp(
            -((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
            / (2 * sigma ** 2)
        )

    def position():
        return np.array([rng.uniform(1.5, SHAPE_UM[0] - 1.5),
                         rng.uniform(2.0, SHAPE_UM[1] - 2.0),
                         rng.uniform(2.0, SHAPE_UM[2] - 2.0)])

    for kind, channel, n in (("exc", 1, n_exc), ("inh", 2, n_inh)):
        for _ in range(n):
            centre = position()
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            blob(volumes[0], centre - direction * 0.075)
            blob(volumes[channel], centre + direction * 0.075)
    for _ in range(n_orphan):
        blob(volumes[0], position())

    stacked = np.stack(volumes)
    noisy = rng.poisson(stacked) + 20 + rng.normal(0, 5, size=stacked.shape)
    data = np.clip(noisy, 0, 65535).astype(np.uint16)
    path = tmp_path / name
    write_ome_tiff(path, data, voxel_size_um=VOXEL,
                   channel_names=["Bassoon", "PSD95", "Gephyrin"])
    return path


def make_config(tmp_path, **overrides):
    data = {
        "input": {"directory": str(tmp_path)},
        "output": {"directory": str(tmp_path / "out")},
        "channels": [{"name": "Bassoon", "emission_nm": 519},
                     {"name": "PSD95", "emission_nm": 617},
                     {"name": "Gephyrin", "emission_nm": 421}],
        "colocalization": {"presynaptic": "Bassoon",
                           "postsynaptic_excitatory": "PSD95",
                           "postsynaptic_inhibitory": "Gephyrin",
                           "chance_randomisations": 3},
        "qc": {"enabled": False},
        "logging": {"file": ""},
    }
    for section, payload in overrides.items():
        if isinstance(payload, dict) and isinstance(data.get(section), dict):
            data[section] = {**data[section], **payload}
        else:
            data[section] = payload
    return Config.from_dict(data)


def test_counts_are_close_to_the_truth(tmp_path):
    path = make_stack(tmp_path, n_exc=12, n_inh=6)
    result = count_stack(make_config(tmp_path), path)
    assert result.status == "ok"
    assert result.synapses["n_excitatory"] == pytest.approx(12, abs=3)
    assert result.synapses["n_inhibitory"] == pytest.approx(6, abs=2)


def test_density_uses_the_analysed_volume_not_the_field(tmp_path):
    path = make_stack(tmp_path)
    cfg = make_config(tmp_path, analysis={"border_exclusion_um": 1.0})
    result = count_stack(cfg, path)
    full_volume = SHAPE_UM[0] * SHAPE_UM[1] * SHAPE_UM[2]
    assert result.analysed_volume_um3 < full_volume
    row = result.summary_row()
    expected = result.synapses["n_excitatory"] / result.analysed_volume_um3 * 100
    assert row["density_excitatory"] == pytest.approx(expected, rel=1e-3)


def test_border_exclusion_shrinks_the_analysed_volume(tmp_path):
    path = make_stack(tmp_path)
    wide = count_stack(make_config(tmp_path, analysis={"border_exclusion_um": 0.0}), path)
    narrow = count_stack(make_config(tmp_path, analysis={"border_exclusion_um": 1.5}), path)
    assert narrow.analysed_volume_um3 < wide.analysed_volume_um3


def test_puncta_csv_has_physical_coordinates_and_annotations(tmp_path):
    path = make_stack(tmp_path)
    result = count_stack(make_config(tmp_path), path)
    rows = result.puncta_rows
    assert rows
    assert {r["channel"] for r in rows} == {"Bassoon", "PSD95", "Gephyrin"}
    for row in rows:
        assert 0 <= row["z_um"] <= SHAPE_UM[0]
        assert 0 <= row["y_um"] <= SHAPE_UM[1]
        assert row["volume_um3"] > 0
        assert row["synapse_type"] in ("none", "excitatory", "inhibitory")
    assert any(r["synapse_type"] == "excitatory" for r in rows)


def test_chance_control_is_reported(tmp_path):
    path = make_stack(tmp_path)
    result = count_stack(make_config(tmp_path), path)
    chance = result.synapses["chance"]
    assert "excitatory" in chance
    assert chance["excitatory"]["n_randomisations"] == 3
    # Real pairs vastly outnumber accidental ones at this density.
    assert chance["excitatory"]["chance_mean"] < result.synapses["n_excitatory"]


def test_control_pair_is_measured_in_every_run(tmp_path):
    """PSD-95 vs Gephyrin is on different synapses: it is the empirical null.

    Measuring it used to take a second run with the roles permuted, which meant
    nobody measured it. The statistical behaviour of the ratio is covered in
    test_colocalization; what matters here is that the pair is profiled at all.
    """
    path = make_stack(tmp_path, n_exc=14, n_inh=12, n_orphan=10)
    result = count_stack(make_config(tmp_path), path)
    profiles = result.synapses["apposition_profile"]
    assert profiles["control"]["observed"], "the control pair must be profiled"
    assert "specific_enrichment" in profiles["excitatory"]
    assert "specific_enrichment" in profiles["inhibitory"]


def test_specific_enrichment_has_a_summary_column(tmp_path):
    path = make_stack(tmp_path, n_exc=14, n_inh=12, n_orphan=10)
    row = count_stack(make_config(tmp_path), path).summary_row()
    assert "specific_enrichment_excitatory" in row
    assert "specific_enrichment_inhibitory" in row


def test_no_puncta_is_flagged_not_silent(tmp_path):
    empty = np.full((3, 20, 100, 100), 20, dtype=np.uint16)
    path = tmp_path / "empty.ome.tif"
    write_ome_tiff(path, empty, voxel_size_um=VOXEL,
                   channel_names=["Bassoon", "PSD95", "Gephyrin"])
    result = count_stack(make_config(tmp_path), path)
    assert result.synapses["n_excitatory"] == 0
    assert any("no punctum detected" in w for w in result.warnings)


def test_batch_writes_csvs_and_manifest(tmp_path):
    make_stack(tmp_path, seed=0, name="stack_00.ome.tif")
    make_stack(tmp_path, seed=1, name="stack_01.ome.tif")
    cfg = make_config(tmp_path, input={"directory": str(tmp_path),
                                       "patterns": ["stack_*.ome.tif"]})
    results = run_counting(cfg)
    assert len(results) == 2

    out = tmp_path / "out" / "counts"
    summary = list(csv.DictReader((out / "summary.csv").open()))
    assert len(summary) == 2
    assert {"density_excitatory", "density_inhibitory", "analysed_volume_um3"} <= set(summary[0])
    for stem in ("stack_00.ome", "stack_01.ome"):
        assert (out / f"{stem}_puncta.csv").is_file()

    manifest = json.loads((out / "counting_manifest.json").read_text())
    assert manifest["config_fingerprint"] == cfg.fingerprint()
    assert len(manifest["stacks"]) == 2


def test_counting_is_deterministic(tmp_path):
    path = make_stack(tmp_path)
    cfg = make_config(tmp_path)
    first = count_stack(cfg, path)
    second = count_stack(cfg, path)
    assert first.synapses["n_excitatory"] == second.synapses["n_excitatory"]
    assert first.puncta_rows == second.puncta_rows


def test_channel_roles_must_name_declared_channels(tmp_path):
    with pytest.raises(ConfigError, match="is not one of the declared channels"):
        make_config(tmp_path, colocalization={"presynaptic": "Synaptophysin"})


def test_background_radius_must_exceed_the_punctum_radius(tmp_path):
    with pytest.raises(ConfigError, match="must exceed the largest punctum radius"):
        make_config(tmp_path, detection={"punctum_diameters_um": [3.0],
                                         "background_radius_um": 1.0})
