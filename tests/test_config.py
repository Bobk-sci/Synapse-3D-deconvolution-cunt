"""Configuration parsing, validation and reproducibility fingerprint."""

from __future__ import annotations

import json

import pytest
import yaml

from synapse_deconv.config import Config, ConfigError, load_config

MINIMAL = {
    "channels": [
        {"name": "Alexa488", "emission_nm": 519, "excitation_nm": 488},
    ]
}


def write(tmp_path, data, name="config.yaml"):
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(data))
    else:
        path.write_text(yaml.safe_dump(data))
    return path


def test_defaults_match_the_documented_optics():
    cfg = Config.from_dict(MINIMAL)
    assert cfg.optics.numerical_aperture == 1.40
    assert cfg.optics.immersion_ri == 1.515
    assert cfg.metadata.fallback_xy_um == 0.095
    assert cfg.metadata.fallback_z_um == 0.30
    assert cfg.deconvolution.iterations == 25
    assert cfg.psf.model == "gibson_lanni"


def test_loads_yaml_and_json(tmp_path):
    from_yaml = load_config(write(tmp_path, MINIMAL, "c.yaml"))
    from_json = load_config(write(tmp_path, MINIMAL, "c.json"))
    assert from_yaml.fingerprint() == from_json.fingerprint()


def test_shipped_default_config_is_valid():
    cfg = load_config("config/default.yaml")
    assert len(cfg.channels) == 3
    assert [c.emission_nm for c in cfg.channels] == [421, 519, 617]


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({**MINIMAL, "optics": {"numerical_apperture": 1.4}})


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level section"):
        Config.from_dict({**MINIMAL, "deconvolutions": {}})


def test_channels_are_required():
    with pytest.raises(ConfigError, match="at least one channel"):
        Config.from_dict({})


def test_na_cannot_exceed_the_immersion_index():
    with pytest.raises(ConfigError, match="cannot exceed"):
        Config.from_dict({**MINIMAL, "optics": {"numerical_aperture": 1.6,
                                                "immersion_ri": 1.515}})


def test_confocal_mode_requires_excitation():
    data = {"channels": [{"name": "c", "emission_nm": 519}], "psf": {"mode": "confocal"}}
    with pytest.raises(ConfigError, match="requires excitation_nm"):
        Config.from_dict(data)


def test_even_psf_size_is_rejected():
    with pytest.raises(ConfigError, match="odd integer"):
        Config.from_dict({**MINIMAL, "psf": {"xy_size": 32}})


def test_duplicate_channel_names_rejected():
    data = {"channels": [{"name": "a", "emission_nm": 500},
                         {"name": "a", "emission_nm": 600}]}
    with pytest.raises(ConfigError, match="duplicate channel name"):
        Config.from_dict(data)


@pytest.mark.parametrize("section,payload,message", [
    ("deconvolution", {"iterations": 0}, "iterations must be"),
    ("deconvolution", {"backend": "opencl"}, "backend must be"),
    ("deconvolution", {"dtype": "float16"}, "dtype must be"),
    ("deconvolution", {"algorithm": "wiener"}, "only 'richardson_lucy'"),
    ("background", {"method": "rolling_ball"}, "background.method must be"),
    ("output", {"bit_depth_policy": "eight_bit"}, "bit_depth_policy must be"),
    ("output", {"intensity_scale": 0}, "intensity_scale must be"),
    ("metadata", {"fallback_z_um": 0}, "fallback voxel sizes must be"),
    ("metadata", {"spacing_divisor": "n_plus_1"}, "spacing_divisor must be"),
    ("psf", {"model": "richards_wolf"}, "psf.model must be"),
])
def test_invalid_values_are_rejected(section, payload, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict({**MINIMAL, section: payload})


def test_fingerprint_changes_with_a_numeric_parameter():
    base = Config.from_dict(MINIMAL)
    changed = Config.from_dict({**MINIMAL, "deconvolution": {"iterations": 30}})
    assert base.fingerprint() != changed.fingerprint()


def test_fingerprint_ignores_paths_and_qc():
    """Moving the output folder must not look like a different analysis."""
    base = Config.from_dict(MINIMAL)
    moved = Config.from_dict({
        **MINIMAL,
        "output": {"directory": "elsewhere"},
        "input": {"directory": "other_input"},
        "qc": {"dpi": 300},
        "logging": {"level": "DEBUG"},
    })
    assert base.fingerprint() == moved.fingerprint()


def test_fingerprint_tracks_intensity_scale():
    """intensity_scale lives under 'output' but does change pixel values."""
    base = Config.from_dict(MINIMAL)
    scaled = Config.from_dict({**MINIMAL, "output": {"intensity_scale": 0.5}})
    assert base.fingerprint() != scaled.fingerprint()


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("channels: [\n  - name: x\n")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(path)
