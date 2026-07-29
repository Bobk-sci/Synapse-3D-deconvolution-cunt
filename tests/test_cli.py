"""Command line overrides.

The overrides exist so that a diagnostic run never requires editing the study
configuration: the file that defines the analysis must stay byte-identical
across groups, and a copy edited "just to test something" is exactly how a
comparative study loses its reproducibility.
"""

from __future__ import annotations

import pytest

from synapse_deconv import cli, counting
from synapse_deconv.config import write_template


@pytest.fixture
def config_file(tmp_path):
    return write_template(tmp_path / "study.yaml")


def _captured_config(monkeypatch, argv):
    """Run the CLI with the counting itself stubbed out, return the config."""
    seen = {}

    def fake_run_counting(cfg, single_file=None):
        seen["cfg"] = cfg
        return []

    monkeypatch.setattr(counting, "run_counting", fake_run_counting)
    assert cli.main(argv) == 0
    return seen["cfg"]


def test_role_overrides_are_applied(monkeypatch, config_file):
    cfg = _captured_config(monkeypatch, [
        "count", str(config_file),
        "--presynaptic", "PSD95",
        "--postsynaptic-excitatory", "Bassoon",
        "--postsynaptic-inhibitory", "Gephyrin",
    ])
    assert cfg.colocalization.presynaptic == "PSD95"
    assert cfg.colocalization.postsynaptic_excitatory == "Bassoon"
    assert cfg.colocalization.postsynaptic_inhibitory == "Gephyrin"


def test_roles_are_untouched_without_the_flags(monkeypatch, config_file):
    from synapse_deconv.config import load_config

    reference = load_config(config_file)
    cfg = _captured_config(monkeypatch, ["count", str(config_file)])
    assert cfg.colocalization.presynaptic == reference.colocalization.presynaptic


def test_an_unknown_role_is_rejected_before_any_file_is_read(config_file, capsys):
    """Validation runs on the overridden config, not on the file as written."""
    assert cli.main(["count", str(config_file), "--presynaptic", "Alexa405"]) != 0
    assert "not one of the declared channels" in capsys.readouterr().err


def test_numeric_overrides_are_applied(monkeypatch, config_file):
    cfg = _captured_config(monkeypatch, [
        "count", str(config_file), "--threshold-sigma", "3", "--tolerance-um", "0.4",
    ])
    assert cfg.detection.threshold_sigma == pytest.approx(3.0)
    assert cfg.colocalization.tolerance_um == pytest.approx(0.4)


def test_overrides_change_the_fingerprint(monkeypatch, config_file):
    """A diagnostic run must not be mistaken for the study run in the manifest."""
    base = _captured_config(monkeypatch, ["count", str(config_file)])
    swapped = _captured_config(monkeypatch, [
        "count", str(config_file), "--threshold-sigma", "3",
    ])
    assert base.fingerprint() != swapped.fingerprint()
