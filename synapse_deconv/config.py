"""Configuration schema for the deconvolution pipeline.

A single YAML (or JSON) file drives the whole batch. Every parameter that can
influence the numerical result lives here so that a run can be reproduced from
the config file alone. :func:`Config.fingerprint` returns a hash of exactly
those parameters, which is written to the log and to the run manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ConfigError", "load_config"]


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or inconsistent."""


def _simple_build(cls: type, data: Any, path: str):
    """Instantiate a flat dataclass from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}. Valid keys: {sorted(known)}"
        )
    return cls(**data)


@dataclass
class InputConfig:
    directory: str = "data/raw"
    patterns: list[str] = field(
        default_factory=lambda: ["*.oib", "*.oif", "*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff"]
    )
    recursive: bool = False


@dataclass
class OutputConfig:
    directory: str = "results"
    suffix: str = "_decon"
    overwrite: bool = False
    #: ``clip`` keeps absolute photometry (values are rounded and clipped to the
    #: uint16 range). ``rescale_per_stack`` normalises each stack to full range
    #: and DESTROYS cross-image comparability -- never use it for group studies.
    bit_depth_policy: str = "clip"
    #: Constant gain applied to every deconvolved stack before the uint16 cast.
    #: Richardson-Lucy concentrates a punctum's photons into far fewer voxels, so
    #: peak values rise by an order of magnitude and can exceed 65535 even when
    #: the raw data was nowhere near saturation. A single fixed factor (< 1)
    #: applied identically to every image prevents that clipping while keeping
    #: all stacks on one common intensity scale. The log reports the factor that
    #: would have been needed if any clipping occurs.
    intensity_scale: float = 1.0
    save_psf: bool = True


@dataclass
class MetadataConfig:
    #: ``metadata`` reads the voxel size from the file and only falls back to the
    #: values below when absent. ``config`` forces the fallback values.
    voxel_size_source: str = "metadata"
    fallback_xy_um: float = 0.095
    fallback_z_um: float = 0.30
    #: Olympus stores the first and last pixel/plane *centre* positions, so the
    #: spacing is (end - start) / (n - 1). Set to ``n`` to reproduce the
    #: convention used by some other readers.
    spacing_divisor: str = "n_minus_1"
    #: Warn when the voxel size read from the file deviates from the fallback by
    #: more than this relative amount (guards against unit/parsing mistakes).
    tolerance_warn_ratio: float = 0.25
    #: Take per-channel emission wavelengths from the file when available.
    read_channel_wavelengths: bool = True


@dataclass
class OpticsConfig:
    numerical_aperture: float = 1.40
    magnification: float = 60.0
    #: Immersion medium: actual and design refractive index (oil = 1.515).
    immersion_ri: float = 1.515
    immersion_ri_design: float = 1.515
    #: Coverslip.
    coverslip_ri: float = 1.515
    coverslip_ri_design: float = 1.515
    coverslip_thickness_um: float = 170.0
    coverslip_thickness_design_um: float = 170.0
    #: Design working distance of the objective (Gibson-Lanni ``ti0``).
    working_distance_um: float = 150.0
    #: Refractive index of the mounting medium / sample.
    sample_ri: float = 1.47
    #: Depth of the imaged structure below the coverslip. Drives the amount of
    #: spherical aberration in the Gibson-Lanni model.
    particle_depth_um: float = 2.0


@dataclass
class ChannelConfig:
    name: str = "channel"
    emission_nm: float = 519.0
    excitation_nm: float | None = None


@dataclass
class PSFConfig:
    #: ``gibson_lanni`` (handles immersion/sample index mismatch) or ``born_wolf``.
    model: str = "gibson_lanni"
    #: ``emission`` = single emission PSF. ``confocal`` = h_ex * (h_em conv pinhole).
    mode: str = "emission"
    pinhole_airy_units: float = 1.0
    #: Back-projected (object-space) pinhole radius. Overrides the Airy-unit
    #: calculation when set.
    pinhole_radius_um: float | None = None
    #: PSF kernel size in voxels; must be odd. ``null`` = auto from optics.
    xy_size: int | None = None
    z_size: int | None = None
    #: Sub-voxel averaging to avoid aliasing at Nyquist-limited pixel sizes.
    oversample_xy: int = 2
    oversample_z: int = 1
    #: Snap the kernel so its brightest plane is the centre plane. Off by
    #: default: the analytic focus used for sampling is wavelength-independent,
    #: so leaving it off keeps the three channels axially registered with each
    #: other, which is what colocalisation measurements rely on.
    center_on_peak: bool = False
    #: Number of quadrature points across the pupil.
    integration_samples: int = 512
    #: Radial samples used to build the profile before Cartesian interpolation.
    radial_samples: int = 1024
    cache_dir: str | None = ".psf_cache"


@dataclass
class BackgroundConfig:
    #: ``none`` (default; Richardson-Lucy is run on raw counts), ``constant``
    #: (a fixed offset, identical for every image -- safe for group studies) or
    #: ``percentile`` (per-image estimate, introduces image-dependent variation).
    method: str = "none"
    constant_value: float = 0.0
    percentile: float = 0.1


@dataclass
class DeconvolutionConfig:
    algorithm: str = "richardson_lucy"
    iterations: int = 25
    #: ``auto`` uses CuPy when importable, otherwise NumPy/SciPy.
    backend: str = "auto"
    dtype: str = "float32"
    #: Edge handling for the FFT padding: ``reflect``, ``edge`` or ``constant``.
    pad_mode: str = "reflect"
    epsilon: float = 1e-9
    #: ``none`` or ``tv`` (Richardson-Lucy with total-variation regularisation).
    regularization: str = "none"
    tv_lambda: float = 0.002


@dataclass
class QCConfig:
    enabled: bool = True
    dpi: int = 150
    #: Display stretch for the QC panels only -- never applied to saved data.
    percentile_clip: list[float] = field(default_factory=lambda: [0.1, 99.9])
    #: Gamma applied to the QC panels only. 0.5 (square root) keeps the dim
    #: puncta visible next to the few very bright ones. 1.0 = linear.
    display_gamma: float = 0.5
    include_xz: bool = True
    subdirectory: str = "qc"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "pipeline.log"


@dataclass
class ProcessingConfig:
    #: Stacks processed in parallel. Each worker holds a full float32 copy of
    #: one channel plus its FFT buffers, so raise this only with RAM to spare.
    max_workers: int = 1
    #: Stop the whole batch on the first failure instead of logging and skipping.
    fail_fast: bool = False


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    channels: list[ChannelConfig] = field(default_factory=list)
    psf: PSFConfig = field(default_factory=PSFConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    #: Absolute path of the file this config was loaded from (informational).
    source_path: str | None = None

    # -- construction ---------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = dict(data or {})
        channels_raw = data.pop("channels", None) or []
        if not isinstance(channels_raw, list):
            raise ConfigError("channels: expected a list of channel mappings")

        sections: dict[str, Any] = {}
        section_types = {
            "input": InputConfig,
            "output": OutputConfig,
            "metadata": MetadataConfig,
            "optics": OpticsConfig,
            "psf": PSFConfig,
            "background": BackgroundConfig,
            "deconvolution": DeconvolutionConfig,
            "qc": QCConfig,
            "logging": LoggingConfig,
            "processing": ProcessingConfig,
        }
        source_path = data.pop("source_path", None)
        unknown = set(data) - set(section_types)
        if unknown:
            raise ConfigError(
                f"unknown top-level section(s) {sorted(unknown)}; "
                f"valid sections: {sorted(section_types) + ['channels']}"
            )
        for key, sub_cls in section_types.items():
            sections[key] = _simple_build(sub_cls, data.get(key), key)

        channels = [_simple_build(ChannelConfig, c, f"channels[{i}]")
                    for i, c in enumerate(channels_raw)]

        cfg = cls(channels=channels, source_path=source_path, **sections)
        cfg.validate()
        return cfg

    # -- validation -----------------------------------------------------

    def validate(self) -> None:
        o = self.optics
        if not 0 < o.numerical_aperture:
            raise ConfigError("optics.numerical_aperture must be > 0")
        if o.numerical_aperture > o.immersion_ri:
            raise ConfigError(
                f"optics.numerical_aperture ({o.numerical_aperture}) cannot exceed "
                f"optics.immersion_ri ({o.immersion_ri}); check the objective spec"
            )
        for name in ("immersion_ri", "immersion_ri_design", "coverslip_ri",
                     "coverslip_ri_design", "sample_ri"):
            if getattr(o, name) <= 0:
                raise ConfigError(f"optics.{name} must be > 0")
        if o.particle_depth_um < 0:
            raise ConfigError("optics.particle_depth_um must be >= 0")

        if not self.channels:
            raise ConfigError("at least one channel must be declared in 'channels'")
        seen: set[str] = set()
        for i, ch in enumerate(self.channels):
            if ch.emission_nm <= 0:
                raise ConfigError(f"channels[{i}].emission_nm must be > 0")
            if ch.name in seen:
                raise ConfigError(f"duplicate channel name {ch.name!r}")
            seen.add(ch.name)

        p = self.psf
        if p.model not in ("gibson_lanni", "born_wolf"):
            raise ConfigError("psf.model must be 'gibson_lanni' or 'born_wolf'")
        if p.mode not in ("emission", "confocal"):
            raise ConfigError("psf.mode must be 'emission' or 'confocal'")
        if p.mode == "confocal":
            missing = [c.name for c in self.channels if not c.excitation_nm]
            if missing:
                raise ConfigError(
                    "psf.mode='confocal' requires excitation_nm for every channel; "
                    f"missing for {missing}"
                )
        for attr in ("xy_size", "z_size"):
            v = getattr(p, attr)
            if v is not None and (v < 3 or v % 2 == 0):
                raise ConfigError(f"psf.{attr} must be an odd integer >= 3 (got {v})")
        for attr in ("oversample_xy", "oversample_z"):
            if getattr(p, attr) < 1:
                raise ConfigError(f"psf.{attr} must be >= 1")
        if p.integration_samples < 32:
            raise ConfigError("psf.integration_samples must be >= 32")

        d = self.deconvolution
        if d.algorithm != "richardson_lucy":
            raise ConfigError("deconvolution.algorithm: only 'richardson_lucy' is implemented")
        if d.iterations < 1:
            raise ConfigError("deconvolution.iterations must be >= 1")
        if d.backend not in ("auto", "numpy", "cupy"):
            raise ConfigError("deconvolution.backend must be 'auto', 'numpy' or 'cupy'")
        if d.dtype not in ("float32", "float64"):
            raise ConfigError("deconvolution.dtype must be 'float32' or 'float64'")
        if d.pad_mode not in ("reflect", "edge", "constant"):
            raise ConfigError("deconvolution.pad_mode must be 'reflect', 'edge' or 'constant'")
        if d.regularization not in ("none", "tv"):
            raise ConfigError("deconvolution.regularization must be 'none' or 'tv'")
        if d.regularization == "tv" and d.tv_lambda < 0:
            raise ConfigError("deconvolution.tv_lambda must be >= 0")

        if self.background.method not in ("none", "constant", "percentile"):
            raise ConfigError("background.method must be 'none', 'constant' or 'percentile'")

        if self.output.bit_depth_policy not in ("clip", "rescale_per_stack"):
            raise ConfigError(
                "output.bit_depth_policy must be 'clip' or 'rescale_per_stack'"
            )
        if self.output.intensity_scale <= 0:
            raise ConfigError("output.intensity_scale must be > 0")
        if self.metadata.voxel_size_source not in ("metadata", "config"):
            raise ConfigError("metadata.voxel_size_source must be 'metadata' or 'config'")
        if self.metadata.spacing_divisor not in ("n_minus_1", "n"):
            raise ConfigError("metadata.spacing_divisor must be 'n_minus_1' or 'n'")
        if self.metadata.fallback_xy_um <= 0 or self.metadata.fallback_z_um <= 0:
            raise ConfigError("metadata fallback voxel sizes must be > 0")
        if self.processing.max_workers < 1:
            raise ConfigError("processing.max_workers must be >= 1")

    # -- reproducibility -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def numeric_parameters(self) -> dict[str, Any]:
        """Config subset that can change pixel values, used for the fingerprint."""
        d = self.to_dict()
        output = d.pop("output", {})
        for volatile in ("input", "qc", "logging", "processing", "source_path"):
            d.pop(volatile, None)
        # Only the two output settings that alter the stored intensities count.
        d["output"] = {
            "bit_depth_policy": output.get("bit_depth_policy"),
            "intensity_scale": output.get("intensity_scale"),
        }
        return d

    def fingerprint(self) -> str:
        """Stable short hash of the numerically relevant parameters."""
        blob = json.dumps(self.numeric_parameters(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML or JSON configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: could not parse configuration ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    data["source_path"] = str(path.resolve())
    return Config.from_dict(data)
