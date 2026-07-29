"""Configuration schema for the deconvolution pipeline.

A single YAML (or JSON) file drives the whole batch. Every parameter that can
influence the numerical result lives here so that a run can be reproduced from
the config file alone. :func:`Config.fingerprint` returns a hash of exactly
those parameters, which is written to the log and to the run manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ConfigError", "load_config", "template_path", "write_template"]


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
    #: A single number applied to every channel, or one number per channel in
    #: channel order. Detector pedestals usually differ a little between
    #: channels; a per-channel list keeps each one honest without making the
    #: subtraction image-dependent.
    constant_value: float | list[float] = 0.0
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
class DetectionConfig:
    """Per-channel 3D punctum detection."""

    #: Expected punctum diameters in micrometres; one LoG scale is run per
    #: entry and the best response per voxel is kept.
    punctum_diameters_um: list[float] = field(default_factory=lambda: [0.25, 0.35, 0.50])
    #: Detection threshold as a z-score above the robust (MAD) noise of the LoG
    #: response. 5 is conservative, 3 permissive. Identical for every image.
    threshold_sigma: float = 5.0
    #: White top-hat radius. Must exceed the largest punctum radius, otherwise
    #: the puncta are removed along with the background.
    background_radius_um: float = 1.0
    #: Minimum centre-to-centre distance between two detected maxima.
    min_separation_um: float = 0.25
    #: Size gate applied to the segmented regions.
    min_volume_um3: float = 0.008
    max_volume_um3: float | None = 2.0
    #: Per-channel override of threshold_sigma, keyed by channel name. Use only
    #: with a documented reason: it breaks the "one threshold for all" rule
    #: across channels (it stays identical across images, which is what matters
    #: for a group comparison).
    threshold_sigma_per_channel: dict[str, float] = field(default_factory=dict)


@dataclass
class ColocalizationConfig:
    """Turning apposed puncta into synapses."""

    #: ``distance`` (centroid apposition, the default) or ``contact``
    #: (segmented masks touching after a small dilation).
    criterion: str = "distance"
    #: Maximum 3D centre-to-centre distance, in micrometres. Euclidean in
    #: physical space, so the anisotropic voxel is already accounted for.
    tolerance_um: float = 0.30
    #: Dilation applied to the presynaptic masks in ``contact`` mode.
    contact_dilation_um: float = 0.10
    #: One punctum takes part in at most one synapse, by globally optimal
    #: assignment. Turning this off inflates counts in dense fields.
    one_to_one: bool = True
    #: A presynaptic punctum matching both post-synaptic markers is assigned to
    #: the closer one instead of being counted twice.
    resolve_ambiguous_partners: bool = True
    #: Measure the systematic displacement between channels (chromatic
    #: aberration, detector misalignment) before pairing. A shift comparable to
    #: the tolerance moves every true pair outside it at once.
    measure_channel_offset: bool = True
    #: Subtract the measured shift before pairing. Off by default: a real offset
    #: should be fixed with a bead calibration, not absorbed silently. Turn it on
    #: only once you have seen the measured value and decided it is instrumental.
    correct_channel_offset: bool = False
    #: Warn when the measured shift exceeds this fraction of tolerance_um.
    offset_warn_fraction: float = 0.5
    #: Randomisation control: re-pair after randomly translating one channel, to
    #: measure how many pairs the tolerance yields by chance at this density.
    #: Reported alongside the observed count; 0 disables it.
    chance_randomisations: int = 10
    #: Channel roles, by channel name as declared in 'channels'. Left unset by
    #: default so that a deconvolution-only config need not name synaptic
    #: markers; the 'count' command requires them.
    presynaptic: str = ""
    postsynaptic_excitatory: str = ""
    postsynaptic_inhibitory: str = ""


@dataclass
class AnalysisConfig:
    """ROI restriction, density normalisation and sanity checks."""

    #: Optional 3D mask (any readable stack, non-zero = inside). ``null`` uses
    #: the whole field.
    roi_mask: str | None = None
    #: Voxels this far from the stack border are excluded: a punctum whose
    #: neighbourhood is truncated cannot be measured or paired reliably.
    border_exclusion_um: float = 0.5
    #: Densities are reported per this volume.
    density_unit_um3: float = 100.0
    #: Flag an image whose punctum count per channel falls outside this range.
    expected_puncta_per_100um3: list[float] = field(default_factory=lambda: [0.5, 500.0])
    #: Flag an image whose detected puncta are barely above the noise.
    min_snr_warn: float = 3.0
    #: Where the CSVs go, relative to output.directory.
    subdirectory: str = "counts"


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
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    colocalization: ColocalizationConfig = field(default_factory=ColocalizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
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
            "detection": DetectionConfig,
            "colocalization": ColocalizationConfig,
            "analysis": AnalysisConfig,
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
        value = self.background.constant_value
        if isinstance(value, list):
            if len(value) != len(self.channels):
                raise ConfigError(
                    f"background.constant_value has {len(value)} entries but "
                    f"{len(self.channels)} channel(s) are declared; give one value per "
                    "channel in the same order, or a single number for all of them"
                )
            if any(not isinstance(v, (int, float)) or v < 0 for v in value):
                raise ConfigError("background.constant_value entries must be numbers >= 0")
        elif not isinstance(value, (int, float)) or value < 0:
            raise ConfigError("background.constant_value must be a number >= 0")

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

        self._validate_counting()

    def _validate_counting(self) -> None:
        """Detection / colocalisation / analysis sections."""
        det = self.detection
        if not det.punctum_diameters_um:
            raise ConfigError("detection.punctum_diameters_um must list at least one diameter")
        if any(d <= 0 for d in det.punctum_diameters_um):
            raise ConfigError("detection.punctum_diameters_um entries must be > 0")
        if det.threshold_sigma <= 0:
            raise ConfigError("detection.threshold_sigma must be > 0")
        if det.background_radius_um <= max(det.punctum_diameters_um) / 2:
            raise ConfigError(
                f"detection.background_radius_um ({det.background_radius_um}) must exceed the "
                f"largest punctum radius ({max(det.punctum_diameters_um) / 2}); otherwise the "
                "top-hat removes the puncta together with the background"
            )
        if det.min_separation_um <= 0:
            raise ConfigError("detection.min_separation_um must be > 0")
        if det.min_volume_um3 < 0:
            raise ConfigError("detection.min_volume_um3 must be >= 0")
        if det.max_volume_um3 is not None and det.max_volume_um3 <= det.min_volume_um3:
            raise ConfigError("detection.max_volume_um3 must exceed min_volume_um3")
        known = {c.name for c in self.channels}
        unknown = set(det.threshold_sigma_per_channel) - known
        if unknown:
            raise ConfigError(
                f"detection.threshold_sigma_per_channel names unknown channel(s) "
                f"{sorted(unknown)}; declared channels are {sorted(known)}"
            )

        coloc = self.colocalization
        if coloc.criterion not in ("distance", "contact"):
            raise ConfigError("colocalization.criterion must be 'distance' or 'contact'")
        if coloc.tolerance_um <= 0:
            raise ConfigError("colocalization.tolerance_um must be > 0")
        if coloc.contact_dilation_um < 0:
            raise ConfigError("colocalization.contact_dilation_um must be >= 0")
        if coloc.chance_randomisations < 0:
            raise ConfigError("colocalization.chance_randomisations must be >= 0")
        if not 0 < coloc.offset_warn_fraction <= 2:
            raise ConfigError("colocalization.offset_warn_fraction must be in (0, 2]")
        roles = {
            "presynaptic": coloc.presynaptic,
            "postsynaptic_excitatory": coloc.postsynaptic_excitatory,
            "postsynaptic_inhibitory": coloc.postsynaptic_inhibitory,
        }
        for role, name in roles.items():
            if name and name not in known:
                raise ConfigError(
                    f"colocalization.{role} = {name!r} is not one of the declared channels "
                    f"{sorted(known)}"
                )
        assigned = [n for n in roles.values() if n]
        if len(set(assigned)) != len(assigned):
            raise ConfigError("colocalization roles must name three different channels")

    def require_synaptic_roles(self) -> None:
        """Check the roles needed by 'count'. Not part of validate(): a config
        that only drives the deconvolution has no reason to name markers."""
        coloc = self.colocalization
        missing = [
            role for role, name in (
                ("presynaptic", coloc.presynaptic),
                ("postsynaptic_excitatory", coloc.postsynaptic_excitatory),
                ("postsynaptic_inhibitory", coloc.postsynaptic_inhibitory),
            ) if not name
        ]
        if missing:
            raise ConfigError(
                f"counting synapses needs colocalization.{', colocalization.'.join(missing)} "
                f"set to one of the declared channels {sorted(c.name for c in self.channels)}"
            )

        ana = self.analysis
        if ana.density_unit_um3 <= 0:
            raise ConfigError("analysis.density_unit_um3 must be > 0")
        if ana.border_exclusion_um < 0:
            raise ConfigError("analysis.border_exclusion_um must be >= 0")
        if len(ana.expected_puncta_per_100um3) != 2:
            raise ConfigError(
                "analysis.expected_puncta_per_100um3 must be a [min, max] pair"
            )
        low, high = ana.expected_puncta_per_100um3
        if low < 0 or high <= low:
            raise ConfigError(
                "analysis.expected_puncta_per_100um3 must satisfy 0 <= min < max"
            )

    def advisories(self) -> list[str]:
        """Non-fatal warnings about parameter combinations worth a second look.

        These are physics problems, not schema problems, so they are reported
        rather than raised: the run is still meaningful, the user just needs to
        know what they are asking for.
        """
        notes: list[str] = []
        o = self.optics

        if o.numerical_aperture > o.sample_ri:
            notes.append(
                f"NA ({o.numerical_aperture}) exceeds the sample refractive index "
                f"(sample_ri={o.sample_ri}): rays beyond the critical angle cannot "
                f"enter the mounting medium, so the effective NA is capped at "
                f"{o.sample_ri}. Expect lower resolution and strong depth-dependent "
                "spherical aberration than the objective's specification suggests."
            )
        mismatch = abs(o.immersion_ri - o.sample_ri)
        if mismatch > 0.03 and o.particle_depth_um >= 5.0:
            notes.append(
                f"immersion ({o.immersion_ri}) and sample ({o.sample_ri}) indices differ "
                f"by {mismatch:.3f} at a depth of {o.particle_depth_um:g} um: the axial PSF "
                "is strongly aberrated. Check optics.particle_depth_um against your real "
                "imaging depth -- run 'synapse-deconv depth' to see how much it matters."
            )
        if mismatch > 0.03 and o.particle_depth_um < 1.0:
            notes.append(
                f"optics.particle_depth_um is {o.particle_depth_um:g} um while the immersion "
                f"and sample indices differ by {mismatch:.3f}. A near-zero depth models an "
                "almost aberration-free PSF; if you actually image several micrometres into "
                "the tissue, the deconvolution will recover far less axial resolution."
            )
        if self.psf.mode == "confocal" and self.psf.pinhole_airy_units > 3.0:
            notes.append(
                f"pinhole of {self.psf.pinhole_airy_units} AU is effectively widefield "
                "detection; psf.mode='emission' would be equivalent and cheaper."
            )
        return notes

    # -- reproducibility -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def numeric_parameters(self) -> dict[str, Any]:
        """Config subset that can change pixel values, used for the fingerprint."""
        d = self.to_dict()
        output = d.pop("output", {})
        analysis = d.pop("analysis", {})
        for volatile in ("input", "qc", "logging", "processing", "source_path"):
            d.pop(volatile, None)
        # Where the CSVs land does not change any number; everything else in
        # the analysis section does.
        analysis.pop("subdirectory", None)
        d["analysis"] = analysis
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


def template_path() -> Path:
    """Path of the reference configuration shipped inside the package.

    It lives in the package rather than in the repository so that it stays
    pristine: it is the file :func:`write_template` copies from, and the one the
    test suite validates. Users edit their own copy, never this one.
    """
    return Path(__file__).parent / "templates" / "default.yaml"


def write_template(destination: str | Path, *, overwrite: bool = False) -> Path:
    """Copy the reference configuration to ``destination`` for the user to edit."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise ConfigError(
            f"{destination} already exists; pass --force to replace it "
            "(this would discard your settings)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template_path().read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def _windows_path_hint(text: str) -> str | None:
    """Explain the double-quoted Windows path trap, if that is what happened.

    In YAML a double-quoted scalar processes backslash escapes, so
    ``"C:\\Users\\..."`` is read as an invalid ``\\U`` unicode escape. Single
    quotes and bare scalars do not, and forward slashes work on Windows too.
    PyYAML's own message ("expected escape sequence of 8 hexadecimal numbers")
    gives no hint of any of this.
    """
    offender = re.search(r'^\s*(\w+)\s*:\s*"([A-Za-z]:\\|\\\\)[^"]*"', text, re.MULTILINE)
    if offender is None:
        return None
    key = offender.group(1)
    return (
        f"the value of '{key}' looks like a Windows path in DOUBLE quotes. YAML reads "
        "backslashes in double-quoted text as escape codes, so C:\\Users becomes an "
        "invalid \\U escape. Use single quotes, no quotes, or forward slashes:\n"
        f"    {key}: 'C:\\Users\\you\\data'\n"
        f"    {key}: C:/Users/you/data"
    )


def _nearby_configs(path: Path) -> str:
    """List the config files sitting next to a path that does not exist.

    A mistyped file name is otherwise a dead end: the user has to go and look up
    what they actually called it.
    """
    directories = [d for d in (path.parent, Path("config"), Path(".")) if d.is_dir()]
    candidates: list[str] = []
    for directory in directories:
        for entry in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.json")):
            text = str(entry)
            if text not in candidates:
                candidates.append(text)
        if candidates:
            break
    if not candidates:
        return ""
    listed = "\n  ".join(candidates[:10])
    return f"\n\nConfiguration files found nearby:\n  {listed}"


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML or JSON configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}{_nearby_configs(path)}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        hint = _windows_path_hint(text)
        message = f"{path}: could not parse configuration"
        if hint:
            message += f".\n\n{hint}\n\nParser said: {exc}"
        else:
            message += f" ({exc})"
        raise ConfigError(message) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    data["source_path"] = str(path.resolve())
    return Config.from_dict(data)
