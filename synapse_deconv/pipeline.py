"""Per-stack processing and batch orchestration.

One config drives every stack: the same optical parameters, the same PSFs, the
same iteration count. Nothing is estimated per image unless the config asks for
it explicitly (``background.method: percentile``), because a multi-group
comparison is only valid if the transformation applied to every image is
identical.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ChannelConfig, Config
from .deconvolution import DeconvolutionResult, richardson_lucy
from .psf import PSFResult, compute_psf, measure_fwhm, theoretical_resolution
from .qc import ChannelStats, compute_stats, detect_saturation_level, write_qc_figure
from .readers import ImageStack, ReadError, is_supported, read_stack
from .writers import to_uint16, write_ome_tiff

logger = logging.getLogger(__name__)

__all__ = ["StackResult", "BatchSummary", "process_stack", "run_batch", "discover_inputs"]

_UINT16_MAX = 65535


@dataclass
class StackResult:
    """Outcome of processing a single stack."""

    source: Path
    status: str                            # "ok" | "failed" | "skipped"
    output: Path | None = None
    qc_figure: Path | None = None
    message: str = ""
    duration_s: float = 0.0
    voxel_size_um: tuple[float, float, float] | None = None
    voxel_size_source: str | None = None
    channels: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": str(self.source),
            "status": self.status,
            "output": str(self.output) if self.output else None,
            "qc_figure": str(self.qc_figure) if self.qc_figure else None,
            "message": self.message,
            "duration_s": round(self.duration_s, 2),
            "voxel_size_um": list(self.voxel_size_um) if self.voxel_size_um else None,
            "voxel_size_source": self.voxel_size_source,
            "channels": self.channels,
            "warnings": self.warnings,
        }


@dataclass
class BatchSummary:
    """Aggregate result of a batch run."""

    results: list[StackResult] = field(default_factory=list)
    config_fingerprint: str = ""
    started_at: str = ""
    duration_s: float = 0.0

    @property
    def n_ok(self) -> int:
        return sum(r.status == "ok" for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(r.status == "failed" for r in self.results)

    @property
    def n_skipped(self) -> int:
        return sum(r.status == "skipped" for r in self.results)

    def as_dict(self) -> dict:
        return {
            "config_fingerprint": self.config_fingerprint,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 2),
            "counts": {"ok": self.n_ok, "failed": self.n_failed, "skipped": self.n_skipped},
            "stacks": [r.as_dict() for r in self.results],
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def discover_inputs(cfg: Config) -> list[Path]:
    """List the input files matching the configured patterns, deduplicated."""
    root = Path(cfg.input.directory)
    if not root.is_dir():
        raise FileNotFoundError(f"input directory not found: {root}")

    found: set[Path] = set()
    for pattern in cfg.input.patterns:
        globber = root.rglob if cfg.input.recursive else root.glob
        for candidate in globber(pattern):
            if candidate.is_file() and is_supported(candidate):
                found.add(candidate.resolve())

    # An .oif ships with a companion .files directory of TIFFs; those TIFFs must
    # not be picked up as standalone inputs.
    oif_dirs = {p.with_suffix("").name + ".files" for p in found if p.suffix.lower() == ".oif"}
    filtered = [
        p for p in found
        if not any(part in oif_dirs or part.endswith(".files") for part in p.parts)
    ]
    return sorted(filtered)


def _resolve_channels(cfg: Config, stack: ImageStack) -> list[ChannelConfig]:
    """Match configured channels to the stack, honouring metadata wavelengths.

    The config is authoritative for the channel count. Emission/excitation
    wavelengths found in the file are used only to fill gaps and to warn about
    mismatches -- silently switching wavelengths per file would break the
    "identical parameters for every image" requirement.
    """
    n = stack.n_channels
    if len(cfg.channels) != n:
        raise ValueError(
            f"config declares {len(cfg.channels)} channel(s) but {stack.path.name} "
            f"has {n}; fix the 'channels' section so the batch stays consistent"
        )

    resolved: list[ChannelConfig] = []
    for i, configured in enumerate(cfg.channels):
        from dataclasses import replace

        channel = replace(configured)
        if cfg.metadata.read_channel_wavelengths and i < len(stack.channels):
            found = stack.channels[i]
            if found.emission_nm and abs(found.emission_nm - channel.emission_nm) > 15:
                stack.warnings.append(
                    f"channel {i} ({channel.name}): file reports emission "
                    f"{found.emission_nm:.0f} nm but the config says "
                    f"{channel.emission_nm:.0f} nm; the config value is used"
                )
            if channel.excitation_nm is None and found.excitation_nm:
                channel = replace(channel, excitation_nm=found.excitation_nm)
        resolved.append(channel)
    return resolved


def _subtract_background(
    volume: np.ndarray, cfg: Config, channel_index: int
) -> tuple[np.ndarray, float]:
    """Apply the configured background subtraction. Returns (volume, level)."""
    method = cfg.background.method
    if method == "none":
        return volume, 0.0
    if method == "constant":
        value = cfg.background.constant_value
        level = float(value[channel_index] if isinstance(value, list) else value)
    else:
        level = float(np.percentile(volume, cfg.background.percentile))
    if level <= 0:
        return volume, 0.0
    return np.maximum(volume - level, 0.0), level


def _sanitise(volume: np.ndarray, name: str, warnings: list[str]) -> np.ndarray:
    """Make a channel safe for Richardson-Lucy: finite and non-negative."""
    data = np.asarray(volume, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        n_bad = int((~np.isfinite(data)).sum())
        warnings.append(f"{name}: {n_bad} non-finite voxel(s) replaced by 0 before deconvolution")
        data = np.nan_to_num(data, nan=0.0, posinf=float(_UINT16_MAX), neginf=0.0)
    if data.min() < 0:
        warnings.append(f"{name}: negative voxels clipped to 0 before deconvolution")
        np.maximum(data, 0.0, out=data)
    return data


def _check_sampling(cfg: Config, channel: ChannelConfig, voxel, warnings: list[str]) -> None:
    """Warn when the voxel grid undersamples the optical resolution."""
    dz, dy, dx = voxel
    lateral, axial = theoretical_resolution(
        channel.emission_nm, cfg.optics.numerical_aperture, cfg.optics.immersion_ri
    )
    # Nyquist: at least two samples across the resolution element.
    if dx > lateral / 2 or dy > lateral / 2:
        warnings.append(
            f"{channel.name}: XY sampling {dx:.4f} um is coarser than Nyquist "
            f"({lateral / 2:.4f} um for {channel.emission_nm:.0f} nm); deconvolution "
            "cannot recover detail that was not sampled"
        )
    if dz > axial / 2:
        warnings.append(
            f"{channel.name}: Z step {dz:.3f} um is coarser than Nyquist "
            f"({axial / 2:.3f} um); axial restoration will be limited"
        )


def _output_path(cfg: Config, source: Path) -> Path:
    stem = source.name
    for suffix in (".ome.tif", ".ome.tiff", ".oib", ".oif", ".tif", ".tiff"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return Path(cfg.output.directory) / f"{stem}{cfg.output.suffix}.ome.tif"


# --------------------------------------------------------------------------
# single stack
# --------------------------------------------------------------------------

def process_stack(cfg: Config, source: Path, psf_cache: dict | None = None) -> StackResult:
    """Read, deconvolve, write and QC a single stack.

    ``psf_cache`` is an optional dict reused across stacks so that identical
    (channel, voxel size) combinations compute their PSF only once.
    """
    started = time.perf_counter()
    result = StackResult(source=source, status="failed")
    psf_cache = psf_cache if psf_cache is not None else {}

    output_path = _output_path(cfg, source)
    if output_path.exists() and not cfg.output.overwrite:
        result.status = "skipped"
        result.output = output_path
        result.message = "output already exists (set output.overwrite: true to redo it)"
        result.duration_s = time.perf_counter() - started
        logger.info("SKIP %s -- %s", source.name, result.message)
        return result

    stack = read_stack(
        source,
        voxel_size_source=cfg.metadata.voxel_size_source,
        fallback_xy_um=cfg.metadata.fallback_xy_um,
        fallback_z_um=cfg.metadata.fallback_z_um,
        spacing_divisor=cfg.metadata.spacing_divisor,
        tolerance_warn_ratio=cfg.metadata.tolerance_warn_ratio,
    )
    logger.info(stack.describe())

    channels = _resolve_channels(cfg, stack)
    voxel = stack.voxel_size_um
    result.voxel_size_um = voxel
    result.voxel_size_source = stack.voxel_size_source

    raw = stack.data
    deconvolved = np.empty(raw.shape, dtype=np.uint16)
    stats_before: list[ChannelStats] = []
    stats_after: list[ChannelStats] = []

    for index, channel in enumerate(channels):
        _check_sampling(cfg, channel, voxel, stack.warnings)

        before = compute_stats(raw[index])
        stats_before.append(before)
        full_scale = detect_saturation_level(raw[index])
        if full_scale is not None and before.saturated_fraction > 0.0001:
            bits = int(round(np.log2(full_scale + 1)))
            stack.warnings.append(
                f"{channel.name}: {before.saturated_voxels} voxel(s) "
                f"({100 * before.saturated_fraction:.3f}%) at {full_scale}, the full scale of "
                f"a {bits}-bit acquisition. Saturation violates the Poisson model; "
                "Richardson-Lucy redistributes intensity around clipped voxels, so peak "
                "intensities there are not quantitative. Lower the PMT gain or the laser "
                "power and re-acquire if those structures matter"
            )

        key = (channel.name, channel.emission_nm, channel.excitation_nm, voxel)
        if key not in psf_cache:
            psf_cache[key] = compute_psf(
                voxel_size_um=voxel,
                emission_nm=channel.emission_nm,
                excitation_nm=channel.excitation_nm,
                optics=cfg.optics,
                psf_cfg=cfg.psf,
                stack_shape=stack.spatial_shape,
            )
        psf: PSFResult = psf_cache[key]
        fwhm = measure_fwhm(psf.data, voxel)

        volume = _sanitise(raw[index], channel.name, stack.warnings)
        volume, background_level = _subtract_background(volume, cfg, index)

        deconv: DeconvolutionResult = richardson_lucy(
            volume,
            psf.data,
            iterations=cfg.deconvolution.iterations,
            backend=cfg.deconvolution.backend,
            dtype=cfg.deconvolution.dtype,
            pad_mode=cfg.deconvolution.pad_mode,
            epsilon=cfg.deconvolution.epsilon,
            regularization=cfg.deconvolution.regularization,
            tv_lambda=cfg.deconvolution.tv_lambda,
        )

        converted, conversion = to_uint16(
            deconv.data, cfg.output.bit_depth_policy, cfg.output.intensity_scale
        )
        deconvolved[index] = converted
        after = compute_stats(converted)
        stats_after.append(after)

        result.channels.append({
            "index": index,
            "name": channel.name,
            "emission_nm": channel.emission_nm,
            "excitation_nm": channel.excitation_nm,
            "psf": {
                "shape": list(psf.shape),
                "cache_key": psf.cache_key,
                "fwhm_x_um": round(fwhm["fwhm_x_um"], 4),
                "fwhm_z_um": round(fwhm["fwhm_z_um"], 4),
            },
            "background_subtracted": background_level,
            "deconvolution": {
                "iterations": deconv.iterations,
                "backend": deconv.backend,
                **{k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in deconv.diagnostics.items()},
            },
            "conversion": conversion.as_dict(),
            "stats_before": before.as_dict(),
            "stats_after": after.as_dict(),
        })

        logger.info(
            "  %s (em %.0f nm): PSF %s FWHM xy=%.3f z=%.3f um | RL %d it on %s | "
            "min/max/mean %.0f/%.0f/%.1f -> %.0f/%.0f/%.1f | sharpness x%.1f",
            channel.name, channel.emission_nm, tuple(psf.shape),
            fwhm["fwhm_x_um"], fwhm["fwhm_z_um"],
            deconv.iterations, deconv.backend,
            before.min, before.max, before.mean,
            after.min, after.max, after.mean,
            after.sharpness / before.sharpness if before.sharpness else float("nan"),
        )

    channel_names = [c.name for c in channels]
    description = (
        f"Deconvolved with synapse_deconv (config fingerprint {cfg.fingerprint()}): "
        f"{cfg.psf.model} PSF, mode={cfg.psf.mode}, NA={cfg.optics.numerical_aperture}, "
        f"ni={cfg.optics.immersion_ri}, ns={cfg.optics.sample_ri}, "
        f"depth={cfg.optics.particle_depth_um} um, "
        f"Richardson-Lucy {cfg.deconvolution.iterations} iterations."
    )
    result.output = write_ome_tiff(
        output_path,
        deconvolved,
        voxel_size_um=voxel,
        channel_names=channel_names,
        description=description,
    )

    if cfg.qc.enabled:
        qc_path = Path(cfg.output.directory) / cfg.qc.subdirectory / f"{output_path.stem}_qc.png"
        result.qc_figure = write_qc_figure(
            qc_path, raw, deconvolved,
            channel_names=channel_names,
            voxel_size_um=voxel,
            title=f"{source.name} -- RL {cfg.deconvolution.iterations} it, "
                  f"{cfg.psf.model} PSF ({cfg.psf.mode})",
            stats_before=stats_before,
            stats_after=stats_after,
            percentile_clip=tuple(cfg.qc.percentile_clip),
            display_gamma=cfg.qc.display_gamma,
            include_xz=cfg.qc.include_xz,
            dpi=cfg.qc.dpi,
        )

    if cfg.output.save_psf:
        psf_dir = Path(cfg.output.directory) / "psf"
        psf_dir.mkdir(parents=True, exist_ok=True)
        for channel, entry in zip(channels, result.channels):
            psf = psf_cache[(channel.name, channel.emission_nm, channel.excitation_nm, voxel)]
            psf_file = psf_dir / f"psf_{channel.name}_{entry['psf']['cache_key']}.ome.tif"
            if not psf_file.exists():
                # Scaled to the 16-bit range: the PSF is a probability density,
                # so absolute values would underflow uint16.
                scaled = (psf.data / psf.data.max() * _UINT16_MAX).astype(np.uint16)
                write_ome_tiff(psf_file, scaled[np.newaxis], voxel_size_um=voxel,
                               channel_names=[channel.name],
                               description="Theoretical PSF, peak-normalised to 65535")

    result.status = "ok"
    result.warnings = stack.warnings
    result.duration_s = time.perf_counter() - started
    for warning in stack.warnings:
        logger.warning("  %s: %s", source.name, warning)
    logger.info("OK   %s -> %s (%.1f s)", source.name, result.output.name, result.duration_s)
    return result


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def _environment() -> dict:
    import scipy
    import tifffile

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "tifffile": tifffile.__version__,
    }
    try:
        import oiffile

        versions["oiffile"] = oiffile.__version__
    except ImportError:
        versions["oiffile"] = "not installed"
    return versions


def run_batch(cfg: Config) -> BatchSummary:
    """Process every input file described by the config."""
    started = time.perf_counter()
    summary = BatchSummary(
        config_fingerprint=cfg.fingerprint(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    inputs = discover_inputs(cfg)
    logger.info("Config fingerprint: %s", summary.config_fingerprint)
    logger.info("Found %d input file(s) in %s", len(inputs), cfg.input.directory)
    if not inputs:
        logger.warning(
            "no input matched patterns %s in %s", cfg.input.patterns, cfg.input.directory
        )

    psf_cache: dict = {}
    for path in inputs:
        try:
            summary.results.append(process_stack(cfg, path, psf_cache))
        except (ReadError, ValueError, RuntimeError, OSError, MemoryError) as exc:
            logger.error("FAIL %s: %s", path.name, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            summary.results.append(
                StackResult(source=path, status="failed", message=f"{type(exc).__name__}: {exc}")
            )
            if cfg.processing.fail_fast:
                raise

    summary.duration_s = time.perf_counter() - started
    _write_manifest(cfg, summary)
    logger.info(
        "Batch finished in %.1f s: %d ok, %d failed, %d skipped",
        summary.duration_s, summary.n_ok, summary.n_failed, summary.n_skipped,
    )
    return summary


def _write_manifest(cfg: Config, summary: BatchSummary) -> Path:
    """Write the run manifest: full config, environment and per-stack results."""
    directory = Path(cfg.output.directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_manifest.json"
    payload = {
        "pipeline": "synapse_deconv",
        "environment": _environment(),
        "config": cfg.to_dict(),
        "config_fingerprint": summary.config_fingerprint,
        "run": summary.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Manifest written to %s", path)
    return path
