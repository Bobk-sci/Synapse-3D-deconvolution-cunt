"""Command line entry point.

    python -m synapse_deconv run    config/default.yaml
    python -m synapse_deconv run    config/default.yaml --file one_stack.oib
    python -m synapse_deconv check  config/default.yaml
    python -m synapse_deconv psf    config/default.yaml --out psf_preview
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, ConfigError, load_config

logger = logging.getLogger("synapse_deconv")


def setup_logging(cfg: Config, log_path: Path | None = None) -> None:
    """Log to the console and to a file inside the output directory."""
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    root = logging.getLogger("synapse_deconv")
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(console)

    if cfg.logging.file:
        path = log_path or Path(cfg.output.directory) / cfg.logging.file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
            )
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("could not open log file %s (%s); logging to console only", path, exc)


def _log_parameters(cfg: Config) -> None:
    """Write every parameter that affects the result into the log."""
    o, p, d = cfg.optics, cfg.psf, cfg.deconvolution
    logger.info("=" * 78)
    logger.info("synapse_deconv -- 3D deconvolution of confocal z-stacks")
    logger.info("config file        : %s", cfg.source_path)
    logger.info("config fingerprint : %s", cfg.fingerprint())
    logger.info("-" * 78)
    logger.info("optics   : NA=%.2f, %gx, ni=%.4f (design %.4f), ns=%.4f",
                o.numerical_aperture, o.magnification, o.immersion_ri,
                o.immersion_ri_design, o.sample_ri)
    logger.info("           coverslip ng=%.4f t=%.1f um (design %.4f / %.1f um), ti0=%.1f um",
                o.coverslip_ri, o.coverslip_thickness_um, o.coverslip_ri_design,
                o.coverslip_thickness_design_um, o.working_distance_um)
    logger.info("           particle depth below coverslip = %.2f um", o.particle_depth_um)
    logger.info("psf      : model=%s mode=%s oversample=(xy %d, z %d) pupil samples=%d",
                p.model, p.mode, p.oversample_xy, p.oversample_z, p.integration_samples)
    if p.mode == "confocal":
        logger.info("           pinhole = %.2f AU%s", p.pinhole_airy_units,
                    f" (radius override {p.pinhole_radius_um} um)" if p.pinhole_radius_um else "")
    logger.info("deconv   : %s, %d iterations, backend=%s, dtype=%s, pad=%s, reg=%s",
                d.algorithm, d.iterations, d.backend, d.dtype, d.pad_mode, d.regularization)
    logger.info("background: %s", cfg.background.method)
    logger.info("output   : %s, bit-depth policy=%s, overwrite=%s",
                cfg.output.directory, cfg.output.bit_depth_policy, cfg.output.overwrite)
    logger.info("voxel    : source=%s, fallback XY=%.4f um Z=%.4f um",
                cfg.metadata.voxel_size_source, cfg.metadata.fallback_xy_um,
                cfg.metadata.fallback_z_um)
    for i, channel in enumerate(cfg.channels):
        logger.info("channel %d: %-12s emission %.0f nm%s", i, channel.name,
                    channel.emission_nm,
                    f", excitation {channel.excitation_nm:.0f} nm" if channel.excitation_nm else "")
    logger.info("=" * 78)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.output:
        cfg.output.directory = args.output
    if args.iterations:
        cfg.deconvolution.iterations = args.iterations
    if args.intensity_scale:
        cfg.output.intensity_scale = args.intensity_scale
    if args.overwrite:
        cfg.output.overwrite = True
    cfg.validate()

    setup_logging(cfg)
    _log_parameters(cfg)

    from .pipeline import _write_manifest, process_stack, run_batch

    if args.file:
        source = Path(args.file)
        if not source.is_absolute() and not source.exists():
            source = Path(cfg.input.directory) / args.file
        if not source.exists():
            logger.error("file not found: %s", source)
            return 2

        from .pipeline import BatchSummary

        summary = BatchSummary(config_fingerprint=cfg.fingerprint())
        try:
            summary.results.append(process_stack(cfg, source.resolve()))
        except Exception as exc:
            logger.error("FAIL %s: %s", source.name, exc, exc_info=True)
            return 1
        _write_manifest(cfg, summary)
        return 0 if summary.n_failed == 0 else 1

    summary = run_batch(cfg)
    if summary.n_failed:
        logger.error("%d stack(s) failed; see the log above and run_manifest.json",
                     summary.n_failed)
        return 1
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Validate the config and list what would be processed, without computing."""
    cfg = load_config(args.config)
    setup_logging(cfg)
    _log_parameters(cfg)

    from .pipeline import discover_inputs
    from .psf import theoretical_resolution

    for channel in cfg.channels:
        lateral, axial = theoretical_resolution(
            channel.emission_nm, cfg.optics.numerical_aperture, cfg.optics.immersion_ri
        )
        dxy, dz = cfg.metadata.fallback_xy_um, cfg.metadata.fallback_z_um
        logger.info(
            "%-12s Rayleigh lateral %.3f um (Nyquist %.4f, config XY %.4f%s), "
            "axial %.3f um (Nyquist %.3f, config Z %.3f%s)",
            channel.name, lateral, lateral / 2, dxy,
            " OK" if dxy <= lateral / 2 else " UNDERSAMPLED",
            axial, axial / 2, dz,
            " OK" if dz <= axial / 2 else " UNDERSAMPLED",
        )

    try:
        inputs = discover_inputs(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("%d file(s) would be processed:", len(inputs))
    for path in inputs:
        logger.info("  %s", path)
    return 0


def cmd_psf(args: argparse.Namespace) -> int:
    """Compute and save the PSFs alone, for inspection in Fiji."""
    cfg = load_config(args.config)
    setup_logging(cfg)
    _log_parameters(cfg)

    import numpy as np

    from .psf import compute_psf, measure_fwhm
    from .writers import write_ome_tiff

    voxel = (cfg.metadata.fallback_z_um, cfg.metadata.fallback_xy_um, cfg.metadata.fallback_xy_um)
    out_dir = Path(args.out or Path(cfg.output.directory) / "psf")
    out_dir.mkdir(parents=True, exist_ok=True)

    for channel in cfg.channels:
        result = compute_psf(
            voxel_size_um=voxel,
            emission_nm=channel.emission_nm,
            excitation_nm=channel.excitation_nm,
            optics=cfg.optics,
            psf_cfg=cfg.psf,
        )
        fwhm = measure_fwhm(result.data, voxel)
        logger.info("%-12s PSF %s  FWHM xy=%.3f um  z=%.3f um  key=%s",
                    channel.name, result.shape, fwhm["fwhm_x_um"], fwhm["fwhm_z_um"],
                    result.cache_key)
        scaled = (result.data / result.data.max() * 65535).astype(np.uint16)
        write_ome_tiff(
            out_dir / f"psf_{channel.name}.ome.tif",
            scaled[np.newaxis],
            voxel_size_um=voxel,
            channel_names=[channel.name],
            description=f"Theoretical {cfg.psf.model} PSF, peak-normalised to 65535",
        )
    logger.info("PSFs written to %s", out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapse_deconv",
        description="Batch 3D deconvolution of confocal z-stacks (Olympus FV1000 / OME-TIFF)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="deconvolve a folder (or a single file with --file)")
    run.add_argument("config", help="YAML or JSON configuration file")
    run.add_argument("--file", help="process only this stack (test one before the batch)")
    run.add_argument("--output", help="override output.directory")
    run.add_argument("--iterations", type=int, help="override deconvolution.iterations")
    run.add_argument("--intensity-scale", type=float,
                     help="override output.intensity_scale (constant gain before the "
                          "16-bit cast; use the value suggested by the clipping warning)")
    run.add_argument("--overwrite", action="store_true", help="reprocess existing outputs")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="validate the config and list the inputs")
    check.add_argument("config")
    check.set_defaults(func=cmd_check)

    psf = sub.add_parser("psf", help="compute the PSFs only and save them as OME-TIFF")
    psf.add_argument("config")
    psf.add_argument("--out", help="destination directory")
    psf.set_defaults(func=cmd_psf)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
