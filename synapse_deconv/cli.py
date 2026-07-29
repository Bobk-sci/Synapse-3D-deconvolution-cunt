"""Command line entry point.

Installed as ``synapse-deconv``; ``python -m synapse_deconv`` is equivalent and
works without the console script being on PATH, which is the usual failure mode
on Windows when the environment has not been activated.

    python -m synapse_deconv check   config/default.yaml
    python -m synapse_deconv inspect config/default.yaml
    python -m synapse_deconv psf     config/default.yaml --out psf_preview
    python -m synapse_deconv depth   config/default.yaml
    python -m synapse_deconv run     config/default.yaml --file one_stack.oib
    python -m synapse_deconv run     config/default.yaml
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
    for note in cfg.advisories():
        logger.warning("config: %s", note)
    logger.info("=" * 78)


def cmd_count(args: argparse.Namespace) -> int:
    """Detect puncta, colocalise them and count synapses."""
    cfg = load_config(args.config)
    if args.input:
        cfg.input.directory = args.input
    if args.output:
        cfg.output.directory = args.output
    if args.threshold_sigma:
        cfg.detection.threshold_sigma = args.threshold_sigma
    if args.tolerance_um:
        cfg.colocalization.tolerance_um = args.tolerance_um
    if args.presynaptic:
        cfg.colocalization.presynaptic = args.presynaptic
    if args.postsynaptic_excitatory:
        cfg.colocalization.postsynaptic_excitatory = args.postsynaptic_excitatory
    if args.postsynaptic_inhibitory:
        cfg.colocalization.postsynaptic_inhibitory = args.postsynaptic_inhibitory
    cfg.validate()

    setup_logging(cfg)
    _log_parameters(cfg)
    _log_counting_parameters(cfg)

    from .counting import run_counting

    source = None
    if args.file:
        source = Path(args.file)
        if not source.is_absolute() and not source.exists():
            source = Path(cfg.input.directory) / args.file
        if not source.exists():
            logger.error("file not found: %s", source)
            return 2
        source = source.resolve()

    results = run_counting(cfg, single_file=source)
    return 1 if any(r.status != "ok" for r in results) else 0


def _log_counting_parameters(cfg: Config) -> None:
    d, c, a = cfg.detection, cfg.colocalization, cfg.analysis
    logger.info("detection: LoG scales %s um, threshold %.1f x robust noise (MAD)",
                d.punctum_diameters_um, d.threshold_sigma)
    logger.info("           top-hat radius %.2f um, min separation %.2f um, "
                "volume gate [%.4f, %s] um3",
                d.background_radius_um, d.min_separation_um, d.min_volume_um3,
                d.max_volume_um3)
    if d.threshold_sigma_per_channel:
        logger.info("           per-channel thresholds: %s", d.threshold_sigma_per_channel)
    logger.info("coloc    : %s, tolerance %.3f um, one-to-one=%s, "
                "ambiguous partners resolved=%s",
                c.criterion, c.tolerance_um, c.one_to_one, c.resolve_ambiguous_partners)
    logger.info("           pre=%s | excitatory post=%s | inhibitory post=%s",
                c.presynaptic, c.postsynaptic_excitatory, c.postsynaptic_inhibitory)
    logger.info("analysis : ROI=%s, border exclusion %.2f um, densities per %.0f um3",
                a.roi_mask or "whole field", a.border_exclusion_um, a.density_unit_um3)
    logger.info("=" * 78)


def cmd_init(args: argparse.Namespace) -> int:
    """Write a fresh configuration file for the user to edit."""
    from .config import write_template

    destination = Path(args.output or "config/my_study.yaml")
    try:
        written = write_template(destination, overwrite=args.force)
    except ConfigError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(f"Wrote {written}")
    print()
    print("Edit it, then:")
    print(f"  python -m synapse_deconv check   {written}")
    print(f"  python -m synapse_deconv inspect {written}")
    print(f"  python -m synapse_deconv run     {written} --file one_stack.oib")
    print()
    print("Windows paths: use single quotes or forward slashes, never double quotes.")
    print(r"  directory: 'C:\Users\you\data'      directory: C:/Users/you/data")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Report what the pipeline reads from real files, without deconvolving.

    This is the first thing to run on a new dataset: it shows the voxel size
    actually parsed from the metadata, the channel order and wavelengths the
    file declares against the ones the config assumes, the sampling against the
    Nyquist limit, and a background estimate for background.constant_value.
    """
    cfg = load_config(args.config)
    setup_logging(cfg)

    import numpy as np

    from .pipeline import discover_inputs
    from .psf import theoretical_resolution
    from .qc import compute_stats, detect_saturation_level, estimate_background
    from .readers import ReadError, read_stack

    if args.file:
        source = Path(args.file)
        if not source.is_absolute() and not source.exists():
            source = Path(cfg.input.directory) / args.file
        if not source.exists():
            logger.error("file not found: %s", source)
            return 2
        paths = [source]
    else:
        try:
            paths = discover_inputs(cfg)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        if args.limit:
            paths = paths[: args.limit]

    if not paths:
        logger.error("no input file matched %s in %s",
                     cfg.input.patterns, cfg.input.directory)
        return 2

    logger.info("Inspecting %d file(s). Nothing is written and nothing is deconvolved.",
                len(paths))
    failures = 0

    for path in paths:
        logger.info("")
        logger.info("=" * 78)
        try:
            stack = read_stack(
                path,
                voxel_size_source=cfg.metadata.voxel_size_source,
                fallback_xy_um=cfg.metadata.fallback_xy_um,
                fallback_z_um=cfg.metadata.fallback_z_um,
                spacing_divisor=cfg.metadata.spacing_divisor,
                tolerance_warn_ratio=cfg.metadata.tolerance_warn_ratio,
            )
        except ReadError as exc:
            logger.error("%s: UNREADABLE -- %s", path.name, exc)
            failures += 1
            continue

        logger.info("%s", stack.describe())
        dz, dy, dx = stack.voxel_size_um

        if len(cfg.channels) != stack.n_channels:
            logger.error(
                "  CHANNEL COUNT MISMATCH: the file has %d channel(s) but the config "
                "declares %d. Fix the 'channels' section before running.",
                stack.n_channels, len(cfg.channels),
            )
            failures += 1

        logger.info("  %-3s %-14s %-22s %-22s", "idx", "config name",
                    "emission (file/config)", "excitation (file/config)")
        for index in range(stack.n_channels):
            found = stack.channels[index] if index < len(stack.channels) else None
            configured = cfg.channels[index] if index < len(cfg.channels) else None
            file_em = f"{found.emission_nm:.0f}" if found and found.emission_nm else "-"
            file_ex = f"{found.excitation_nm:.0f}" if found and found.excitation_nm else "-"
            cfg_em = f"{configured.emission_nm:.0f}" if configured else "-"
            cfg_ex = (f"{configured.excitation_nm:.0f}"
                      if configured and configured.excitation_nm else "-")
            flag = ""
            if (found and found.emission_nm and configured
                    and abs(found.emission_nm - configured.emission_nm) > 15):
                flag = "   <-- CHECK THE CHANNEL ORDER"
            logger.info("  %-3d %-14s %-22s %-22s%s", index,
                        configured.name if configured else "(none)",
                        f"{file_em} / {cfg_em}", f"{file_ex} / {cfg_ex}", flag)
            if found and found.pinhole_um:
                logger.info("      pinhole in file: %.1f um (back-projected)",
                            found.pinhole_um)

        undersampled = False
        for configured in cfg.channels[: stack.n_channels]:
            lateral, axial = theoretical_resolution(
                configured.emission_nm, cfg.optics.numerical_aperture, cfg.optics.immersion_ri
            )
            undersampled = undersampled or dx > lateral / 2
            logger.info(
                "  %-12s Nyquist XY %.4f um (file %.4f%s) | Nyquist Z %.3f um (file %.3f%s)",
                configured.name, lateral / 2, dx,
                " OK" if dx <= lateral / 2 else " UNDERSAMPLED",
                axial / 2, dz, " OK" if dz <= axial / 2 else " UNDERSAMPLED",
            )

        if undersampled:
            # Turn the warning into an acquisition setting. On a laser scanner
            # the XY pixel is field / pixel count, so either term can fix it.
            n_x = stack.data.shape[3]
            field_um = n_x * dx
            finest = min(
                theoretical_resolution(c.emission_nm, cfg.optics.numerical_aperture,
                                       cfg.optics.immersion_ri)[0] / 2
                for c in cfg.channels[: stack.n_channels]
            )
            needed_pixels = int(np.ceil(field_um / finest))
            zoom_factor = dx / finest
            logger.info(
                "  Field of view is %.1f um (%d px). To reach Nyquist on every channel "
                "(%.4f um/px), either:", field_um, n_x, finest,
            )
            logger.info(
                "    - keep the zoom and scan %d x %d instead of %d x %d "
                "(same field, %.1fx longer, %.2fx fewer photons per pixel), or",
                needed_pixels, needed_pixels, n_x, n_x,
                (needed_pixels / n_x) ** 2, (n_x / needed_pixels) ** 2,
            )
            logger.info(
                "    - keep %d x %d and raise the zoom by x%.2f "
                "(field shrinks to %.1f um, same scan time).",
                n_x, n_x, zoom_factor, field_um / zoom_factor,
            )
            logger.info(
                "    Deconvolution cannot recover detail below 2 pixels (%.3f um here); "
                "that floor is set by the acquisition, not by the algorithm.", 2 * dx,
            )

        logger.info("  %-12s %8s %8s %8s %10s %8s %10s", "channel", "min", "max", "mean",
                    "offset", "mode", "saturated")
        for index in range(stack.n_channels):
            channel_data = stack.data[index]
            stats = compute_stats(channel_data)
            # Two candidates for background.constant_value, see below.
            offset = float(np.percentile(channel_data, 0.1))
            mode = estimate_background(channel_data)
            name = cfg.channels[index].name if index < len(cfg.channels) else f"ch{index}"
            full_scale = detect_saturation_level(channel_data)
            note = ""
            if full_scale is not None and stats.saturated_fraction > 0.0001:
                bits = int(round(np.log2(full_scale + 1)))
                note = f"  <-- SATURATED at {full_scale} ({bits}-bit full scale)"
            logger.info("  %-12s %8.0f %8.0f %8.1f %10.0f %8.0f %9d%s", name,
                        stats.min, stats.max, stats.mean, offset, mode,
                        stats.saturated_voxels, note)
        logger.info(
            "  'offset' (0.1st percentile) = detector pedestal. 'mode' also includes the "
            "diffuse tissue background."
        )
        logger.info(
            "  For background.constant_value: use 'offset' when punctum INTENSITIES are "
            "measured (it preserves photometric linearity); 'mode' only when you just "
            "COUNT puncta -- it detects better but biases dim puncta."
        )

        for warning in stack.warnings:
            logger.warning("  %s", warning)

    logger.info("")
    logger.info("=" * 78)
    if failures:
        logger.error("%d file(s) need attention before the batch can run", failures)
        return 1
    logger.info("All %d file(s) read cleanly.", len(paths))
    logger.info("Next: 'run --file <one stack>' and look at the QC PNG.")
    return 0


def cmd_depth(args: argparse.Namespace) -> int:
    """Show how the PSF and the restoration depend on optics.particle_depth_um.

    Answers the practical question "what do I put in particle_depth_um?" by
    measuring, for the configuration's own optics, how much the PSF changes with
    depth and how much restoration is lost when the assumed depth is wrong.
    """
    cfg = load_config(args.config)
    setup_logging(cfg)

    import numpy as np
    from scipy.signal import fftconvolve

    from .deconvolution import richardson_lucy
    from .psf import compute_psf, measure_fwhm

    channel = cfg.channels[min(args.channel, len(cfg.channels) - 1)]
    voxel = (cfg.metadata.fallback_z_um, cfg.metadata.fallback_xy_um,
             cfg.metadata.fallback_xy_um)
    depths = args.depths or [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    from dataclasses import replace

    def psf_at(depth: float):
        return compute_psf(
            voxel_size_um=voxel, emission_nm=channel.emission_nm,
            excitation_nm=channel.excitation_nm,
            optics=replace(cfg.optics, particle_depth_um=depth),
            psf_cfg=replace(cfg.psf, xy_size=args.psf_size, z_size=args.psf_size,
                            cache_dir=None),
        )

    logger.info("Depth sensitivity for channel %s (em %.0f nm), ni=%.4f, ns=%.4f, NA=%.2f",
                channel.name, channel.emission_nm, cfg.optics.immersion_ri,
                cfg.optics.sample_ri, cfg.optics.numerical_aperture)
    logger.info("voxel (dz, dy, dx) = %s um", voxel)
    logger.info("")
    logger.info("%12s %12s %12s %14s", "depth", "FWHM lateral", "FWHM axial", "relative peak")
    reference = None
    for depth in depths:
        result = psf_at(depth)
        fwhm = measure_fwhm(result.data, voxel)
        peak = float(result.data.max())
        reference = reference if reference is not None else peak
        logger.info("%9.1f um %9.3f um %9.3f um %13.2f",
                    depth, fwhm["fwhm_x_um"], fwhm["fwhm_z_um"], peak / reference)

    if args.no_restoration:
        return 0

    # How much restoration is lost when the assumed depth is wrong: simulate a
    # point source at --true-depth, then deconvolve with the PSF of each depth.
    true_depth = args.true_depth
    logger.info("")
    logger.info("Restoration of a point source truly at %.1f um, deconvolved with the PSF "
                "of each assumed depth (%d iterations):", true_depth, cfg.deconvolution.iterations)
    logger.info("%16s %20s %14s", "assumed depth", "axial concentration", "peak")

    rng = np.random.default_rng(0)
    shape = (40, 96, 96)
    truth = np.zeros(shape)
    points = [(20, 48, 48), (14, 70, 30), (26, 30, 66)]
    for z, y, x in points:
        truth[z, y, x] = 3.0e5
    blurred = np.maximum(fftconvolve(truth, psf_at(true_depth).data, mode="same"), 0.0)
    observed = rng.poisson(blurred).astype(np.float32) + 100.0

    def concentration(volume):
        """Fraction of a punctum's local energy inside +/- 1 z-plane."""
        values, peaks = [], []
        for z, y, x in points:
            box = volume[z - 6:z + 7, y - 8:y + 9, x - 8:x + 9]
            total = box.sum()
            values.append(box[5:8].sum() / total if total else float("nan"))
            peaks.append(box.max())
        return float(np.mean(values)), float(np.mean(peaks))

    raw_conc, raw_peak = concentration(observed)
    logger.info("%16s %19.1f%% %14.0f", "raw (none)", 100 * raw_conc, raw_peak)
    for depth in depths:
        deconvolved = richardson_lucy(
            observed, psf_at(depth).data,
            iterations=cfg.deconvolution.iterations, dtype=cfg.deconvolution.dtype,
        ).data
        conc, peak = concentration(deconvolved)
        marker = "  <-- true depth" if depth == true_depth else ""
        logger.info("%13.1f um %19.1f%% %14.0f%s", depth, 100 * conc, peak, marker)
    return 0


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

    count = sub.add_parser(
        "count",
        help="detect puncta, colocalise them and count synapses on deconvolved stacks",
    )
    count.add_argument("config")
    count.add_argument("--file", help="process only this stack")
    count.add_argument("--input",
                       help="folder of DECONVOLVED stacks to count. input.directory "
                            "normally points at the raw acquisitions, which are not "
                            "what should be counted")
    count.add_argument("--output", help="override output.directory")
    count.add_argument("--threshold-sigma", type=float,
                       help="override detection.threshold_sigma")
    count.add_argument("--tolerance-um", type=float,
                       help="override colocalization.tolerance_um")
    count.add_argument("--presynaptic",
                       help="override colocalization.presynaptic (channel name). "
                            "Permuting the three roles and comparing "
                            "fraction_pre_paired identifies which marker sits on "
                            "which fluorophore when the acquisition sheet is lost")
    count.add_argument("--postsynaptic-excitatory",
                       help="override colocalization.postsynaptic_excitatory")
    count.add_argument("--postsynaptic-inhibitory",
                       help="override colocalization.postsynaptic_inhibitory")
    count.set_defaults(func=cmd_count)

    init = sub.add_parser(
        "init", help="write a fresh configuration file to edit (start here)"
    )
    init.add_argument("output", nargs="?",
                      help="destination (default config/my_study.yaml)")
    init.add_argument("--force", action="store_true",
                      help="overwrite an existing file")
    init.set_defaults(func=cmd_init)

    inspect = sub.add_parser(
        "inspect",
        help="report what is read from real files (voxel size, channels, sampling, "
             "background) without deconvolving anything",
    )
    inspect.add_argument("config")
    inspect.add_argument("--file", help="inspect only this stack")
    inspect.add_argument("--limit", type=int, help="inspect at most N files")
    inspect.set_defaults(func=cmd_inspect)

    depth = sub.add_parser(
        "depth",
        help="show how much optics.particle_depth_um matters for your optics",
    )
    depth.add_argument("config")
    depth.add_argument("--channel", type=int, default=1,
                       help="index of the channel to probe (default 1)")
    depth.add_argument("--depths", type=float, nargs="+",
                       help="depths in um to evaluate")
    depth.add_argument("--true-depth", type=float, default=10.0,
                       help="depth of the simulated point source (default 10)")
    depth.add_argument("--psf-size", type=int, default=25,
                       help="PSF kernel size in voxels, odd (default 25)")
    depth.add_argument("--no-restoration", action="store_true",
                       help="only tabulate the PSF, skip the deconvolution test")
    depth.set_defaults(func=cmd_depth)

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
