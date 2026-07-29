"""Synapse detection, colocalisation and counting over a folder of stacks.

Per stack: detect puncta in each channel, pair pre- with post-synaptic puncta by
3D apposition, restrict to a ROI, and report counts as densities. Everything is
driven by the same config as the deconvolution, so a study is described by one
file end to end.

Densities, not raw counts, are the comparable quantity: the analysed volume
differs between images as soon as a ROI mask or the border exclusion is in play.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .colocalization import (
    PairingResult,
    chance_pairing_rate,
    estimate_channel_offset,
    nearest_neighbour_profile,
    pair_by_contact,
    pair_puncta,
    resolve_exclusive_partners,
)
from .config import Config
from .detection import DetectionResult, detect_puncta
from .pipeline import discover_inputs
from .readers import ReadError, read_stack

logger = logging.getLogger(__name__)

__all__ = ["CountResult", "count_stack", "run_counting", "PUNCTA_COLUMNS", "SUMMARY_COLUMNS"]

PUNCTA_COLUMNS = [
    "image", "channel", "punctum_id",
    "z_um", "y_um", "x_um",
    "volume_um3", "mean_intensity", "max_intensity", "integrated_intensity",
    "in_roi", "synapse_type", "partner_distance_um",
]

SUMMARY_COLUMNS = [
    "image", "voxel_dz_um", "voxel_dy_um", "voxel_dx_um",
    "analysed_volume_um3",
    "n_presynaptic", "n_post_excitatory", "n_post_inhibitory",
    "n_synapses_excitatory", "n_synapses_inhibitory", "n_synapses_total",
    "density_presynaptic", "density_post_excitatory", "density_post_inhibitory",
    "density_excitatory", "density_inhibitory", "density_total",
    "fraction_pre_paired",
    "specific_enrichment_excitatory", "specific_enrichment_inhibitory",
    "median_distance_exc_um", "median_distance_inh_um",
    "n_ambiguous_resolved", "warnings",
]


@dataclass
class CountResult:
    """Everything measured on one stack."""

    source: Path
    status: str = "ok"
    voxel_size_um: tuple[float, float, float] | None = None
    analysed_volume_um3: float = 0.0
    per_channel: dict[str, dict[str, Any]] = field(default_factory=dict)
    synapses: dict[str, dict[str, Any]] = field(default_factory=dict)
    puncta_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    duration_s: float = 0.0
    qc_figure: Path | None = None

    def summary_row(self) -> dict[str, Any]:
        dz, dy, dx = self.voxel_size_um or (float("nan"),) * 3
        volume = self.analysed_volume_um3
        unit = self.synapses.get("density_unit_um3", 100.0)

        def density(n: int) -> float:
            return (n / volume * unit) if volume > 0 else float("nan")

        pre = self.synapses.get("presynaptic_channel", "")
        exc = self.synapses.get("excitatory_channel", "")
        inh = self.synapses.get("inhibitory_channel", "")
        n_pre = self.per_channel.get(pre, {}).get("n_in_roi", 0)
        n_exc = self.per_channel.get(exc, {}).get("n_in_roi", 0)
        n_inh = self.per_channel.get(inh, {}).get("n_in_roi", 0)
        s_exc = self.synapses.get("n_excitatory", 0)
        s_inh = self.synapses.get("n_inhibitory", 0)
        profiles = self.synapses.get("apposition_profile", {})

        def profile_at(key: str) -> Any:
            value = profiles.get(key, {}).get("specific_at_tolerance")
            return "" if value is None else value

        return {
            "image": self.source.name,
            "voxel_dz_um": dz, "voxel_dy_um": dy, "voxel_dx_um": dx,
            "analysed_volume_um3": round(volume, 2),
            "n_presynaptic": n_pre,
            "n_post_excitatory": n_exc,
            "n_post_inhibitory": n_inh,
            "n_synapses_excitatory": s_exc,
            "n_synapses_inhibitory": s_inh,
            "n_synapses_total": s_exc + s_inh,
            "density_presynaptic": round(density(n_pre), 4),
            "density_post_excitatory": round(density(n_exc), 4),
            "density_post_inhibitory": round(density(n_inh), 4),
            "density_excitatory": round(density(s_exc), 4),
            "density_inhibitory": round(density(s_inh), 4),
            "density_total": round(density(s_exc + s_inh), 4),
            "fraction_pre_paired": round(
                (s_exc + s_inh) / n_pre if n_pre else float("nan"), 4
            ),
            # How much closer the pair is than the two postsynaptic markers are
            # to each other. Near 1 means the count is a proximity artefact, and
            # comparing such counts across groups compares labelling density,
            # not synapses.
            "specific_enrichment_excitatory": profile_at("excitatory"),
            "specific_enrichment_inhibitory": profile_at("inhibitory"),
            "median_distance_exc_um": self.synapses.get("median_distance_exc_um", ""),
            "median_distance_inh_um": self.synapses.get("median_distance_inh_um", ""),
            "n_ambiguous_resolved": self.synapses.get("n_ambiguous_resolved", 0),
            "warnings": " | ".join(self.warnings),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

#: Below this, a synaptic pair is not measurably closer than two markers that
#: merely share the neuropile, and its count is not evidence of apposition.
_MIN_SPECIFIC_ENRICHMENT = 1.25


def specific_enrichment(profile: dict, control: dict) -> list[float | None]:
    """Enrichment of a synaptic pair relative to the non-specific floor.

    ``profile`` and ``control`` must come from the same stack, so that the
    shared spatial support cancels: the control pair is subject to exactly the
    same neuropile clustering as the synaptic pair, and is the only null
    available that carries it. A ratio of 1 means the "synaptic" pair is no
    closer than two markers that are known not to be apposed.
    """
    if not profile.get("enrichment") or not control.get("enrichment"):
        return []
    return [
        round(float(e / c), 3) if (e is not None and c) else None
        for e, c in zip(profile["enrichment"], control["enrichment"])
    ]


def _log_specific_enrichment(
    profiles: dict[str, dict],
    pre_name: str,
    exc_name: str,
    inh_name: str,
    tolerance_um: float,
    result: CountResult,
) -> None:
    """Report each synaptic pair against the postsynaptic control pair."""
    control = profiles.get("control", {})
    if not control.get("observed"):
        return

    cells = [
        f"r<={r:.2f}:x{e:g}"
        for r, e in zip(control["radius_um"], control["enrichment"])
        if e is not None
    ][:6]
    logger.info(
        "  apposition %s vs %s [CONTROL -- different synapses, must not be "
        "apposed]: %s", exc_name, inh_name, "  ".join(cells) or "too few puncta",
    )

    radii = control["radius_um"]
    at_tolerance = int(np.argmin(np.abs(np.array(radii) - tolerance_um)))
    for key, post_name in (("excitatory", exc_name), ("inhibitory", inh_name)):
        ratios = specific_enrichment(profiles.get(key, {}), control)
        if not ratios:
            continue
        profiles[key]["specific_enrichment"] = ratios
        cells = [f"r<={r:.2f}:x{s:g}" for r, s in zip(radii, ratios)
                 if s is not None][:6]
        logger.info("  specific enrichment %s vs %s (above the control pair): %s",
                    pre_name, post_name, "  ".join(cells) or "too few puncta")

        value = ratios[at_tolerance] if at_tolerance < len(ratios) else None
        if value is None:
            usable = [s for s in ratios if s is not None]
            value = usable[0] if usable else None
        profiles[key]["specific_at_tolerance"] = value
        if value is not None and value < _MIN_SPECIFIC_ENRICHMENT:
            result.warnings.append(
                f"{key}: {pre_name}/{post_name} is only {value:g}x closer than the "
                f"{exc_name}/{inh_name} control pair, which is on different synapses "
                "and should show no apposition at all. The raw enrichment against the "
                "randomised null is inflated by both markers sharing the neuropile; "
                "corrected for that, this pair shows no specific apposition and its "
                "synapse count should be reported as an upper bound, not as a count"
            )


def _load_roi_mask(path: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Read an optional ROI mask and check it matches the stack."""
    import tifffile

    mask = np.asarray(tifffile.imread(path))
    if mask.ndim == 4:                       # (C, Z, Y, X) or (Z, C, Y, X)
        mask = mask.max(axis=int(np.argmin(mask.shape[:2])))
    if mask.shape != shape:
        raise ValueError(
            f"ROI mask shape {mask.shape} does not match the stack {shape}"
        )
    return mask > 0


def _roi_and_volume(
    cfg: Config, shape: tuple[int, int, int], voxel: tuple[float, float, float]
) -> tuple[np.ndarray, float]:
    """Build the analysis mask and the physical volume it covers."""
    dz, dy, dx = voxel
    roi = np.ones(shape, dtype=bool)

    if cfg.analysis.roi_mask:
        roi &= _load_roi_mask(cfg.analysis.roi_mask, shape)

    margin = cfg.analysis.border_exclusion_um
    if margin > 0:
        mz = int(round(margin / dz))
        my = int(round(margin / dy))
        mx = int(round(margin / dx))
        border = np.zeros(shape, dtype=bool)
        border[mz:shape[0] - mz or None, my:shape[1] - my or None,
               mx:shape[2] - mx or None] = True
        roi &= border

    volume_um3 = float(roi.sum()) * dz * dy * dx
    return roi, volume_um3


def _inside(centroids_um: np.ndarray, roi: np.ndarray,
            voxel: tuple[float, float, float]) -> np.ndarray:
    """Boolean mask of the centroids that fall inside the ROI."""
    if len(centroids_um) == 0:
        return np.zeros((0,), dtype=bool)
    dz, dy, dx = voxel
    idx = np.round(centroids_um / np.array([dz, dy, dx])).astype(int)
    for axis in range(3):
        np.clip(idx[:, axis], 0, roi.shape[axis] - 1, out=idx[:, axis])
    return roi[idx[:, 0], idx[:, 1], idx[:, 2]]


def _pair_arrays(cfg: Config, pre: DetectionResult, post: DetectionResult,
                 post_centroids_um: np.ndarray,
                 voxel: tuple[float, float, float]) -> PairingResult:
    """Pair, using possibly offset-corrected postsynaptic centroids."""
    coloc = cfg.colocalization
    if coloc.criterion == "contact":
        return pair_by_contact(
            pre.labels, post.labels, pre.centroids_um, post.centroids_um, voxel,
            dilation_um=coloc.contact_dilation_um, one_to_one=coloc.one_to_one,
        )
    return pair_puncta(
        pre.centroids_um, post_centroids_um,
        tolerance_um=coloc.tolerance_um, one_to_one=coloc.one_to_one,
    )


# --------------------------------------------------------------------------
# single stack
# --------------------------------------------------------------------------

def count_stack(cfg: Config, source: Path) -> CountResult:
    """Detect, colocalise and count on one deconvolved stack."""
    started = time.perf_counter()
    cfg.require_synaptic_roles()
    result = CountResult(source=source)

    stack = read_stack(
        source,
        voxel_size_source=cfg.metadata.voxel_size_source,
        fallback_xy_um=cfg.metadata.fallback_xy_um,
        fallback_z_um=cfg.metadata.fallback_z_um,
        spacing_divisor=cfg.metadata.spacing_divisor,
        tolerance_warn_ratio=cfg.metadata.tolerance_warn_ratio,
    )
    voxel = stack.voxel_size_um
    result.voxel_size_um = voxel
    result.warnings.extend(stack.warnings)
    logger.info(stack.describe())

    if len(cfg.channels) != stack.n_channels:
        raise ValueError(
            f"config declares {len(cfg.channels)} channel(s) but {source.name} has "
            f"{stack.n_channels}"
        )

    roi, volume_um3 = _roi_and_volume(cfg, stack.spatial_shape, voxel)
    result.analysed_volume_um3 = volume_um3
    unit = cfg.analysis.density_unit_um3

    detections: dict[str, DetectionResult] = {}
    inside_flags: dict[str, np.ndarray] = {}

    for index, channel in enumerate(cfg.channels):
        sigma = cfg.detection.threshold_sigma_per_channel.get(
            channel.name, cfg.detection.threshold_sigma
        )
        detection = detect_puncta(
            stack.data[index], voxel,
            punctum_diameters_um=cfg.detection.punctum_diameters_um,
            threshold_sigma=sigma,
            background_radius_um=cfg.detection.background_radius_um,
            min_separation_um=cfg.detection.min_separation_um,
            min_volume_um3=cfg.detection.min_volume_um3,
            max_volume_um3=cfg.detection.max_volume_um3,
        )
        detections[channel.name] = detection
        inside = _inside(detection.centroids_um, roi, voxel)
        inside_flags[channel.name] = inside

        n_in = int(inside.sum())
        density = n_in / volume_um3 * unit if volume_um3 > 0 else float("nan")
        snr = (
            float(np.median(detection.max_intensity[inside]) / detection.diagnostics["tophat_mad"])
            if n_in and detection.diagnostics.get("tophat_mad", 0) > 0 else float("nan")
        )
        result.per_channel[channel.name] = {
            "n_detected": detection.count,
            "n_in_roi": n_in,
            "density_per_unit": round(density, 4),
            "threshold_sigma": sigma,
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in detection.diagnostics.items()},
            "median_snr": round(snr, 3) if np.isfinite(snr) else None,
        }

        # -- coherence checks ------------------------------------------------
        low, high = cfg.analysis.expected_puncta_per_100um3
        if n_in == 0:
            result.warnings.append(
                f"{channel.name}: no punctum detected. Lower detection.threshold_sigma "
                f"(currently {sigma}) or check that this channel carries signal"
            )
        elif np.isfinite(density) and not (low <= density * 100.0 / unit <= high):
            result.warnings.append(
                f"{channel.name}: {density * 100.0 / unit:.1f} puncta/100 um3 is outside the "
                f"expected range [{low}, {high}]; verify the threshold before trusting it"
            )
        if np.isfinite(snr) and snr < cfg.analysis.min_snr_warn:
            result.warnings.append(
                f"{channel.name}: median punctum only {snr:.1f}x the noise "
                f"(< {cfg.analysis.min_snr_warn}); this image is weak"
            )

        logger.info(
            "  %-12s %5d puncta (%d in ROI, %.2f /%.0f um3) | LoG thr %.4g = %.1f x MAD "
            "| median SNR %s",
            channel.name, detection.count, n_in, density, unit,
            detection.diagnostics["log_threshold"], sigma,
            f"{snr:.1f}" if np.isfinite(snr) else "n/a",
        )

    # --- pair pre with each post-synaptic channel ---------------------------
    coloc = cfg.colocalization
    pre_name = coloc.presynaptic
    exc_name = coloc.postsynaptic_excitatory
    inh_name = coloc.postsynaptic_inhibitory

    offsets: dict[str, dict[str, float]] = {}
    if coloc.measure_channel_offset:
        for key, post_name in (("excitatory", exc_name), ("inhibitory", inh_name)):
            offsets[key] = estimate_channel_offset(
                detections[pre_name].centroids_um, detections[post_name].centroids_um,
                search_radius_um=max(1.0, 3 * coloc.tolerance_um),
            )
            o = offsets[key]
            if o["n_pairs"]:
                logger.info(
                    "  channel offset %s vs %s: (z %+.3f, y %+.3f, x %+.3f) um, "
                    "|shift| %.3f um from %d mutual neighbours "
                    "(scatter %.3f um, ratio %.2f -- below 1 means a real shift)",
                    pre_name, post_name, o["shift_z_um"], o["shift_y_um"],
                    o["shift_x_um"], o["shift_norm_um"], o["n_pairs"], o["scatter_um"],
                    o["scatter_ratio"],
                )
                if (o["shift_norm_um"] > coloc.offset_warn_fraction * coloc.tolerance_um
                        and o["scatter_ratio"] < 1.0):
                    result.warnings.append(
                        f"{pre_name} and {post_name} are systematically offset by "
                        f"{o['shift_norm_um']:.3f} um, which is "
                        f"{o['shift_norm_um'] / coloc.tolerance_um:.0%} of the "
                        f"{coloc.tolerance_um} um tolerance. Chromatic aberration or "
                        "detector misalignment of this size pushes true pairs outside "
                        "the criterion and collapses the pairing rate. Calibrate with "
                        "beads, or set colocalization.correct_channel_offset once you "
                        "have confirmed the shift is instrumental."
                    )

    def shifted(name: str, key: str) -> np.ndarray:
        centroids = detections[name].centroids_um
        o_check = offsets.get(key, {})
        if not (coloc.correct_channel_offset and o_check.get("n_pairs")
                and o_check.get("scatter_ratio", float("inf")) < 1.0):
            return centroids
        o = offsets[key]
        return centroids - np.array([o["shift_z_um"], o["shift_y_um"], o["shift_x_um"]])

    excitatory = _pair_arrays(cfg, detections[pre_name], detections[exc_name],
                              shifted(exc_name, "excitatory"), voxel)
    inhibitory = _pair_arrays(cfg, detections[pre_name], detections[inh_name],
                              shifted(inh_name, "inhibitory"), voxel)

    n_ambiguous = 0
    if coloc.resolve_ambiguous_partners:
        excitatory, inhibitory, n_ambiguous = resolve_exclusive_partners(excitatory, inhibitory)

    # A synapse counts only if its midpoint lies in the ROI.
    exc_inside = _inside(excitatory.midpoints_um, roi, voxel)
    inh_inside = _inside(inhibitory.midpoints_um, roi, voxel)
    n_exc, n_inh = int(exc_inside.sum()), int(inh_inside.sum())

    chance: dict[str, dict[str, float]] = {}
    if coloc.chance_randomisations > 0 and coloc.criterion == "distance":
        extent = np.array(stack.spatial_shape) * np.array(voxel)
        for key, post_name in (("excitatory", exc_name), ("inhibitory", inh_name)):
            chance[key] = chance_pairing_rate(
                detections[pre_name].centroids_um, detections[post_name].centroids_um,
                extent, tolerance_um=coloc.tolerance_um, one_to_one=coloc.one_to_one,
                n_randomisations=coloc.chance_randomisations,
            )
        for key, observed in (("excitatory", n_exc), ("inhibitory", n_inh)):
            mean = chance[key]["chance_mean"]
            logger.info(
                "  chance control (%s): %.1f +/- %.1f pairs by coincidence "
                "-> %.0f%% of the %d observed",
                key, mean, chance[key]["chance_std"],
                100 * mean / observed if observed else float("nan"), observed,
            )
            if observed and mean / observed > 0.25:
                result.warnings.append(
                    f"{key}: {100 * mean / observed:.0f}% of the pairs are reproduced by "
                    f"chance at this punctum density and a {coloc.tolerance_um} um "
                    "tolerance; tighten the tolerance or treat the count as an upper bound"
                )

    profiles: dict[str, dict] = {}
    if coloc.chance_randomisations > 0 and coloc.criterion == "distance":
        extent = np.array(stack.spatial_shape) * np.array(voxel)
        for key, post_name in (("excitatory", exc_name), ("inhibitory", inh_name)):
            profile = nearest_neighbour_profile(
                detections[pre_name].centroids_um, detections[post_name].centroids_um,
                extent, n_randomisations=coloc.chance_randomisations,
            )
            profiles[key] = profile
            if profile["observed"]:
                cells = [
                    f"r<={r:.2f}:x{e:g}"
                    for r, e in zip(profile["radius_um"], profile["enrichment"])
                    if e is not None
                ][:6]
                logger.info("  apposition %s vs %s (cumulative enrichment vs chance): %s",
                            pre_name, post_name, "  ".join(cells) or "too few puncta")
                usable = [e for e in profile["enrichment"] if e is not None]
                first = usable[0] if usable else None
                if first is not None and first < 2.0:
                    result.warnings.append(
                        f"{pre_name}/{post_name}: nearest-neighbour enrichment is only "
                        f"{first:g} in the closest bin. The two channels show no clear "
                        "apposition, so no tolerance will give a meaningful synapse "
                        "count -- check the marker assignment and the detection "
                        "threshold before trusting these numbers"
                    )

        # The two postsynaptic markers sit on DIFFERENT synapses, so this pair
        # must not be apposed. Whatever enrichment it still shows is the
        # non-specific floor of the measurement: both markers live in the
        # neuropile and avoid the same cell bodies and vessels, so they are
        # neighbours far more often than the null predicts. The null randomises
        # by translating one channel, which preserves each channel's own
        # clustering but not the fact that the two share a support -- it cannot
        # see this floor, and every enrichment above is inflated by it.
        profiles["control"] = nearest_neighbour_profile(
            detections[exc_name].centroids_um, detections[inh_name].centroids_um,
            extent, n_randomisations=coloc.chance_randomisations,
        )
        _log_specific_enrichment(profiles, pre_name, exc_name, inh_name,
                                 coloc.tolerance_um, result)

    result.synapses = {
        "chance": chance,
        "apposition_profile": profiles,
        "channel_offsets": offsets,
        "presynaptic_channel": pre_name,
        "excitatory_channel": exc_name,
        "inhibitory_channel": inh_name,
        "criterion": coloc.criterion,
        "tolerance_um": coloc.tolerance_um,
        "one_to_one": coloc.one_to_one,
        "n_excitatory": n_exc,
        "n_inhibitory": n_inh,
        "n_ambiguous_resolved": n_ambiguous,
        "density_unit_um3": unit,
        "median_distance_exc_um": (
            round(float(np.median(excitatory.distances_um[exc_inside])), 4)
            if n_exc else ""
        ),
        "median_distance_inh_um": (
            round(float(np.median(inhibitory.distances_um[inh_inside])), 4)
            if n_inh else ""
        ),
    }

    logger.info(
        "  synapses: %d excitatory (%s+%s), %d inhibitory (%s+%s) in %.0f um3 "
        "-> %.2f / %.2f per %.0f um3",
        n_exc, pre_name, exc_name, n_inh, pre_name, inh_name, volume_um3,
        n_exc / volume_um3 * unit if volume_um3 else float("nan"),
        n_inh / volume_um3 * unit if volume_um3 else float("nan"), unit,
    )

    result.puncta_rows = _build_puncta_rows(
        source, cfg, detections, inside_flags, excitatory, inhibitory
    )

    if cfg.qc.enabled:
        from .qc_counts import write_detection_overlay

        qc_path = (Path(cfg.output.directory) / cfg.analysis.subdirectory / "qc"
                   / f"{source.stem}_detection.png")
        synaptic_indices: dict[str, set[int]] = {name: set() for name in detections}
        for pairing, selection, post_name in (
            (excitatory, exc_inside, exc_name),
            (inhibitory, inh_inside, inh_name),
        ):
            for (pre_i, post_i) in pairing.pairs[selection]:
                synaptic_indices[pre_name].add(int(pre_i))
                synaptic_indices[post_name].add(int(post_i))

        result.qc_figure = write_detection_overlay(
            qc_path, stack.data, detections, inside_flags,
            channel_names=[c.name for c in cfg.channels],
            voxel_size_um=voxel,
            synaptic_indices=synaptic_indices,
            synapse_points_um={
                "excitatory": excitatory.midpoints_um[exc_inside],
                "inhibitory": inhibitory.midpoints_um[inh_inside],
            },
            title=f"{source.name} — thr {cfg.detection.threshold_sigma} sigma, "
                  f"apposition <= {coloc.tolerance_um} um",
            percentile_clip=tuple(cfg.qc.percentile_clip),
            display_gamma=cfg.qc.display_gamma,
            dpi=cfg.qc.dpi,
        )
    result.status = "ok"
    result.duration_s = time.perf_counter() - started
    for warning in result.warnings:
        logger.warning("  %s: %s", source.name, warning)
    return result


def _build_puncta_rows(
    source: Path,
    cfg: Config,
    detections: dict[str, DetectionResult],
    inside_flags: dict[str, np.ndarray],
    excitatory: PairingResult,
    inhibitory: PairingResult,
) -> list[dict[str, Any]]:
    """One CSV row per punctum, annotated with its synaptic partner if any."""
    coloc = cfg.colocalization
    annotation: dict[tuple[str, int], tuple[str, float]] = {}
    for label, pairing, post_name in (
        ("excitatory", excitatory, coloc.postsynaptic_excitatory),
        ("inhibitory", inhibitory, coloc.postsynaptic_inhibitory),
    ):
        for (pre_i, post_i), distance in zip(pairing.pairs, pairing.distances_um):
            annotation[(coloc.presynaptic, int(pre_i))] = (label, float(distance))
            annotation[(post_name, int(post_i))] = (label, float(distance))

    rows: list[dict[str, Any]] = []
    for name, detection in detections.items():
        inside = inside_flags[name]
        for i in range(detection.count):
            synapse_type, distance = annotation.get((name, i), ("none", ""))
            z, y, x = detection.centroids_um[i]
            rows.append({
                "image": source.name,
                "channel": name,
                "punctum_id": i + 1,
                "z_um": round(float(z), 4),
                "y_um": round(float(y), 4),
                "x_um": round(float(x), 4),
                "volume_um3": round(float(detection.volumes_um3[i]), 5),
                "mean_intensity": round(float(detection.mean_intensity[i]), 2),
                "max_intensity": round(float(detection.max_intensity[i]), 2),
                "integrated_intensity": round(float(detection.integrated_intensity[i]), 2),
                "in_roi": bool(inside[i]),
                "synapse_type": synapse_type,
                "partner_distance_um": round(distance, 4) if distance != "" else "",
            })
    return rows


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_counting(cfg: Config, single_file: Path | None = None) -> list[CountResult]:
    """Count synapses on every deconvolved stack described by the config."""
    started = time.perf_counter()
    out_dir = Path(cfg.output.directory) / cfg.analysis.subdirectory

    paths = [single_file] if single_file else discover_inputs(cfg)
    logger.info("Config fingerprint: %s", cfg.fingerprint())
    logger.info("Counting synapses on %d stack(s)", len(paths))

    results: list[CountResult] = []
    for path in paths:
        logger.info("-" * 78)
        try:
            result = count_stack(cfg, path)
        except (ReadError, ValueError, RuntimeError, OSError, MemoryError) as exc:
            logger.error("FAIL %s: %s", path.name, exc,
                         exc_info=logger.isEnabledFor(logging.DEBUG))
            result = CountResult(source=path, status="failed",
                                 message=f"{type(exc).__name__}: {exc}")
            results.append(result)
            if cfg.processing.fail_fast:
                raise
            continue

        _write_csv(out_dir / f"{path.stem}_puncta.csv", PUNCTA_COLUMNS, result.puncta_rows)
        results.append(result)

    ok = [r for r in results if r.status == "ok"]
    _write_csv(out_dir / "summary.csv", SUMMARY_COLUMNS, [r.summary_row() for r in ok])

    manifest = out_dir / "counting_manifest.json"
    manifest.write_text(json.dumps({
        "config_fingerprint": cfg.fingerprint(),
        "config": cfg.to_dict(),
        "duration_s": round(time.perf_counter() - started, 2),
        "stacks": [
            {
                "source": str(r.source), "status": r.status, "message": r.message,
                "analysed_volume_um3": r.analysed_volume_um3,
                "per_channel": r.per_channel, "synapses": r.synapses,
                "warnings": r.warnings,
            }
            for r in results
        ],
    }, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 78)
    logger.info("Counted %d stack(s), %d failed. CSVs in %s",
                len(ok), len(results) - len(ok), out_dir)
    return results
