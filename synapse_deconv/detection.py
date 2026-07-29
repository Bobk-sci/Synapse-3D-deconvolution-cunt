"""3D punctum detection on anisotropic confocal stacks.

The pipeline per channel is:

1. **Background removal** -- a 3D white top-hat with a structuring element larger
   than a punctum. Everything broader than the element (the diffuse neuropil
   veil, out-of-focus haze) is removed, and what remains sits on a flat, near
   zero baseline. This is what makes a single global threshold defensible.
2. **Laplacian-of-Gaussian** at one or more scales. The LoG is the matched
   filter for a blob: its response peaks at the centre of an object whose radius
   is ``sigma * sqrt(3)`` in 3D. Sigmas are given in **micrometres** and
   converted separately for XY and Z, because a voxel here is 3x taller than it
   is wide -- using one sigma in voxels would look for cigars, not spheres.
3. **Robust thresholding.** The threshold is expressed as a z-score above the
   noise, and the noise is estimated with the median absolute deviation of the
   LoG response. MAD is insensitive to the puncta themselves (they are a small
   fraction of the voxels), so the threshold adapts to the noise level of each
   image without being dragged around by the signal it is meant to find.
4. **Seeded watershed** to turn each detected maximum into a region, which gives
   a volume and an integrated intensity per punctum.

All coordinates leave this module in micrometres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DetectionResult", "detect_puncta", "estimate_noise_mad", "subtract_background"]

#: A LoG filter tuned to sigma responds maximally to a blob of this radius in 3D.
_BLOB_RADIUS_FACTOR = np.sqrt(3.0)

#: How far above the top-hat noise a voxel must be to belong to a punctum's
#: region. Only sets the extent used for volume/intensity, not the detection.
_REGION_SIGMA = 3.0


@dataclass
class DetectionResult:
    """Puncta found in one channel, with the numbers behind the threshold."""

    labels: np.ndarray                  # (Z, Y, X) int32, 0 = background
    centroids_um: np.ndarray            # (N, 3) float, (z, y, x) in micrometres
    volumes_um3: np.ndarray             # (N,)
    mean_intensity: np.ndarray          # (N,) on the background-subtracted image
    max_intensity: np.ndarray           # (N,)
    integrated_intensity: np.ndarray    # (N,) sum of counts, background removed
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.centroids_um)


def estimate_noise_mad(values: np.ndarray) -> float:
    """Robust standard deviation from the median absolute deviation.

    1.4826 * MAD equals the standard deviation for Gaussian noise. Unlike a
    plain std it is not inflated by the puncta, which is essential when the
    quantity being thresholded is the very thing we are looking for.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return 1.4826 * mad


def _structuring_radius_voxels(radius_um: float, voxel: tuple[float, float, float]):
    dz, dy, dx = voxel
    return (max(1, int(round(radius_um / dz))),
            max(1, int(round(radius_um / dy))),
            max(1, int(round(radius_um / dx))))


def subtract_background(
    volume: np.ndarray,
    voxel_size_um: tuple[float, float, float],
    radius_um: float,
) -> np.ndarray:
    """3D white top-hat: keeps only structures smaller than ``radius_um``.

    An ellipsoidal element is built in voxels from the physical radius, so the
    same physical scale is removed along every axis despite the anisotropy.
    """
    from scipy.ndimage import grey_opening

    rz, ry, rx = _structuring_radius_voxels(radius_um, voxel_size_um)
    zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    element = (zz / rz) ** 2 + (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0

    data = np.asarray(volume, dtype=np.float32)
    # Top-hat = image - opening. grey_opening with an explicit footprint is the
    # separable-free but correct route for an ellipsoid.
    opened = grey_opening(data, footprint=element)
    return np.maximum(data - opened, 0.0)


def _log_response(
    volume: np.ndarray,
    sigma_um: float,
    voxel_size_um: tuple[float, float, float],
) -> np.ndarray:
    """Scale-normalised Laplacian-of-Gaussian, anisotropic sigma."""
    from scipy.ndimage import gaussian_laplace

    dz, dy, dx = voxel_size_um
    sigma_vox = (sigma_um / dz, sigma_um / dy, sigma_um / dx)
    # -sigma^2 * LoG makes the response positive on bright blobs and comparable
    # across scales, which is what lets one threshold serve every scale.
    response = -gaussian_laplace(volume, sigma=sigma_vox) * (sigma_um ** 2)
    return response


def detect_puncta(
    volume: np.ndarray,
    voxel_size_um: tuple[float, float, float],
    *,
    punctum_diameters_um: list[float],
    threshold_sigma: float = 5.0,
    background_radius_um: float = 1.0,
    min_separation_um: float = 0.25,
    min_volume_um3: float = 0.0,
    max_volume_um3: float | None = None,
    absolute_threshold: float | None = None,
) -> DetectionResult:
    """Detect blob-like puncta in a 3D volume.

    Args:
        volume: 3D array (Z, Y, X), background-subtracted or raw.
        voxel_size_um: (dz, dy, dx).
        punctum_diameters_um: expected punctum diameters; one LoG scale each.
        threshold_sigma: detection threshold, in robust noise units of the LoG
            response. 5 is conservative, 3 is permissive.
        background_radius_um: white top-hat radius; must exceed the largest
            punctum radius or the puncta themselves get removed.
        min_separation_um: minimum distance between two maxima.
        min_volume_um3 / max_volume_um3: size gate on the segmented regions.
        absolute_threshold: bypass the robust estimate with a fixed LoG
            threshold (only for reproducing a previous run exactly).
    """
    from scipy.ndimage import maximum_filter
    from skimage.measure import regionprops_table
    from skimage.segmentation import watershed

    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {data.ndim}D")
    if not punctum_diameters_um:
        raise ValueError("at least one punctum diameter must be given")

    dz, dy, dx = voxel_size_um
    voxel_volume = dz * dy * dx

    largest_radius = max(punctum_diameters_um) / 2.0
    if background_radius_um <= largest_radius:
        logger.warning(
            "background_radius_um (%.2f) is not larger than the largest punctum "
            "radius (%.2f); the top-hat will erode the puncta it should preserve",
            background_radius_um, largest_radius,
        )

    tophat = subtract_background(data, voxel_size_um, background_radius_um)

    # --- multiscale LoG, keep the best-responding scale per voxel -----------
    best = None
    for diameter in punctum_diameters_um:
        sigma_um = (diameter / 2.0) / _BLOB_RADIUS_FACTOR
        response = _log_response(tophat, sigma_um, voxel_size_um)
        best = response if best is None else np.maximum(best, response)

    # --- local maxima, separated by a physical distance ---------------------
    footprint_shape = (
        max(1, int(round(min_separation_um / dz))) * 2 + 1,
        max(1, int(round(min_separation_um / dy))) * 2 + 1,
        max(1, int(round(min_separation_um / dx))) * 2 + 1,
    )
    # The top-hat is undefined within one structuring element of the array
    # boundary: the element does not fit, the boundary extension invents data,
    # and a signal running off the edge of the field leaves a bright rim that
    # local maxima latch onto. Those voxels are removed from the detection
    # rather than left to be filtered downstream, because they also corrupt the
    # threshold calibration by shifting the distribution of the maxima.
    valid = np.zeros(data.shape, dtype=bool)
    bz, by, bx = _structuring_radius_voxels(background_radius_um, voxel_size_um)
    valid[bz:data.shape[0] - bz or None,
          by:data.shape[1] - by or None,
          bx:data.shape[2] - bx or None] = True

    is_maximum = (best == maximum_filter(best, size=footprint_shape)) & valid
    peak_values = best[is_maximum]

    # The threshold is calibrated on the distribution of the LOCAL MAXIMA, not
    # on the bulk of the volume. In a sparse image almost every local maximum is
    # noise, but the maximum of a neighbourhood sits far above the bulk median:
    # a k-sigma cut derived from the whole volume lets essentially all of them
    # through (extreme-value statistics over ~n_voxels / footprint independent
    # neighbourhoods). Measuring the median and the MAD of the maxima themselves
    # puts the threshold where it belongs, and makes it self-calibrating.
    peak_median = float(np.median(peak_values)) if peak_values.size else 0.0
    peak_noise = estimate_noise_mad(peak_values)
    if absolute_threshold is not None:
        threshold = float(absolute_threshold)
    elif peak_noise <= 0:
        threshold = peak_median + float(np.finfo(np.float32).eps)
        logger.warning(
            "the local maxima of the LoG response have zero spread; "
            "the threshold falls back to their median"
        )
    else:
        threshold = peak_median + threshold_sigma * peak_noise

    seed_coords = np.argwhere(is_maximum & (best > threshold))

    # A field can hold at most one maximum per min_separation neighbourhood.
    # Approaching that ceiling means the puncta are packed closer than this
    # sampling can resolve: the count becomes a lower bound, and the threshold
    # calibration degrades too, since it assumes most local maxima are noise.
    footprint_volume = float(np.prod(footprint_shape)) * dz * dy * dx
    valid_volume = float(valid.sum()) * dz * dy * dx
    max_resolvable = valid_volume / footprint_volume if footprint_volume > 0 else float("inf")
    occupancy = len(seed_coords) / max_resolvable if max_resolvable else 0.0

    diagnostics = {
        "log_peak_median": round(peak_median, 6),
        "log_peak_mad": round(float(peak_noise), 6),
        "log_threshold": float(threshold),
        "threshold_sigma": float(threshold_sigma),
        "n_local_maxima": int(peak_values.size),
        "border_excluded_voxels": [int(bz), int(by), int(bx)],
        "max_resolvable_puncta": int(max_resolvable),
        "maxima_occupancy": round(float(occupancy), 4),
        "n_seeds": int(len(seed_coords)),
        "tophat_p99_9": float(np.percentile(tophat, 99.9)),
        "background_radius_um": float(background_radius_um),
        "scales_um": list(punctum_diameters_um),
    }

    if occupancy > 0.3:
        logger.warning(
            "detected puncta fill %.0f%% of what this sampling can resolve "
            "(%d of at most %d at min_separation_um=%.2f um). The field is near the "
            "packing limit: counts are a LOWER BOUND, not a count. Sample finer.",
            100 * occupancy, len(seed_coords), int(max_resolvable), min_separation_um,
        )

    if len(seed_coords) == 0:
        empty = np.zeros((0,), dtype=float)
        # Distinguish "nothing there" from "the calibration broke". The
        # calibration assumes most local maxima are noise; in a field packed
        # with puncta that assumption fails, the median of the maxima rises
        # with the signal, and the threshold ends up above everything.
        tophat_p999 = float(np.percentile(tophat, 99.9))
        dense_field = tophat_p999 > 10 * estimate_noise_mad(tophat)
        logger.warning(
            "no punctum passed the threshold (%.4g on the LoG response). %s",
            threshold,
            "The image clearly carries signal (top-hat p99.9 = %.0f), so the "
            "automatic threshold has broken down: it assumes most local maxima are "
            "noise, which fails in a field packed with puncta. Set "
            "detection.absolute_threshold, or sample finer." % tophat_p999
            if dense_field else
            "The image appears empty at %.1f sigma above the typical noise maximum."
            % threshold_sigma,
        )
        return DetectionResult(
            labels=np.zeros(data.shape, dtype=np.int32),
            centroids_um=np.zeros((0, 3), dtype=float),
            volumes_um3=empty, mean_intensity=empty,
            max_intensity=empty, integrated_intensity=empty,
            diagnostics=diagnostics,
        )

    # --- grow each seed into a region so it has a volume --------------------
    markers = np.zeros(data.shape, dtype=np.int32)
    markers[tuple(seed_coords.T)] = np.arange(1, len(seed_coords) + 1)
    # The region mask bounds how far a punctum may grow. A top-hat of noisy data
    # leaves a positive noise floor, so the cut has to be taken relative to the
    # MEDIAN of the top-hat, not from zero -- otherwise the mask swallows the
    # noise, the watershed floods it into the regions, and every punctum ends up
    # hundreds of voxels wide.
    tophat_median = float(np.median(tophat))
    tophat_noise = estimate_noise_mad(tophat)
    mask_level = tophat_median + _REGION_SIGMA * tophat_noise
    mask = tophat > mask_level
    mask[tuple(seed_coords.T)] = True
    labels = watershed(-tophat, markers=markers, mask=mask)
    diagnostics["tophat_median"] = round(tophat_median, 4)
    diagnostics["tophat_mad"] = round(float(tophat_noise), 4)
    diagnostics["region_mask_level"] = round(float(mask_level), 4)

    props = regionprops_table(
        labels, intensity_image=tophat,
        properties=("label", "centroid", "area", "intensity_mean", "intensity_max"),
    )
    centroids = np.stack(
        [props["centroid-0"], props["centroid-1"], props["centroid-2"]], axis=1
    )
    centroids_um = centroids * np.array([dz, dy, dx])
    volumes = props["area"] * voxel_volume
    mean_int = props["intensity_mean"]
    integrated = mean_int * props["area"]

    keep = volumes >= min_volume_um3
    if max_volume_um3 is not None:
        keep &= volumes <= max_volume_um3
    diagnostics["n_rejected_by_size"] = int((~keep).sum())

    if not np.all(keep):
        surviving = set(np.asarray(props["label"])[keep].tolist())
        remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for new, old in enumerate(sorted(surviving), start=1):
            remap[old] = new
        labels = remap[labels]

    return DetectionResult(
        labels=labels.astype(np.int32),
        centroids_um=centroids_um[keep],
        volumes_um3=np.asarray(volumes)[keep],
        mean_intensity=np.asarray(mean_int)[keep],
        max_intensity=np.asarray(props["intensity_max"])[keep],
        integrated_intensity=np.asarray(integrated)[keep],
        diagnostics=diagnostics,
    )
