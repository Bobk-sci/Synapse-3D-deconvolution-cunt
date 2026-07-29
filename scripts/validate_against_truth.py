"""Quantify what the deconvolution actually did, against a known ground truth.

Only usable on stacks produced by ``make_test_stack.py``, where the true punctum
positions are known. Measures, per channel:

  * the lateral and axial FWHM of isolated puncta, raw vs deconvolved;
  * how well the total intensity of each punctum is preserved (Richardson-Lucy
    redistributes photons, it should not create or destroy them);
  * the background / haze level away from any punctum;
  * the localisation error of the punctum centroids.

    python scripts/validate_against_truth.py \
        --raw data/raw/synthetic_stack_01.ome.tif \
        --decon results/synthetic_stack_01_decon.ome.tif \
        --truth data/truth/synthetic_stack_01_truth.ome.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def load(path: Path) -> np.ndarray:
    """Read an OME-TIFF written by this pipeline back as (C, Z, Y, X)."""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
    if axes == "ZCYX":
        data = np.transpose(data, (1, 0, 2, 3))
    elif axes == "CZYX":
        pass
    elif data.ndim == 3:
        data = data[np.newaxis]
    else:
        raise SystemExit(f"unexpected axes {axes!r} in {path}")
    return data


def isolated_puncta(truth_channel: np.ndarray, min_separation: int, limit: int) -> list[tuple]:
    """Coordinates of truth puncta with no neighbour within ``min_separation``."""
    coords = np.argwhere(truth_channel > truth_channel.max() * 0.15)
    if len(coords) == 0:
        return []
    keep = []
    for point in coords:
        distances = np.abs(coords - point).max(axis=1)
        if np.count_nonzero(distances < min_separation) == 1:
            keep.append(tuple(int(v) for v in point))
        if len(keep) >= limit:
            break
    return keep


def profile_fwhm(values: np.ndarray, spacing: float) -> float:
    """FWHM of a 1D profile that peaks in the middle, above its own baseline."""
    baseline = float(np.min(values))
    peak = float(np.max(values))
    if peak <= baseline:
        return float("nan")
    half = baseline + (peak - baseline) / 2.0
    above = np.flatnonzero(values >= half)
    if len(above) < 1:
        return float("nan")
    left, right = above[0], above[-1]
    # Linear interpolation on each flank for sub-voxel precision.
    if left > 0 and values[left] != values[left - 1]:
        left = left - (values[left] - half) / (values[left] - values[left - 1])
    if right < len(values) - 1 and values[right] != values[right + 1]:
        right = right + (values[right] - half) / (values[right] - values[right + 1])
    return float(right - left) * spacing


def measure(volume: np.ndarray, points: list[tuple], voxel, half: int = 8) -> dict:
    """Mean lateral/axial FWHM and integrated intensity over the given puncta."""
    dz, _, dx = voxel
    nz, ny, nx = volume.shape
    lateral, axial, integrated = [], [], []

    for z, y, x in points:
        if not (half <= y < ny - half and half <= x < nx - half and 2 <= z < nz - 2):
            continue
        # Re-centre on the local maximum: deconvolution may shift a punctum by
        # a voxel, and measuring off-peak would inflate the width.
        box = volume[max(z - 2, 0):z + 3, y - 3:y + 4, x - 3:x + 4]
        dz_i, dy_i, dx_i = np.unravel_index(int(np.argmax(box)), box.shape)
        zc = max(z - 2, 0) + dz_i
        yc, xc = y - 3 + dy_i, x - 3 + dx_i
        if not (half <= yc < ny - half and half <= xc < nx - half):
            continue

        lateral.append(profile_fwhm(volume[zc, yc, xc - half:xc + half + 1].astype(float), dx))
        z0, z1 = max(zc - 5, 0), min(zc + 6, nz)
        axial.append(profile_fwhm(volume[z0:z1, yc, xc].astype(float), dz))
        integrated.append(float(volume[z0:z1, yc - half:yc + half + 1,
                                       xc - half:xc + half + 1].sum()))

    return {
        "n": len(lateral),
        "fwhm_lateral_um": float(np.nanmedian(lateral)) if lateral else float("nan"),
        "fwhm_axial_um": float(np.nanmedian(axial)) if axial else float("nan"),
        "integrated_median": float(np.median(integrated)) if integrated else float("nan"),
    }


def background_level(volume: np.ndarray, truth_channel: np.ndarray) -> float:
    """Median intensity in voxels far from every true punctum (the haze)."""
    from scipy.ndimage import maximum_filter

    near = maximum_filter(truth_channel > 0, size=(5, 15, 15))
    empty = volume[~near]
    return float(np.median(empty)) if empty.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--decon", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--voxel", type=float, nargs=3, default=[0.30, 0.095, 0.095])
    parser.add_argument("--max-puncta", type=int, default=60)
    parser.add_argument("--intensity-scale", type=float, default=1.0,
                        help="output.intensity_scale used for the run, to undo the gain")
    args = parser.parse_args()

    raw = load(Path(args.raw))
    decon = load(Path(args.decon))
    truth = load(Path(args.truth))
    voxel = tuple(args.voxel)

    if not (raw.shape == decon.shape == truth.shape):
        raise SystemExit(
            f"shape mismatch: raw {raw.shape}, decon {decon.shape}, truth {truth.shape}"
        )

    print(f"voxel (dz, dy, dx) = {voxel} um\n")
    header = (f"{'channel':>8} {'n':>4} {'FWHM lateral (um)':>26} "
              f"{'FWHM axial (um)':>26} {'background':>22}")
    print(header)
    print(f"{'':>8} {'':>4} {'raw -> decon':>26} {'raw -> decon':>26} {'raw -> decon':>22}")
    print("-" * len(header))

    for c in range(raw.shape[0]):
        points = isolated_puncta(truth[c], min_separation=14, limit=args.max_puncta)
        before = measure(raw[c], points, voxel)
        after = measure(decon[c], points, voxel)
        bg_before = background_level(raw[c], truth[c])
        bg_after = background_level(decon[c], truth[c]) / args.intensity_scale

        print(
            f"{c:>8} {before['n']:>4} "
            f"{before['fwhm_lateral_um']:>10.3f} -> {after['fwhm_lateral_um']:<12.3f} "
            f"{before['fwhm_axial_um']:>10.3f} -> {after['fwhm_axial_um']:<12.3f} "
            f"{bg_before:>8.1f} -> {bg_after:<10.1f}"
        )

    print("\nIntensity conservation (sum over the whole stack, gain removed):")
    for c in range(raw.shape[0]):
        raw_total = float(raw[c].astype(np.float64).sum())
        dec_total = float(decon[c].astype(np.float64).sum()) / args.intensity_scale
        print(f"  channel {c}: raw {raw_total:.4g} -> decon {dec_total:.4g} "
              f"(ratio {dec_total / raw_total:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
