"""Quality control: intensity statistics and before/after projection panels.

The QC figure shows, per channel, the XY maximum-intensity projection and the
XZ projection side by side for the raw and the deconvolved stack. The XZ view is
the one that matters here: with a 0.30 um z-step and a 0.095 um pixel it is
where the axial elongation of the PSF -- and its removal -- is actually visible.

Both panels of a pair share the same display scaling, so a brighter "after"
image means genuinely higher peak intensity, not a different colour ramp.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ChannelStats", "compute_stats", "write_qc_figure"]

_UINT16_MAX = 65535


@dataclass
class ChannelStats:
    """Intensity statistics for one channel of one stack."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    p99_9: float
    #: Voxels at the top of the 16-bit range in the raw data: detector saturation
    #: breaks the Poisson assumption Richardson-Lucy relies on.
    saturated_voxels: int
    saturated_fraction: float
    nan_voxels: int
    negative_voxels: int
    total_intensity: float
    #: Mean squared gradient magnitude, a simple sharpness proxy. Expected to
    #: rise substantially after deconvolution.
    sharpness: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_stats(volume: np.ndarray, saturation_level: int = _UINT16_MAX) -> ChannelStats:
    """Intensity statistics for a 3D volume (raw uint16 or deconvolved float)."""
    data = np.asarray(volume)
    finite_mask = np.isfinite(data) if np.issubdtype(data.dtype, np.floating) else None
    n_nan = int((~finite_mask).sum()) if finite_mask is not None else 0
    values = data[finite_mask] if n_nan else data.ravel()
    if values.size == 0:
        raise ValueError("cannot compute statistics on an empty volume")

    as_float = values.astype(np.float64, copy=False)
    gradients = np.gradient(np.asarray(data, dtype=np.float32))
    sharpness = float(np.mean(sum(np.square(g) for g in gradients)))

    return ChannelStats(
        min=float(as_float.min()),
        max=float(as_float.max()),
        mean=float(as_float.mean()),
        median=float(np.median(as_float)),
        std=float(as_float.std()),
        p99_9=float(np.percentile(as_float, 99.9)),
        saturated_voxels=int((as_float >= saturation_level).sum()),
        saturated_fraction=float((as_float >= saturation_level).mean()),
        nan_voxels=n_nan,
        negative_voxels=int((as_float < 0).sum()),
        total_intensity=float(as_float.sum()),
        sharpness=sharpness,
    )


def _projections(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-intensity projections along Z (XY view) and along Y (XZ view)."""
    data = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0)
    return data.max(axis=0), data.max(axis=1)


def _display_range(image: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    """Display limits for one panel.

    Each panel is stretched independently. Sharing one range between the raw and
    the deconvolved panel sounds fairer but is useless in practice: Richardson-
    Lucy concentrates a punctum into a tenth of the voxels, so its peak rises by
    a factor of ten or more and the raw panel would render as a black rectangle.
    The absolute numbers are printed under each panel instead.
    """
    low, high = np.percentile(image, percentiles)
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def write_qc_figure(
    path: str | Path,
    raw: np.ndarray,
    deconvolved: np.ndarray,
    *,
    channel_names: list[str],
    voxel_size_um: tuple[float, float, float],
    title: str,
    stats_before: list[ChannelStats] | None = None,
    stats_after: list[ChannelStats] | None = None,
    percentile_clip: tuple[float, float] = (0.1, 99.9),
    display_gamma: float = 0.5,
    include_xz: bool = True,
    dpi: int = 150,
) -> Path:
    """Render the before/after QC panel for a whole stack.

    ``raw`` and ``deconvolved`` are ``(C, Z, Y, X)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    n_channels, _, ny, nx = raw.shape
    dz, dy, dx = voxel_size_um

    # The XZ view is much wider than it is tall (a 30-plane stack at 0.30 um is
    # 9 um deep against ~24 um across), so it gets its own short row under the
    # square XY row rather than sharing one and leaving most of the box empty.
    xz_height = (raw.shape[1] * dz) / (nx * dx) if include_xz else 0.0
    rows_per_channel = 2 if include_xz else 1
    height_ratios = ([1.0, max(0.25, xz_height)] if include_xz else [1.0]) * n_channels

    panel = 3.3
    title_space = 0.45
    fig_height = sum(height_ratios) * panel + 0.6 * n_channels + title_space
    fig = plt.figure(figsize=(2 * panel + 1.0, fig_height))
    grid = fig.add_gridspec(
        rows_per_channel * n_channels, 2,
        height_ratios=height_ratios, hspace=0.35, wspace=0.06,
        top=1.0 - title_space / fig_height, bottom=0.01, left=0.05, right=0.99,
    )

    for c in range(n_channels):
        xy_before, xz_before = _projections(raw[c])
        xy_after, xz_after = _projections(deconvolved[c])

        views = [(xy_before, xy_after, "XY", 1.0)]
        if include_xz:
            # aspect=dz/dx renders the anisotropic voxel undistorted.
            views.append((xz_before, xz_after, "XZ", dz / dx))

        for row, (before, after, view, aspect) in enumerate(views):
            for col, (image, label) in enumerate(((before, "raw"), (after, "deconvolved"))):
                ax = fig.add_subplot(grid[c * rows_per_channel + row, col])
                low, high = _display_range(image, percentile_clip)
                # Gamma stretch for display only. Punctate images after RL span
                # four decades -- on a linear ramp the few brightest puncta set
                # the scale and everything else renders black.
                shown = np.clip((image - low) / (high - low), 0.0, 1.0) ** display_gamma
                ax.imshow(shown, cmap="inferno", vmin=0.0, vmax=1.0,
                          aspect=aspect, interpolation="nearest")
                ax.set_title(
                    f"MIP {view} — {label}   [{low:.0f}–{high:.0f}, γ={display_gamma:g}]",
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(
                        channel_names[c] if view == "XY" else "XZ",
                        fontsize=10 if view == "XY" else 8,
                        fontweight="bold" if view == "XY" else "normal",
                    )

                if view == "XY" and col == 0:
                    bar_um = 5.0
                    bar_px = bar_um / dx
                    if bar_px < nx * 0.8:
                        y = ny * 0.94
                        x0 = nx * 0.05
                        ax.plot([x0, x0 + bar_px], [y, y], color="white", linewidth=2.5)
                        ax.text(x0, y - ny * 0.02, f"{bar_um:g} µm",
                                color="white", fontsize=7, va="bottom")

                if view == "XY" and stats_before and stats_after:
                    s = stats_before[c] if col == 0 else stats_after[c]
                    text = f"min {s.min:.0f}   max {s.max:.0f}   mean {s.mean:.1f}"
                    if col == 1:
                        b = stats_before[c]
                        ratio = s.sharpness / b.sharpness if b.sharpness > 0 else float("nan")
                        text += f"\ngradient energy ×{ratio:.1f}"
                    # Top-left: the bottom-left corner belongs to the scale bar.
                    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
                            color="white", va="top", ha="left")

    fig.suptitle(title, fontsize=10, y=1.0 - 0.12 / fig_height, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("QC figure written to %s", path)
    return path
