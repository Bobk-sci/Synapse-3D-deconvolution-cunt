"""QC overlays for punctum detection and synapse colocalisation.

One row per channel: the MIP with every detected punctum circled, then the same
MIP with only the puncta that form a synapse highlighted. A final row shows the
three channels merged with the synapse positions, which is where an over- or
under-detected channel becomes obvious at a glance.

Circles are drawn at the projected centroid, so a punctum hidden behind a
brighter one in the projection still shows a marker -- the count on the figure
matches the count in the CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["write_detection_overlay"]


def _stretch(image: np.ndarray, percentiles: tuple[float, float], gamma: float) -> np.ndarray:
    low, high = np.percentile(image, percentiles)
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low), 0.0, 1.0) ** gamma


def write_detection_overlay(
    path: str | Path,
    stack: np.ndarray,
    detections: dict[str, Any],
    inside_flags: dict[str, np.ndarray],
    *,
    channel_names: list[str],
    voxel_size_um: tuple[float, float, float],
    synaptic_indices: dict[str, set[int]],
    synapse_points_um: dict[str, np.ndarray],
    title: str,
    percentile_clip: tuple[float, float] = (0.1, 99.9),
    display_gamma: float = 0.5,
    dpi: int = 150,
) -> Path:
    """Render the detection/colocalisation QC panel.

    Args:
        stack: (C, Z, Y, X) image the detection ran on.
        detections: channel name -> DetectionResult.
        inside_flags: channel name -> boolean mask of puncta inside the ROI.
        synaptic_indices: channel name -> indices of the puncta that were paired.
            Taken straight from the pairing result, so the counts printed on the
            figure are the counts written to the CSV.
        synapse_points_um: {"excitatory": (N,3), "inhibitory": (M,3)} midpoints.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    n_channels = len(channel_names)
    _, dy, dx = voxel_size_um

    n_rows = n_channels + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, 5.2 * n_rows), squeeze=False)

    for row, name in enumerate(channel_names):
        detection = detections[name]
        inside = inside_flags[name]
        mip = np.asarray(stack[row], dtype=np.float32).max(axis=0)
        shown = _stretch(mip, percentile_clip, display_gamma)

        for col, only_synaptic in enumerate((False, True)):
            ax = axes[row][col]
            ax.imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            if detection.count:
                selection = inside.copy()
                if only_synaptic:
                    paired = np.zeros(detection.count, dtype=bool)
                    listed = np.fromiter(synaptic_indices.get(name, ()), dtype=int)
                    if listed.size:
                        paired[listed] = True
                    selection = selection & paired
                ys = detection.centroids_um[selection, 1] / dy
                xs = detection.centroids_um[selection, 2] / dx
                ax.scatter(xs, ys, s=26, facecolors="none",
                           edgecolors="#ffb000" if not only_synaptic else "#00d0ff",
                           linewidths=0.7)
                label = "all detected" if not only_synaptic else "forming a synapse"
                ax.set_title(f"{name} — {label}: {int(selection.sum())}", fontsize=9)
            else:
                ax.set_title(f"{name} — no punctum detected", fontsize=9)

            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")

    # --- merged view with the synapses ------------------------------------
    merged = np.zeros(stack.shape[2:] + (3,), dtype=np.float32)
    for channel in range(min(3, n_channels)):
        mip = np.asarray(stack[channel], dtype=np.float32).max(axis=0)
        merged[..., channel] = _stretch(mip, percentile_clip, display_gamma)

    for col, (key, colour, label) in enumerate((
        ("excitatory", "#00ff88", "excitatory"),
        ("inhibitory", "#ff4fd8", "inhibitory"),
    )):
        ax = axes[n_channels][col]
        ax.imshow(merged, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        points = synapse_points_um.get(key, np.zeros((0, 3)))
        if len(points):
            ax.scatter(points[:, 2] / dx, points[:, 1] / dy, s=44, facecolors="none",
                       edgecolors=colour, linewidths=1.0)
        ax.set_title(f"merged RGB — {label} synapses: {len(points)}", fontsize=9)
        if col == 0:
            ax.set_ylabel("synapses", fontsize=10, fontweight="bold")

    # Scale bar on the first panel.
    ny, nx = stack.shape[2], stack.shape[3]
    bar_px = 5.0 / dx
    if bar_px < nx * 0.8:
        ax = axes[0][0]
        ax.plot([nx * 0.05, nx * 0.05 + bar_px], [ny * 0.95, ny * 0.95],
                color="white", linewidth=2.5)
        ax.text(nx * 0.05, ny * 0.93, "5 µm", color="white", fontsize=8, va="bottom")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("detection QC written to %s", path)
    return path
