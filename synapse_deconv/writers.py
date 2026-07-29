"""16-bit OME-TIFF output with the voxel calibration preserved.

Converting the float Richardson-Lucy result back to uint16 is the one place
where a careless choice silently ruins a comparative study, so the policy is
explicit:

``clip`` (default)
    Round to the nearest integer and clip to [0, 65535]. Intensities stay on the
    same absolute scale as the raw data, so puncta intensities are comparable
    across images and groups. The number of clipped voxels is reported.

``rescale_per_stack``
    Stretch each stack to the full 16-bit range. Convenient for display,
    but every image then has its own intensity scale -- never use it when the
    stacks are going to be compared to each other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

__all__ = ["ConversionReport", "to_uint16", "write_ome_tiff"]

_UINT16_MAX = 65535


@dataclass
class ConversionReport:
    """What the float -> uint16 conversion did, per channel."""

    policy: str
    clipped_voxels: int
    clipped_fraction: float
    scale_factor: float
    max_before: float

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "clipped_voxels": int(self.clipped_voxels),
            "clipped_fraction": float(self.clipped_fraction),
            "scale_factor": float(self.scale_factor),
            "max_before_conversion": float(self.max_before),
        }


def to_uint16(
    volume: np.ndarray,
    policy: str = "clip",
    intensity_scale: float = 1.0,
) -> tuple[np.ndarray, ConversionReport]:
    """Convert a non-negative float volume to uint16 under the given policy."""
    data = np.asarray(volume, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        n_bad = int((~np.isfinite(data)).sum())
        logger.warning("%d non-finite voxel(s) set to 0 before 16-bit conversion", n_bad)
        data = np.nan_to_num(data, nan=0.0, posinf=float(_UINT16_MAX), neginf=0.0)
    np.maximum(data, 0.0, out=data)

    max_before = float(data.max()) if data.size else 0.0

    if policy == "rescale_per_stack":
        scale = (_UINT16_MAX / max_before) if max_before > 0 else 1.0
        data = data * scale
        clipped = 0
    elif policy == "clip":
        scale = float(intensity_scale)
        if scale != 1.0:
            data = data * scale
        clipped = int((data > _UINT16_MAX).sum())
    else:
        raise ValueError(f"unknown bit-depth policy {policy!r}")

    out = np.rint(np.clip(data, 0, _UINT16_MAX)).astype(np.uint16)
    fraction = clipped / data.size if data.size else 0.0
    if clipped:
        # Gain that would have kept this stack inside the range, with 10% margin.
        suggested = 0.9 * _UINT16_MAX / max_before if max_before > 0 else 1.0
        logger.warning(
            "%d voxel(s) (%.4f%%) exceeded 65535 and were clipped; peak was %.1f. "
            "Set output.intensity_scale to <= %.3f (the SAME value for every image "
            "of the study) to keep the full dynamic range.",
            clipped, 100 * fraction, max_before * scale, suggested,
        )
    return out, ConversionReport(policy, clipped, fraction, scale, max_before)


def write_ome_tiff(
    path: str | Path,
    data: np.ndarray,
    *,
    voxel_size_um: tuple[float, float, float],
    channel_names: list[str] | None = None,
    description: str | None = None,
    software: str = "synapse_deconv",
) -> Path:
    """Write a ``(C, Z, Y, X)`` uint16 array as an OME-TIFF.

    The physical voxel size is stored in the OME-XML header (PhysicalSizeX/Y/Z),
    which is what Fiji, napari, Imaris and CellProfiler read back.
    """
    path = Path(path)
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError(f"expected a (C, Z, Y, X) array, got shape {data.shape}")
    if data.dtype != np.uint16:
        raise ValueError(f"refusing to write {data.dtype}; this pipeline stays 16-bit")

    dz, dy, dx = voxel_size_um
    # ZCYX keeps the channels interleaved per plane, which is what Fiji expects
    # from an OME-TIFF and what the FV1000 export produces.
    volume = np.ascontiguousarray(np.transpose(data, (1, 0, 2, 3)))

    metadata = {
        "axes": "ZCYX",
        "PhysicalSizeX": float(dx),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": float(dy),
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": float(dz),
        "PhysicalSizeZUnit": "µm",
    }
    if channel_names:
        metadata["Channel"] = {"Name": list(channel_names)}
    if description:
        metadata["Description"] = description

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        volume,
        ome=True,
        photometric="minisblack",
        metadata=metadata,
        software=software,
        # 1/um -> resolution tags, so plain TIFF readers also get the XY scale.
        resolution=(1.0 / dx, 1.0 / dy),
        resolutionunit="NONE",
        compression="zlib",
    )
    logger.debug("wrote %s (%s, %s)", path.name, volume.shape, volume.dtype)
    return path
