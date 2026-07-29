"""Readers for Olympus FluoView (.oib/.oif) and OME-TIFF z-stacks.

Both readers return the same :class:`ImageStack`: a ``(C, Z, Y, X)`` uint16
array plus the physical voxel size in micrometres. The voxel size is *always*
read from the file when present -- XY and Z are kept separate and never assumed
equal, which is the whole point for a 0.095 / 0.30 um anisotropic stack.

Olympus does not use a single convention for the axis units, so every position
value is converted through :func:`_to_micrometres` using the ``UnitName`` field
that accompanies it (Z positions are frequently stored in nanometres while XY
are in micrometres).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ImageStack", "ReadError", "read_stack", "is_supported"]

#: Multipliers onto micrometres.
_UNIT_TO_UM = {
    "m": 1e6, "meter": 1e6, "metre": 1e6,
    "mm": 1e3, "millimeter": 1e3,
    "um": 1.0, "µm": 1.0, "micron": 1.0, "micrometer": 1.0, "micrometre": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3,
    "pm": 1e-6,
}

_OLYMPUS_SUFFIXES = {".oib", ".oif"}
_TIFF_SUFFIXES = {".tif", ".tiff", ".ome.tif", ".ome.tiff"}


class ReadError(RuntimeError):
    """Raised when a file cannot be read or is not a usable 3D multi-channel stack."""


@dataclass
class ChannelInfo:
    """Per-channel metadata found in the file (any field may be None)."""

    name: str | None = None
    emission_nm: float | None = None
    excitation_nm: float | None = None
    dye: str | None = None
    pinhole_um: float | None = None


@dataclass
class ImageStack:
    """A multi-channel 3D stack with its physical calibration."""

    data: np.ndarray                    # (C, Z, Y, X), uint16
    voxel_size_um: tuple[float, float, float]   # (dz, dy, dx)
    path: Path
    channels: list[ChannelInfo] = field(default_factory=list)
    voxel_size_source: str = "metadata"          # "metadata" | "fallback" | "config"
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape[1:])       # type: ignore[return-value]

    def describe(self) -> str:
        dz, dy, dx = self.voxel_size_um
        c, z, y, x = self.data.shape
        return (
            f"{self.path.name}: {c} channel(s), {z}x{y}x{x} (ZYX), {self.data.dtype}, "
            f"voxel dz={dz:.4f} dy={dy:.4f} dx={dx:.4f} um "
            f"(anisotropy z/xy = {dz / dy:.2f}, source: {self.voxel_size_source})"
        )


def is_supported(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in _OLYMPUS_SUFFIXES or any(
        name.endswith(s) for s in _TIFF_SUFFIXES
    )


def _to_micrometres(value: float, unit: str | None) -> float:
    """Convert ``value`` expressed in ``unit`` to micrometres.

    Unknown or missing units are treated as micrometres, which is the Olympus
    default for lateral axes; the caller logs a warning in that case.
    """
    if unit is None:
        return float(value)
    key = str(unit).strip().lower()
    return float(value) * _UNIT_TO_UM.get(key, 1.0)


def _as_float(value: Any) -> float | None:
    """Coerce an Olympus metadata field (often a string) to a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group()) if match else None


# --------------------------------------------------------------------------
# Olympus FluoView
# --------------------------------------------------------------------------

def _olympus_axis_sections(mainfile: dict) -> dict[str, dict]:
    """Map axis code ('X', 'Y', 'Z', 'C', 'T') to its parameter section."""
    axes: dict[str, dict] = {}
    for key, section in mainfile.items():
        if not (isinstance(section, dict) and key.startswith("Axis ")
                and "Parameters Common" in key):
            continue
        code = str(section.get("AxisCode", "")).strip().upper()
        if code:
            axes[code] = section
    return axes


def _olympus_axis_spacing(section: dict, divisor_mode: str) -> float | None:
    """Spacing between consecutive samples along one Olympus axis, in um."""
    size = _as_float(section.get("MaxSize"))
    start = _as_float(section.get("StartPosition"))
    end = _as_float(section.get("EndPosition"))
    if size is None or start is None or end is None or size < 2:
        return None
    unit = section.get("UnitName") or section.get("PixUnit")
    span = abs(_to_micrometres(end, unit) - _to_micrometres(start, unit))
    divisor = (size - 1) if divisor_mode == "n_minus_1" else size
    if divisor <= 0:
        return None
    spacing = span / divisor
    return spacing if spacing > 0 else None


def _olympus_channels(mainfile: dict) -> list[ChannelInfo]:
    """Per-channel info from the ``Channel N Parameters`` sections, in order."""
    found: list[tuple[int, ChannelInfo]] = []
    for key, section in mainfile.items():
        match = re.fullmatch(r"Channel (\d+) Parameters", str(key))
        if not match or not isinstance(section, dict):
            continue
        pinhole_nm = _as_float(section.get("PinholeDiameter"))
        found.append((
            int(match.group(1)),
            ChannelInfo(
                name=(section.get("CH Name") or section.get("DyeName") or None),
                emission_nm=_as_float(section.get("EmissionWavelength")),
                excitation_nm=_as_float(section.get("ExcitationWavelength")),
                dye=section.get("DyeName") or None,
                # PinholeDiameter is stored in nanometres by the FV1000.
                pinhole_um=(pinhole_nm / 1000.0) if pinhole_nm else None,
            ),
        ))
    return [info for _, info in sorted(found, key=lambda item: item[0])]


def read_olympus(path: Path, *, spacing_divisor: str = "n_minus_1") -> ImageStack:
    """Read an Olympus .oib (single container) or .oif (folder-based) stack."""
    try:
        import oiffile
    except ImportError as exc:      # pragma: no cover - environment issue
        raise ReadError(
            "reading .oib/.oif requires the 'oiffile' package (pip install oiffile)"
        ) from exc

    warnings: list[str] = []
    try:
        with oiffile.OifFile(str(path)) as oif:
            mainfile = dict(oif.mainfile)
            axes = oif.axes
            data = oif.asarray()
    except Exception as exc:        # oiffile raises a wide range of errors
        raise ReadError(f"could not open Olympus file: {exc}") from exc

    data, warns = _normalise_axes(np.asarray(data), axes, path)
    warnings.extend(warns)

    axis_sections = _olympus_axis_sections(mainfile)

    # Lateral: 'Reference Image Parameter' gives the calibration directly and is
    # the most reliable field; fall back to the X/Y axis position span.
    ref = mainfile.get("Reference Image Parameter", {})
    dx = dy = None
    if isinstance(ref, dict):
        wcv, hcv = _as_float(ref.get("WidthConvertValue")), _as_float(ref.get("HeightConvertValue"))
        if wcv:
            dx = _to_micrometres(wcv, ref.get("WidthUnit"))
        if hcv:
            dy = _to_micrometres(hcv, ref.get("HeightUnit"))
    if dx is None and "X" in axis_sections:
        dx = _olympus_axis_spacing(axis_sections["X"], spacing_divisor)
    if dy is None and "Y" in axis_sections:
        dy = _olympus_axis_spacing(axis_sections["Y"], spacing_divisor)

    dz = None
    if "Z" in axis_sections:
        dz = _olympus_axis_spacing(axis_sections["Z"], spacing_divisor)
    if dz is None:
        warnings.append("no usable Z axis calibration in the Olympus metadata")

    channels = _olympus_channels(mainfile)
    if channels and len(channels) != data.shape[0]:
        warnings.append(
            f"metadata describes {len(channels)} channel(s) but the array has "
            f"{data.shape[0]}; per-channel metadata will be ignored"
        )
        channels = []

    return ImageStack(
        data=data,
        voxel_size_um=(dz, dy, dx),     # type: ignore[arg-type]  # None handled by caller
        path=path,
        channels=channels,
        raw_metadata={"format": "olympus", "axes": axes,
                      "sections": sorted(str(k) for k in mainfile)},
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# OME-TIFF
# --------------------------------------------------------------------------

def _ome_physical_sizes(tif) -> tuple[float | None, float | None, float | None, list[ChannelInfo]]:
    """Pull PhysicalSizeX/Y/Z and channel entries out of the OME-XML header."""
    import xml.etree.ElementTree as ET

    description = getattr(tif, "ome_metadata", None)
    if not description:
        return None, None, None, []
    try:
        root = ET.fromstring(description)
    except ET.ParseError:
        return None, None, None, []

    ns = {"ome": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    pixels = root.find(".//ome:Pixels", ns) if ns else root.find(".//Pixels")
    if pixels is None:
        return None, None, None, []

    def size(axis: str) -> float | None:
        value = _as_float(pixels.get(f"PhysicalSize{axis}"))
        if value is None:
            return None
        return _to_micrometres(value, pixels.get(f"PhysicalSize{axis}Unit", "um"))

    channels: list[ChannelInfo] = []
    entries = pixels.findall("ome:Channel", ns) if ns else pixels.findall("Channel")
    for entry in entries:
        channels.append(ChannelInfo(
            name=entry.get("Name"),
            emission_nm=_as_float(entry.get("EmissionWavelength")),
            excitation_nm=_as_float(entry.get("ExcitationWavelength")),
            dye=entry.get("Fluor"),
            pinhole_um=_as_float(entry.get("PinholeSize")),
        ))
    return size("Z"), size("Y"), size("X"), channels


def _tiff_resolution_um(tif) -> tuple[float | None, float | None]:
    """Lateral pixel size from the TIFF resolution tags, or (None, None).

    The tags are only meaningful when a real unit accompanies them. A plain
    TIFF carries ``XResolution = 1`` with ``ResolutionUnit = NONE``, which means
    "no calibration" -- taking it at face value would silently assert a 1 um
    pixel and skew every PSF in the batch.
    """
    page = tif.pages[0]
    unit_tag = page.tags.get("ResolutionUnit")
    unit_code = int(getattr(unit_tag, "value", 1) or 1)
    imagej = tif.imagej_metadata or {}

    if unit_code == 2:              # inch
        unit_um = 25400.0
    elif unit_code == 3:            # centimetre
        unit_um = 10000.0
    elif imagej.get("unit"):
        # ImageJ stores the unit separately and leaves ResolutionUnit at NONE.
        unit_um = _to_micrometres(1.0, imagej["unit"])
    else:
        return None, None

    sizes: list[float | None] = []
    for tag_name in ("XResolution", "YResolution"):
        entry = page.tags.get(tag_name)
        value = getattr(entry, "value", None)
        if not value or len(value) != 2 or not value[1] or not value[0]:
            sizes.append(None)
            continue
        pixels_per_unit = value[0] / value[1]
        sizes.append(unit_um / pixels_per_unit if pixels_per_unit > 0 else None)
    return sizes[0], sizes[1]


def _axes_from_imagej(axes: str, shape: tuple[int, ...], imagej: dict | None) -> str:
    """Correct tifffile's axis guess for a non-OME TIFF using ImageJ's counts.

    Without OME-XML, tifffile has to guess what a stack of pages means and
    labels a plain multi-plane stack ``CYX``. ImageJ does record the real
    split in its ``slices`` / ``channels`` / ``frames`` keys, so use those; with
    no metadata at all, a stack of pages is a z-stack as far as this pipeline
    is concerned.
    """
    imagej = imagej or {}
    slices = int(imagej.get("slices", 0) or 0)
    channels = int(imagej.get("channels", 0) or 0)

    if len(shape) == 3 and axes.endswith("YX"):
        # One non-spatial axis: it is Z unless ImageJ says it holds channels.
        if channels > 1 and slices <= 1:
            return "CYX"
        return "ZYX"
    if len(shape) == 4 and axes.endswith("YX"):
        # ImageJ writes hyperstacks plane-interleaved as ZCYX.
        leading = shape[:2]
        if (slices, channels) == leading:
            return "ZCYX"
        if (channels, slices) == leading:
            return "CZYX"
        return axes if set(axes[:2]) == {"Z", "C"} else "ZCYX"
    return axes


def read_ome_tiff(path: Path) -> ImageStack:
    """Read an OME-TIFF (or a plain TIFF stack with usable resolution tags)."""
    import tifffile

    warnings: list[str] = []
    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            data = series.asarray()
            axes = series.axes
            dz, dy, dx = None, None, None
            channels: list[ChannelInfo] = []
            if tif.is_ome:
                dz, dy, dx, channels = _ome_physical_sizes(tif)
            if dx is None or dy is None:
                lateral = _tiff_resolution_um(tif)
                dx = dx if dx is not None else lateral[0]
                dy = dy if dy is not None else lateral[1]
            if dz is None and tif.imagej_metadata:
                spacing = _as_float(tif.imagej_metadata.get("spacing"))
                unit = tif.imagej_metadata.get("unit")
                if spacing:
                    dz = _to_micrometres(spacing, unit)
            if not tif.is_ome:
                axes = _axes_from_imagej(axes, data.shape, tif.imagej_metadata)
    except Exception as exc:
        raise ReadError(f"could not open TIFF: {exc}") from exc

    data, warns = _normalise_axes(np.asarray(data), axes, path)
    warnings.extend(warns)

    if channels and len(channels) != data.shape[0]:
        warnings.append(
            f"OME header describes {len(channels)} channel(s) but the array has "
            f"{data.shape[0]}; per-channel metadata will be ignored"
        )
        channels = []

    return ImageStack(
        data=data,
        voxel_size_um=(dz, dy, dx),     # type: ignore[arg-type]
        path=path,
        channels=channels,
        raw_metadata={"format": "ome-tiff", "axes": axes},
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

def _normalise_axes(data: np.ndarray, axes: str, path: Path) -> tuple[np.ndarray, list[str]]:
    """Reorder an arbitrary axis layout to ``(C, Z, Y, X)``.

    Time points and any other extra leading axis are reduced to their first
    element with a warning: this pipeline deconvolves single z-stacks.
    """
    warnings: list[str] = []
    axes = (axes or "").upper()

    if data.ndim != len(axes):
        # Trust the array over a stale axis string.
        if data.ndim == 3:
            axes = "ZYX"
        elif data.ndim == 4:
            axes = "CZYX"
        else:
            raise ReadError(
                f"unsupported array shape {data.shape} with axes {axes!r}; "
                "expected a 3D or 4D (channel + z) stack"
            )
        warnings.append(f"axis string did not match array rank; assuming {axes}")

    # A plain (non-OME, non-ImageJ) TIFF gets the placeholder codes 'Q'/'I'/'S'
    # from tifffile because the format simply does not record what the pages
    # mean. For this pipeline the pages of such a file are z-planes.
    if "Z" not in axes:
        unknown = [i for i, code in enumerate(axes) if code not in "CZYXT"]
        if len(unknown) == 1:
            i = unknown[0]
            axes = axes[:i] + "Z" + axes[i + 1:]
            warnings.append(
                f"axis {i} had no declared meaning; interpreted as Z "
                f"({data.shape[i]} planes)"
            )

    # Collapse anything that is not C/Z/Y/X (T, and Olympus 'L'/lambda) to index 0.
    for i, code in enumerate(axes):
        if code not in "CZYX" and data.shape[i] > 1:
            warnings.append(
                f"axis {code!r} has {data.shape[i]} elements; only the first is processed"
            )
    keep = [slice(None) if code in "CZYX" else 0 for code in axes]
    data = data[tuple(keep)]
    axes = "".join(code for code in axes if code in "CZYX")

    if "C" not in axes:
        data = data[np.newaxis]
        axes = "C" + axes
    if "Z" not in axes:
        raise ReadError(f"{path.name}: no Z axis found (axes={axes!r}); not a z-stack")
    for required in "YX":
        if required not in axes:
            raise ReadError(f"{path.name}: missing {required} axis (axes={axes!r})")

    order = [axes.index(code) for code in "CZYX"]
    data = np.transpose(data, order)
    return np.ascontiguousarray(data), warnings


def _resolve_voxel_size(
    stack: ImageStack,
    *,
    source: str,
    fallback_xy_um: float,
    fallback_z_um: float,
    tolerance_warn_ratio: float,
) -> ImageStack:
    """Fill in missing voxel sizes from the config and sanity-check the values."""
    dz, dy, dx = stack.voxel_size_um
    used_fallback = []

    if source == "config":
        dz, dy, dx = fallback_z_um, fallback_xy_um, fallback_xy_um
        stack.voxel_size_source = "config"
    else:
        if dx is None or not np.isfinite(dx) or dx <= 0:
            dx = fallback_xy_um
            used_fallback.append("X")
        if dy is None or not np.isfinite(dy) or dy <= 0:
            dy = fallback_xy_um
            used_fallback.append("Y")
        if dz is None or not np.isfinite(dz) or dz <= 0:
            dz = fallback_z_um
            used_fallback.append("Z")
        stack.voxel_size_source = "fallback" if used_fallback else "metadata"

    if used_fallback:
        stack.warnings.append(
            f"voxel size missing from metadata for {'/'.join(used_fallback)}; "
            f"using configured fallback (XY={fallback_xy_um} um, Z={fallback_z_um} um)"
        )

    # Guard against unit-parsing mistakes: a factor-1000 slip is the classic one.
    for label, value, expected in (("XY", dx, fallback_xy_um), ("Z", dz, fallback_z_um)):
        if expected > 0 and abs(value - expected) / expected > tolerance_warn_ratio:
            stack.warnings.append(
                f"{label} voxel size read from metadata ({value:.5f} um) differs from the "
                f"expected {expected} um by more than {tolerance_warn_ratio:.0%}; "
                "verify the acquisition settings and the metadata units"
            )
    if abs(dx - dy) / max(dx, dy) > 0.01:
        stack.warnings.append(
            f"non-square pixels: dx={dx:.5f} um, dy={dy:.5f} um (both are honoured)"
        )

    stack.voxel_size_um = (float(dz), float(dy), float(dx))
    return stack


def read_stack(
    path: str | Path,
    *,
    voxel_size_source: str = "metadata",
    fallback_xy_um: float = 0.095,
    fallback_z_um: float = 0.30,
    spacing_divisor: str = "n_minus_1",
    tolerance_warn_ratio: float = 0.25,
) -> ImageStack:
    """Read any supported stack and return it calibrated and axis-normalised."""
    path = Path(path)
    if not path.exists():
        raise ReadError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in _OLYMPUS_SUFFIXES:
        stack = read_olympus(path, spacing_divisor=spacing_divisor)
    elif suffix in {".tif", ".tiff"}:
        stack = read_ome_tiff(path)
    else:
        raise ReadError(f"unsupported file extension {suffix!r} for {path.name}")

    if stack.data.dtype == np.uint8:
        stack.warnings.append(
            "input is 8-bit; the pipeline works in 16-bit and will widen it, but "
            "the lost dynamic range cannot be recovered"
        )
        stack.data = stack.data.astype(np.uint16)
    elif stack.data.dtype != np.uint16:
        if np.issubdtype(stack.data.dtype, np.floating):
            if not np.all(np.isfinite(stack.data)):
                stack.warnings.append("input contains NaN/Inf; those voxels are set to 0")
                stack.data = np.nan_to_num(stack.data, nan=0.0, posinf=65535.0, neginf=0.0)
            stack.warnings.append(f"input is {stack.data.dtype}; converting to uint16 by clipping")
            stack.data = np.clip(np.rint(stack.data), 0, 65535).astype(np.uint16)
        else:
            stack.warnings.append(f"input dtype {stack.data.dtype}; converting to uint16")
            stack.data = np.clip(stack.data, 0, 65535).astype(np.uint16)

    return _resolve_voxel_size(
        stack,
        source=voxel_size_source,
        fallback_xy_um=fallback_xy_um,
        fallback_z_um=fallback_z_um,
        tolerance_warn_ratio=tolerance_warn_ratio,
    )
