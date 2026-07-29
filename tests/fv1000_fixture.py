"""Build a synthetic Olympus FluoView .oif dataset for the reader tests.

An .oif acquisition is a UTF-16LE INI-style text file next to a ``<name>.files``
directory holding one TIFF per plane, named ``s_C00nZ00m.tif``. Reproducing that
layout lets the FV1000 metadata parsing be tested without a proprietary sample
file, which is the part of the pipeline most likely to break on real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

#: Section order does not matter to the parser, but keeping the real one makes
#: the fixture readable next to an actual FV1000 export.
_TEMPLATE_SECTIONS = ("ProfileSaveInfo", "Axis Parameter Common")


def _render(sections: dict[str, dict[str, object]]) -> str:
    lines = []
    for name, entries in sections.items():
        lines.append(f"[{name}]")
        for key, value in entries.items():
            lines.append(f"{key}={value}")
        lines.append("")
    return "\r\n".join(lines)


def write_oif(
    directory: Path,
    stem: str = "scan",
    *,
    shape: tuple[int, int, int, int] = (2, 8, 32, 32),   # (C, Z, Y, X)
    xy_um: float = 0.095,
    z_um: float = 0.30,
    z_unit: str = "um",
    emission_nm: tuple[float, ...] = (421.0, 519.0),
    excitation_nm: tuple[float, ...] = (405.0, 488.0),
    channel_names: tuple[str, ...] = ("CH1", "CH2"),
    pinhole_nm: float = 100000.0,
    fill: np.ndarray | None = None,
) -> Path:
    """Write a complete .oif dataset and return the path of the main file."""
    n_c, n_z, n_y, n_x = shape
    directory.mkdir(parents=True, exist_ok=True)
    main_path = directory / f"{stem}.oif"
    storage = directory / f"{stem}.oif.files"
    storage.mkdir(exist_ok=True)

    # Olympus records the centre of the first and last sample, so the span
    # covers (n - 1) intervals.
    unit_scale = {"um": 1.0, "nm": 1000.0}[z_unit]
    x_end = xy_um * (n_x - 1)
    y_end = xy_um * (n_y - 1)
    z_end = z_um * (n_z - 1) * unit_scale

    sections: dict[str, dict[str, object]] = {
        "ProfileSaveInfo": {"Name": f'"{stem}"', "Version": '"2.1.1.5"'},
        "Version Info": {"SystemName": '"FLUOVIEW"', "SystemVersion": '"2.1.1.5"',
                         "FileVersion": '"1.0.0.0"', "DeviceName": '"FV1000"'},
        "Axis Parameter Common": {"AxisOrder": '"XYZC"'},
        "Axis 0 Parameters Common": {
            "AxisCode": '"X"', "AxisName": '"X"', "MaxSize": n_x,
            "StartPosition": 0.0, "EndPosition": x_end,
            "UnitName": '"um"', "PixUnit": '"um"',
        },
        "Axis 1 Parameters Common": {
            "AxisCode": '"Y"', "AxisName": '"Y"', "MaxSize": n_y,
            "StartPosition": 0.0, "EndPosition": y_end,
            "UnitName": '"um"', "PixUnit": '"um"',
        },
        "Axis 2 Parameters Common": {
            "AxisCode": '"C"', "AxisName": '"Ch"', "MaxSize": n_c,
            "StartPosition": 1.0, "EndPosition": float(n_c), "UnitName": '""',
        },
        "Axis 3 Parameters Common": {
            "AxisCode": '"Z"', "AxisName": '"Z"', "MaxSize": n_z,
            "StartPosition": 0.0, "EndPosition": z_end,
            "UnitName": f'"{z_unit}"', "PixUnit": f'"{z_unit}"',
        },
        "Reference Image Parameter": {
            "ImageWidth": n_x, "ImageHeight": n_y,
            "WidthConvertValue": xy_um, "WidthUnit": '"um"',
            "HeightConvertValue": xy_um, "HeightUnit": '"um"',
            "ValidBitCounts": 16,
        },
    }
    # Unused axes still appear in a real file; keep them so the parser sees the
    # same shape of input.
    for index in range(4, 8):
        sections[f"Axis {index} Parameters Common"] = {
            "AxisCode": '""', "MaxSize": 0, "StartPosition": 0.0,
            "EndPosition": 0.0, "UnitName": '""',
        }
    for index in range(n_c):
        sections[f"Channel {index + 1} Parameters"] = {
            "CH Name": f'"{channel_names[index]}"',
            "DyeName": f'"Dye{index + 1}"',
            "EmissionWavelength": emission_nm[index],
            "ExcitationWavelength": excitation_nm[index],
            "PinholeDiameter": pinhole_nm,
        }

    # Olympus writes little-endian UTF-16 with a BOM; the parser keys off it.
    main_path.write_bytes(b"\xff\xfe" + _render(sections).encode("utf-16-le"))

    if fill is None:
        rng = np.random.default_rng(5)
        fill = rng.integers(0, 3000, size=(n_c, n_z, n_y, n_x), dtype=np.uint16)

    for c in range(n_c):
        for z in range(n_z):
            tifffile.imwrite(
                str(storage / f"s_C{c + 1:03d}Z{z + 1:03d}.tif"),
                np.ascontiguousarray(fill[c, z]),
            )
    return main_path
