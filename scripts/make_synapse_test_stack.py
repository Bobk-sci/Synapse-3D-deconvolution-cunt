"""Generate a synthetic triple-labelled stack with a KNOWN synapse count.

Mimics a deconvolved acquisition: Bassoon (presynaptic), PSD-95 (excitatory
post) and Gephyrin (inhibitory post). Excitatory and inhibitory synapses are
built as apposed pairs separated by a realistic centre-to-centre distance, and
a controlled number of orphan puncta is added to each channel so that a
detector cannot score well simply by pairing everything.

The ground truth is written next to the stack as JSON, which is what
``validate_counts.py`` compares the pipeline against.

    python scripts/make_synapse_test_stack.py --out data/synth --n-stacks 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synapse_deconv.writers import write_ome_tiff

CHANNELS = ["Bassoon", "PSD95", "Gephyrin"]


def _add_punctum(volume, centre_um, voxel, amplitude, sigma_um, rng):
    """Add an isotropic Gaussian blob at a physical position."""
    dz, dy, dx = voxel
    cz, cy, cx = centre_um
    rz = max(1, int(np.ceil(3 * sigma_um / dz)))
    ry = max(1, int(np.ceil(3 * sigma_um / dy)))
    rx = max(1, int(np.ceil(3 * sigma_um / dx)))
    iz, iy, ix = int(round(cz / dz)), int(round(cy / dy)), int(round(cx / dx))

    z0, z1 = max(0, iz - rz), min(volume.shape[0], iz + rz + 1)
    y0, y1 = max(0, iy - ry), min(volume.shape[1], iy + ry + 1)
    x0, x1 = max(0, ix - rx), min(volume.shape[2], ix + rx + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return

    zz = (np.arange(z0, z1) * dz - cz)[:, None, None]
    yy = (np.arange(y0, y1) * dy - cy)[None, :, None]
    xx = (np.arange(x0, x1) * dx - cx)[None, None, :]
    volume[z0:z1, y0:y1, x0:x1] += amplitude * np.exp(
        -(zz**2 + yy**2 + xx**2) / (2 * sigma_um**2)
    )


def build(shape_um, voxel, counts, rng, sigma_um, cleft_um, amplitude_range):
    """Return (3, Z, Y, X) float volumes and the ground-truth record."""
    dz, dy, dx = voxel
    shape = (int(shape_um[0] / dz), int(shape_um[1] / dy), int(shape_um[2] / dx))
    volumes = [np.zeros(shape, dtype=np.float32) for _ in CHANNELS]

    margin = 1.0
    def position():
        return np.array([
            rng.uniform(margin, shape_um[0] - margin),
            rng.uniform(margin, shape_um[1] - margin),
            rng.uniform(margin, shape_um[2] - margin),
        ])

    truth = {"excitatory": [], "inhibitory": [],
             "orphans": {name: 0 for name in CHANNELS}}

    for kind, post_channel, n in (
        ("excitatory", 1, counts["excitatory"]),
        ("inhibitory", 2, counts["inhibitory"]),
    ):
        for _ in range(n):
            centre = position()
            # Random orientation of the cleft, fixed physical separation.
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            offset = direction * (cleft_um / 2.0)
            pre, post = centre - offset, centre + offset
            _add_punctum(volumes[0], pre, voxel, rng.uniform(*amplitude_range), sigma_um, rng)
            _add_punctum(volumes[post_channel], post, voxel,
                         rng.uniform(*amplitude_range), sigma_um, rng)
            truth[kind].append({"z_um": float(centre[0]), "y_um": float(centre[1]),
                                "x_um": float(centre[2])})

    for index, name in enumerate(CHANNELS):
        for _ in range(counts["orphans"][name]):
            _add_punctum(volumes[index], position(), voxel,
                         rng.uniform(*amplitude_range), sigma_um, rng)
            truth["orphans"][name] += 1

    return np.stack(volumes), truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--n-stacks", type=int, default=1)
    parser.add_argument("--shape-um", type=float, nargs=3, default=[9.0, 25.0, 25.0],
                        metavar=("Z", "Y", "X"))
    parser.add_argument("--voxel", type=float, nargs=3, default=[0.30, 0.095, 0.095],
                        metavar=("DZ", "DY", "DX"))
    parser.add_argument("--excitatory", type=int, default=180)
    parser.add_argument("--inhibitory", type=int, default=60)
    parser.add_argument("--orphan-pre", type=int, default=60)
    parser.add_argument("--orphan-exc", type=int, default=70)
    parser.add_argument("--orphan-inh", type=int, default=30)
    parser.add_argument("--sigma-um", type=float, default=0.11,
                        help="punctum sigma after deconvolution")
    parser.add_argument("--cleft-um", type=float, default=0.15,
                        help="centre-to-centre distance of an apposed pair")
    parser.add_argument("--amplitude", type=float, nargs=2, default=[800.0, 3000.0])
    parser.add_argument("--offset", type=float, default=20.0)
    parser.add_argument("--read-noise", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel = tuple(args.voxel)

    for i in range(args.n_stacks):
        rng = np.random.default_rng(args.seed + i)
        counts = {
            "excitatory": args.excitatory,
            "inhibitory": args.inhibitory,
            "orphans": {"Bassoon": args.orphan_pre, "PSD95": args.orphan_exc,
                        "Gephyrin": args.orphan_inh},
        }
        clean, truth = build(tuple(args.shape_um), voxel, counts, rng,
                             args.sigma_um, args.cleft_um, tuple(args.amplitude))
        noisy = rng.poisson(np.maximum(clean, 0)).astype(np.float32)
        noisy += args.offset + rng.normal(0, args.read_noise, size=noisy.shape)
        observed = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)

        stem = f"synth_synapses_{i + 1:02d}"
        path = out_dir / f"{stem}.ome.tif"
        write_ome_tiff(path, observed, voxel_size_um=voxel, channel_names=CHANNELS,
                       description="Synthetic triple-labelled synapse stack")

        truth["voxel_size_um"] = list(voxel)
        truth["shape_um"] = list(args.shape_um)
        truth["cleft_um"] = args.cleft_um
        truth["n_excitatory"] = args.excitatory
        truth["n_inhibitory"] = args.inhibitory
        truth["n_puncta"] = {
            "Bassoon": args.excitatory + args.inhibitory + args.orphan_pre,
            "PSD95": args.excitatory + args.orphan_exc,
            "Gephyrin": args.inhibitory + args.orphan_inh,
        }
        (out_dir / f"{stem}_truth.json").write_text(json.dumps(truth, indent=2))

        volume_um3 = float(np.prod(args.shape_um))
        print(f"wrote {path}  {observed.shape}  range {observed.min()}-{observed.max()}")
        print(f"  vérité: {args.excitatory} exc + {args.inhibitory} inh dans "
              f"{volume_um3:.0f} um3  "
              f"({100 * args.excitatory / volume_um3:.2f} / "
              f"{100 * args.inhibitory / volume_um3:.2f} par 100 um3)")
        print(f"  puncta: {truth['n_puncta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
