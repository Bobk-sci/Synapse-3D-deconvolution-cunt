"""Generate a synthetic FV1000-like z-stack for testing the pipeline.

Produces a 3-channel, 16-bit, anisotropic OME-TIFF that mimics a synaptic
immunolabelling: small puncta (some colocalised between channels, some not),
a faint dendritic background, blurred with the true Gibson-Lanni PSF of each
channel and corrupted with Poisson shot noise plus Gaussian read noise.

Because the ground truth is known, this is what the pipeline's end-to-end test
checks against.

    python scripts/make_test_stack.py --out data/raw --n-stacks 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from synapse_deconv.config import OpticsConfig, PSFConfig
from synapse_deconv.psf import compute_psf
from synapse_deconv.writers import write_ome_tiff

CHANNELS = [
    ("Alexa405", 421.0, 405.0),
    ("Alexa488", 519.0, 488.0),
    ("Alexa594", 617.0, 594.0),
]


def make_ground_truth(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    n_puncta: int,
    n_shared: int,
    brightness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (C, Z, Y, X) ground truth and the (N, 3) coordinates of the puncta.

    ``n_shared`` puncta appear in all three channels (colocalised synaptic
    markers); the rest are channel-specific. ``brightness`` is the total photon
    budget of a punctum: after convolution the PSF spreads it over ~10^3 voxels,
    so a value of 3e5 gives peak counts in the low thousands, like a
    well-exposed FV1000 acquisition.
    """
    nz, ny, nx = shape
    truth = np.zeros((3, nz, ny, nx), dtype=np.float32)

    margin = 6
    shared = np.stack([
        rng.integers(margin, nz - margin, n_shared),
        rng.integers(margin, ny - margin, n_shared),
        rng.integers(margin, nx - margin, n_shared),
    ], axis=1)

    for c in range(3):
        # Colocalised puncta, jittered by a voxel as in real data.
        for z, y, x in shared:
            zz = int(np.clip(z + rng.integers(-1, 2), 0, nz - 1))
            yy = int(np.clip(y + rng.integers(-1, 2), 0, ny - 1))
            xx = int(np.clip(x + rng.integers(-1, 2), 0, nx - 1))
            truth[c, zz, yy, xx] += brightness * rng.uniform(0.4, 1.6)

        n_own = n_puncta - n_shared
        for _ in range(n_own):
            z = int(rng.integers(margin, nz - margin))
            y = int(rng.integers(margin, ny - margin))
            x = int(rng.integers(margin, nx - margin))
            truth[c, z, y, x] += brightness * rng.uniform(0.3, 1.2)

    # Faint dendrite-like structures so the image is not just isolated dots.
    for c in range(3):
        for _ in range(4):
            z = int(rng.integers(margin, nz - margin))
            y0, x0 = rng.integers(margin, ny - margin), rng.integers(margin, nx - margin)
            angle = rng.uniform(0, 2 * np.pi)
            length = int(rng.integers(nx // 4, nx // 2))
            for t in range(length):
                y = int(np.clip(y0 + t * np.sin(angle), 0, ny - 1))
                x = int(np.clip(x0 + t * np.cos(angle), 0, nx - 1))
                truth[c, z, y, x] += 0.06 * brightness
    return truth, shared


def simulate(
    truth: np.ndarray,
    voxel: tuple[float, float, float],
    optics: OpticsConfig,
    psf_cfg: PSFConfig,
    rng: np.random.Generator,
    offset: float,
    read_noise: float,
) -> np.ndarray:
    """Blur with the true PSF, then apply Poisson + Gaussian read noise."""
    observed = np.empty(truth.shape, dtype=np.uint16)
    for c, (_, emission, excitation) in enumerate(CHANNELS):
        psf = compute_psf(
            voxel_size_um=voxel, emission_nm=emission, excitation_nm=excitation,
            optics=optics, psf_cfg=psf_cfg,
        ).data
        blurred = fftconvolve(truth[c], psf, mode="same")
        np.maximum(blurred, 0.0, out=blurred)
        noisy = rng.poisson(blurred).astype(np.float32)
        noisy += offset + rng.normal(0.0, read_noise, size=noisy.shape)
        observed[c] = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--n-stacks", type=int, default=1)
    parser.add_argument("--shape", type=int, nargs=3, default=[30, 256, 256],
                        metavar=("Z", "Y", "X"))
    parser.add_argument("--voxel", type=float, nargs=3, default=[0.30, 0.095, 0.095],
                        metavar=("DZ", "DY", "DX"))
    parser.add_argument("--puncta", type=int, default=220)
    parser.add_argument("--shared", type=int, default=90)
    parser.add_argument("--brightness", type=float, default=3.0e5,
                        help="total photon budget of one punctum")
    parser.add_argument("--offset", type=float, default=100.0, help="PMT offset in counts")
    parser.add_argument("--read-noise", type=float, default=8.0)
    parser.add_argument("--sample-ri", type=float, default=1.40,
                        help="mounting medium RI; matches config/default.yaml")
    parser.add_argument("--depth", type=float, default=8.0,
                        help="depth below the coverslip; matches config/default.yaml")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-truth", action="store_true",
                        help="also write the noiseless ground truth")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel = tuple(args.voxel)
    optics = OpticsConfig(sample_ri=args.sample_ri, particle_depth_um=args.depth)
    psf_cfg = PSFConfig(cache_dir=None, xy_size=31, z_size=31)
    names = [name for name, _, _ in CHANNELS]

    for i in range(args.n_stacks):
        rng = np.random.default_rng(args.seed + i)
        truth, shared = make_ground_truth(
            tuple(args.shape), rng, args.puncta, args.shared, args.brightness
        )
        observed = simulate(truth, voxel, optics, psf_cfg, rng, args.offset, args.read_noise)

        path = out_dir / f"synthetic_stack_{i + 1:02d}.ome.tif"
        write_ome_tiff(path, observed, voxel_size_um=voxel, channel_names=names,
                       description="Synthetic FV1000-like test stack")
        print(f"wrote {path}  shape={observed.shape}  dtype={observed.dtype}  "
              f"range={observed.min()}-{observed.max()}  ({len(shared)} colocalised puncta)")

        if args.save_truth:
            truth_path = out_dir / f"synthetic_stack_{i + 1:02d}_truth.ome.tif"
            write_ome_tiff(
                truth_path,
                np.clip(np.rint(truth / truth.max() * 65535), 0, 65535).astype(np.uint16),
                voxel_size_um=voxel, channel_names=names,
                description="Ground truth (noiseless, unblurred)",
            )
            print(f"wrote {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
