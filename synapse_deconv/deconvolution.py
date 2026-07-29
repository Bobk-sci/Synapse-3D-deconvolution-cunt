"""3D Richardson-Lucy deconvolution.

Classic multiplicative Richardson-Lucy for Poisson-distributed photon counts:

    x_{k+1} = x_k * [ h^T (*) ( y / (h (*) x_k) ) ]

with ``h`` the normalised PSF, ``y`` the measurement and ``(*)`` a convolution.
Non-negativity is intrinsic to the update (a positive start stays positive) and
is enforced explicitly against round-off.

Optionally the total-variation regularised variant of Dey et al. (2006) damps
the noise amplification that plain RL produces after ~20 iterations:

    x_{k+1} = x_k / (1 - lambda * div(grad x_k / |grad x_k|)) * [ h^T (*) (y / (h (*) x_k)) ]

Everything runs on the FFT, with the image padded by the PSF half-width so that
the circular convolution does not wrap signal from the opposite edge -- the
usual source of bright rims on deconvolved confocal stacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DeconvolutionResult", "richardson_lucy", "get_backend", "available_backends"]


@dataclass
class DeconvolutionResult:
    """Deconvolved volume plus per-run diagnostics."""

    data: np.ndarray                 # float32/float64, same shape as the input
    iterations: int
    backend: str
    #: Relative change ||x_k - x_{k-1}|| / ||x_{k-1}|| for each iteration.
    convergence: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first."""
    backends = []
    try:
        import cupy                                   # noqa: F401
        import cupy.cuda                              # noqa: F401

        if cupy.cuda.runtime.getDeviceCount() > 0:
            backends.append("cupy")
    except Exception:                                 # no CuPy, no driver, no device
        pass
    backends.append("numpy")
    return backends


def get_backend(requested: str = "auto"):
    """Return ``(name, array_module, fft_module)`` for the requested backend."""
    if requested == "cupy" or (requested == "auto" and "cupy" in available_backends()):
        try:
            import cupy as cp
            import cupyx.scipy.fft as cufft

            return "cupy", cp, cufft
        except Exception as exc:
            if requested == "cupy":
                raise RuntimeError(f"CuPy backend requested but unavailable: {exc}") from exc
            logger.debug("CuPy unavailable, falling back to NumPy (%s)", exc)

    import scipy.fft as spfft

    return "numpy", np, spfft


def _pad_shape(image_shape, psf_shape, fft_mod) -> tuple[list[int], list[tuple[int, int]]]:
    """FFT-friendly padded shape and the per-axis padding widths."""
    pads = []
    shape = []
    for n, m in zip(image_shape, psf_shape):
        half = m // 2
        target = fft_mod.next_fast_len(n + 2 * half)
        extra = target - (n + 2 * half)
        pads.append((half, half + extra))
        shape.append(target)
    return shape, pads


def _psf_otf(psf, shape, xp, fft_mod):
    """Zero-pad and centre the PSF, then return its real FFT (the OTF).

    The PSF centre must land on index 0 of the padded array, otherwise the
    convolution shifts the image by half the kernel.
    """
    padded = xp.zeros(shape, dtype=psf.dtype)
    slices = tuple(slice(0, s) for s in psf.shape)
    padded[slices] = psf
    for axis, size in enumerate(psf.shape):
        padded = xp.roll(padded, -(size // 2), axis=axis)
    return fft_mod.rfftn(padded, s=shape)


def _tv_factor(x, lam: float, eps: float, xp):
    """1 / (1 - lambda * div(grad x / |grad x|)), the Dey et al. TV damping term."""
    grads = xp.gradient(x)
    norm = xp.sqrt(sum(g * g for g in grads) + eps)
    divergence = sum(
        xp.gradient(g / norm, axis=axis) for axis, g in enumerate(grads)
    )
    factor = 1.0 - lam * divergence
    # Keep the factor safely away from zero so the division cannot explode.
    return xp.clip(factor, 0.5, 2.0)


def richardson_lucy(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    iterations: int = 25,
    backend: str = "auto",
    dtype: str = "float32",
    pad_mode: str = "reflect",
    epsilon: float = 1e-9,
    regularization: str = "none",
    tv_lambda: float = 0.002,
    progress: Callable[[int, float], None] | None = None,
) -> DeconvolutionResult:
    """Deconvolve a 3D volume with a 3D PSF.

    Args:
        image: 3D array (Z, Y, X) of non-negative intensities.
        psf: 3D PSF, same voxel grid as ``image``; renormalised to sum 1 here.
        iterations: number of RL iterations.
        backend: ``auto``, ``numpy`` or ``cupy``.
        dtype: ``float32`` (default; ample for 16-bit data) or ``float64``.
        pad_mode: how the image is extended before the FFT.
        epsilon: floor used in the divisions, guards against 0/0.
        regularization: ``none`` or ``tv``.
        tv_lambda: TV weight; only used when ``regularization='tv'``.
        progress: optional callback ``(iteration, relative_change)``.

    Returns:
        A :class:`DeconvolutionResult` whose ``data`` has the input's shape and
        the same integrated intensity, up to the edge handling.
    """
    if image.ndim != 3 or psf.ndim != 3:
        raise ValueError(f"expected 3D image and PSF, got {image.ndim}D and {psf.ndim}D")
    if any(p > i for p, i in zip(psf.shape, image.shape)):
        raise ValueError(
            f"PSF {psf.shape} is larger than the image {image.shape}; "
            "reduce psf.xy_size / psf.z_size"
        )
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    np_dtype = np.float32 if dtype == "float32" else np.float64
    name, xp, fft_mod = get_backend(backend)

    image_host = np.asarray(image, dtype=np_dtype)
    if not np.all(np.isfinite(image_host)):
        raise ValueError("image contains NaN or Inf; clean it before deconvolving")
    if image_host.min() < 0:
        raise ValueError("image contains negative values; Richardson-Lucy needs counts >= 0")

    psf_host = np.asarray(psf, dtype=np.float64)
    psf_sum = psf_host.sum()
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError("PSF must have a finite positive sum")
    psf_host = (psf_host / psf_sum).astype(np_dtype)

    shape, pads = _pad_shape(image_host.shape, psf_host.shape, fft_mod)
    padded = np.pad(image_host, pads, mode=pad_mode)
    inner = tuple(slice(before, before + n)
                  for (before, _), n in zip(pads, image_host.shape))

    observed = xp.asarray(padded)
    otf = _psf_otf(xp.asarray(psf_host), shape, xp, fft_mod)
    otf_conj = xp.conj(otf)
    eps = np_dtype(epsilon)

    def convolve(volume, transfer):
        return fft_mod.irfftn(fft_mod.rfftn(volume, s=shape) * transfer, s=shape)

    # Starting from the observation converges faster than a flat guess and keeps
    # the total intensity in the right ballpark from iteration 1.
    estimate = xp.maximum(observed, eps)
    convergence: list[float] = []
    use_tv = regularization == "tv" and tv_lambda > 0

    for k in range(iterations):
        blurred = convolve(estimate, otf)
        xp.maximum(blurred, eps, out=blurred)
        ratio = observed / blurred
        correction = convolve(ratio, otf_conj)
        xp.maximum(correction, 0.0, out=correction)

        previous = estimate
        estimate = estimate * correction
        if use_tv:
            estimate = estimate / _tv_factor(previous, tv_lambda, float(eps), xp)
        xp.maximum(estimate, 0.0, out=estimate)

        denominator = float(xp.linalg.norm(previous))
        change = float(xp.linalg.norm(estimate - previous)) / denominator if denominator else 0.0
        convergence.append(change)
        if progress is not None:
            progress(k + 1, change)
        if not np.isfinite(change):
            raise RuntimeError(
                f"Richardson-Lucy diverged at iteration {k + 1} (non-finite estimate); "
                "check the PSF and the input for zeros or saturation"
            )

    result = estimate[inner]
    if name == "cupy":
        result = xp.asnumpy(result)
        del observed, otf, otf_conj, estimate
        xp.get_default_memory_pool().free_all_blocks()

    result = np.ascontiguousarray(np.asarray(result, dtype=np_dtype))
    np.maximum(result, 0.0, out=result)

    input_total = float(image_host.sum())
    output_total = float(result.sum())
    diagnostics = {
        "input_total_intensity": input_total,
        "output_total_intensity": output_total,
        "intensity_ratio": output_total / input_total if input_total else float("nan"),
        "padded_shape": list(shape),
        "final_relative_change": convergence[-1] if convergence else 0.0,
        "regularization": regularization if use_tv else "none",
    }
    logger.debug(
        "RL done: %d iterations on %s, intensity ratio %.4f, last change %.2e",
        iterations, name, diagnostics["intensity_ratio"], diagnostics["final_relative_change"],
    )
    return DeconvolutionResult(
        data=result, iterations=iterations, backend=name,
        convergence=convergence, diagnostics=diagnostics,
    )
