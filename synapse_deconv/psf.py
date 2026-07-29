"""Theoretical 3D point spread functions on an anisotropic voxel grid.

Implements the Gibson & Lanni (1991) scalar diffraction model, which accounts
for the refractive-index and thickness mismatch between the design conditions of
the objective and the actual immersion / coverslip / sample stack. This is the
model that matters for a 1.40 NA oil lens looking into an aqueous or
glycerol-based mounting medium: the immersion oil (n = 1.515) and the sample
(n ~ 1.33-1.47) differ, which produces depth-dependent spherical aberration and
an axially asymmetric PSF. Born & Wolf is available as a special case
(no mismatch: the sample and coverslip are treated as immersion medium).

The optical path difference across the pupil is

    OPD(rho) =  ns  * zp        * sqrt(1 - (NA rho / ns )^2)
              + ni  * (ti0 + z) * sqrt(1 - (NA rho / ni )^2)
              - ni0 * ti0       * sqrt(1 - (NA rho / ni0)^2)
              + ng  * tg        * sqrt(1 - (NA rho / ng )^2)
              - ng0 * tg0       * sqrt(1 - (NA rho / ng0)^2)

with rho the normalised pupil radius, zp the depth of the emitter below the
coverslip and z the defocus. The amplitude PSF is the Hankel transform

    h(r, z) = | integral_0^1 J0(k NA r rho) exp(i k OPD(rho, z)) rho drho |^2 ,

evaluated on a radial grid and then interpolated onto the Cartesian voxel grid.
Because NA > ns for an oil lens imaging into water, sqrt() is evaluated in the
complex plane: beyond the critical angle the contribution becomes evanescent and
decays instead of oscillating, which is the physically correct behaviour.

All lengths are in micrometres unless a name says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import j0

from .config import ChannelConfig, Config, OpticsConfig, PSFConfig

logger = logging.getLogger(__name__)

__all__ = ["PSFResult", "compute_psf", "psf_for_channel", "theoretical_resolution"]

#: Guards exp() against overflow/underflow for strongly evanescent contributions.
_MAX_EXPONENT = 700.0


@dataclass
class PSFResult:
    """A normalised 3D PSF kernel and the parameters that produced it."""

    data: np.ndarray                 # (Z, Y, X), float64, sums to 1
    voxel_size_um: tuple[float, float, float]   # (dz, dy, dx)
    parameters: dict                 # everything needed to recompute it
    cache_key: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape


def theoretical_resolution(wavelength_nm: float, na: float, ri: float) -> tuple[float, float]:
    """Rayleigh lateral and axial resolution in micrometres.

    Used for automatic kernel sizing and reported in the log so the sampling can
    be checked against the Nyquist criterion.
    """
    lam = wavelength_nm / 1000.0
    lateral = 0.61 * lam / na
    axial = 2.0 * lam * ri / (na * na)
    return lateral, axial


def _optical_path_difference(
    rho: np.ndarray,
    z_um: np.ndarray,
    o: OpticsConfig,
    particle_depth_um: float,
) -> np.ndarray:
    """OPD in micrometres, shape (n_z, n_rho), complex.

    ``rho`` is the normalised pupil radius in [0, 1]; ``z_um`` the defocus of
    each plane relative to the nominal focus.
    """
    na = o.numerical_aperture
    a = na * rho                       # transverse wavevector component / k0
    csqrt = np.emath.sqrt             # returns complex for negative arguments

    def term(n: float | np.ndarray, thickness: float | np.ndarray) -> np.ndarray:
        return n * thickness * csqrt(1.0 - (a / n) ** 2)

    # Sample: the only term that can become evanescent (NA may exceed ns).
    opd = term(o.sample_ri, particle_depth_um)[None, :]
    # Coverslip mismatch (zero when actual == design).
    opd = opd + (term(o.coverslip_ri, o.coverslip_thickness_um)
                 - term(o.coverslip_ri_design, o.coverslip_thickness_design_um))[None, :]
    # Immersion: carries the defocus through ti = ti0 + z.
    ti = o.working_distance_um + np.asarray(z_um, dtype=float)[:, None]
    opd = opd + o.immersion_ri * ti * csqrt(1.0 - (a / o.immersion_ri) ** 2)[None, :]
    opd = opd - term(o.immersion_ri_design, o.working_distance_um)[None, :]
    return np.asarray(opd, dtype=np.complex128)


def _born_wolf_optics(o: OpticsConfig) -> OpticsConfig:
    """Collapse the layered model to a homogeneous one (Born & Wolf limit)."""
    from dataclasses import replace

    return replace(
        o,
        sample_ri=o.immersion_ri,
        coverslip_ri=o.immersion_ri,
        coverslip_ri_design=o.immersion_ri,
        coverslip_thickness_um=0.0,
        coverslip_thickness_design_um=0.0,
        immersion_ri_design=o.immersion_ri,
        particle_depth_um=0.0,
    )


def _radial_intensity(
    r_um: np.ndarray,
    z_um: np.ndarray,
    wavelength_nm: float,
    o: OpticsConfig,
    particle_depth_um: float,
    n_pupil: int,
) -> np.ndarray:
    """Intensity PSF sampled on a (z, r) grid, shape (n_z, n_r).

    The pupil integral is discretised with the trapezoidal rule. J0 depends only
    on (r, rho) so its matrix is built once and reused for every plane, turning
    the whole computation into a single matrix product per plane.
    """
    k = 2.0 * np.pi / (wavelength_nm / 1000.0)     # rad / um
    na = o.numerical_aperture

    rho = np.linspace(0.0, 1.0, n_pupil)
    weights = np.full(n_pupil, 1.0 / (n_pupil - 1))
    weights[0] *= 0.5
    weights[-1] *= 0.5

    # (n_r, n_pupil): Bessel kernel of the Hankel transform, times rho drho.
    bessel = j0(k * na * np.outer(np.asarray(r_um, dtype=float), rho))
    bessel = bessel * (rho * weights)[None, :]

    opd = _optical_path_difference(rho, z_um, o, particle_depth_um)
    exponent = 1j * k * opd
    # Evanescent terms give a large negative real part; clip before exp().
    np.clip(exponent.real, -_MAX_EXPONENT, 0.0, out=exponent.real)
    phasor = np.exp(exponent)                       # (n_z, n_pupil)

    amplitude = phasor @ bessel.T                   # (n_z, n_r)
    return np.abs(amplitude) ** 2


def _profile_to_cartesian(
    profile: np.ndarray,
    r_grid: np.ndarray,
    z_index: np.ndarray,
    shape: tuple[int, int, int],
    voxel: tuple[float, float, float],
    oversample_xy: int,
) -> np.ndarray:
    """Interpolate an (n_z, n_r) radial profile onto a Cartesian (Z, Y, X) grid.

    Each output voxel is averaged over an ``oversample_xy**2`` sub-grid, which
    matters at 0.095 um pixels where the PSF core spans only ~2 pixels.
    """
    nz, ny, nx = shape
    _, dy, dx = voxel
    os_xy = max(1, oversample_xy)

    # Sub-pixel offsets centred on each pixel.
    offs = (np.arange(os_xy) - (os_xy - 1) / 2.0) / os_xy
    y_centres = (np.arange(ny) - (ny - 1) / 2.0) * dy
    x_centres = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y_sub = (y_centres[:, None] + offs[None, :] * dy).ravel()
    x_sub = (x_centres[:, None] + offs[None, :] * dx).ravel()
    radii = np.hypot(y_sub[:, None], x_sub[None, :])

    out = np.empty(shape, dtype=np.float64)
    flat_r = radii.ravel()
    for i, zi in enumerate(z_index):
        plane = np.interp(flat_r, r_grid, profile[zi], left=profile[zi][0], right=0.0)
        plane = plane.reshape(ny, os_xy, nx, os_xy).mean(axis=(1, 3))
        out[i] = plane
    return out


def _pinhole_kernel(radius_um: float, dy: float, dx: float) -> np.ndarray:
    """Normalised disk of the back-projected pinhole, sampled on the pixel grid."""
    ry = max(1, int(np.ceil(radius_um / dy)))
    rx = max(1, int(np.ceil(radius_um / dx)))
    yy = (np.arange(-ry, ry + 1))[:, None] * dy
    xx = (np.arange(-rx, rx + 1))[None, :] * dx
    # Supersample the disk edge so a sub-pixel pinhole is still represented.
    sub = 4
    offs = (np.arange(sub) - (sub - 1) / 2.0) / sub
    ys = (yy[..., None] + offs * dy)[..., None]
    xs = (xx[..., None, None] + offs * dx)
    disk = (np.hypot(ys, xs) <= radius_um).mean(axis=(-1, -2))
    total = disk.sum()
    if total <= 0:      # pinhole far below one pixel: treat as a point detector
        disk = np.zeros_like(disk)
        disk[ry, rx] = 1.0
        return disk
    return disk / total


def _airy_unit_um(wavelength_nm: float, na: float) -> float:
    """One Airy unit (object-space diameter) in micrometres."""
    return 1.22 * (wavelength_nm / 1000.0) / na


def _cache_key(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _auto_kernel_size(
    extent_um: float, voxel_um: float, limit: int, minimum: int = 5
) -> int:
    """Odd voxel count covering +/- ``extent_um``, capped at ``limit``."""
    n = int(np.ceil(extent_um / voxel_um)) * 2 + 1
    n = max(minimum, min(n, limit))
    return n if n % 2 == 1 else n - 1


def compute_psf(
    *,
    voxel_size_um: tuple[float, float, float],
    emission_nm: float,
    optics: OpticsConfig,
    psf_cfg: PSFConfig,
    excitation_nm: float | None = None,
    stack_shape: tuple[int, int, int] | None = None,
) -> PSFResult:
    """Compute a normalised 3D PSF for one channel on the data's voxel grid.

    ``stack_shape`` (Z, Y, X), when given, caps the kernel so it never exceeds
    the image it will be convolved with.
    """
    dz, dy, dx = voxel_size_um
    if min(dz, dy, dx) <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel_size_um}")

    optics_eff = optics if psf_cfg.model == "gibson_lanni" else _born_wolf_optics(optics)
    depth = optics_eff.particle_depth_um

    lateral_res, axial_res = theoretical_resolution(
        emission_nm, optics.numerical_aperture, optics.immersion_ri
    )

    # The kernel must hold not just the core but the out-of-focus skirt, which is
    # what carries the haze the deconvolution is supposed to remove. Half-widths
    # of 4 lateral and 2 axial Rayleigh units capture it without making the FFT
    # needlessly large.
    max_z = stack_shape[0] if stack_shape else 10_001
    max_xy = min(stack_shape[1], stack_shape[2]) if stack_shape else 10_001
    nz = psf_cfg.z_size or _auto_kernel_size(2.0 * axial_res, dz, max_z)
    nxy = psf_cfg.xy_size or _auto_kernel_size(4.0 * lateral_res, dy, max_xy)
    if nz % 2 == 0:
        nz -= 1
    if nxy % 2 == 0:
        nxy -= 1

    params = {
        "model": psf_cfg.model,
        "mode": psf_cfg.mode,
        "emission_nm": emission_nm,
        "excitation_nm": excitation_nm if psf_cfg.mode == "confocal" else None,
        "voxel_size_um": [dz, dy, dx],
        "shape": [nz, nxy, nxy],
        "oversample_xy": psf_cfg.oversample_xy,
        "oversample_z": psf_cfg.oversample_z,
        "center_on_peak": psf_cfg.center_on_peak,
        "integration_samples": psf_cfg.integration_samples,
        "radial_samples": psf_cfg.radial_samples,
        "optics": {
            "na": optics_eff.numerical_aperture,
            "ni": optics_eff.immersion_ri,
            "ni0": optics_eff.immersion_ri_design,
            "ng": optics_eff.coverslip_ri,
            "ng0": optics_eff.coverslip_ri_design,
            "tg": optics_eff.coverslip_thickness_um,
            "tg0": optics_eff.coverslip_thickness_design_um,
            "ti0": optics_eff.working_distance_um,
            "ns": optics_eff.sample_ri,
            "zp": depth,
        },
    }
    if psf_cfg.mode == "confocal":
        params["pinhole_airy_units"] = psf_cfg.pinhole_airy_units
        params["pinhole_radius_um"] = psf_cfg.pinhole_radius_um

    key = _cache_key(params)
    cached = _load_cached(psf_cfg.cache_dir, key)
    if cached is not None:
        logger.debug("PSF cache hit (%s, em=%.0f nm)", key, emission_nm)
        return PSFResult(cached, (dz, dy, dx), params, key)

    # --- sampling grids -------------------------------------------------
    # With index mismatch the emitter at depth zp is in focus not at z = 0 but
    # at the stage position that cancels the quadratic (defocus) term of the
    # OPD expansion, i.e. z0 = -zp * ni / ns. Sampling the kernel around z0
    # puts the PSF core near the centre voxel. z0 does not depend on the
    # wavelength, so all channels are shifted identically and their relative
    # axial registration -- what synapse colocalisation depends on -- is
    # preserved. Any residual offset is genuine (wavelength-dependent)
    # spherical aberration and is reported below.
    z0 = -depth * optics_eff.immersion_ri / optics_eff.sample_ri
    os_z = max(1, psf_cfg.oversample_z)
    z_offsets = (np.arange(os_z) - (os_z - 1) / 2.0) / os_z * dz
    z_centres = (np.arange(nz) - (nz - 1) / 2.0) * dz + z0
    z_all = (z_centres[:, None] + z_offsets[None, :]).ravel()

    r_max = float(np.hypot((nxy - 1) / 2.0 * dy, (nxy - 1) / 2.0 * dx)) + max(dy, dx)
    r_grid = np.linspace(0.0, r_max, psf_cfg.radial_samples)

    def build(wavelength_nm: float) -> np.ndarray:
        profile = _radial_intensity(
            r_grid, z_all, wavelength_nm, optics_eff, depth, psf_cfg.integration_samples
        )
        vol = _profile_to_cartesian(
            profile,
            r_grid,
            np.arange(len(z_all)),
            (len(z_all), nxy, nxy),
            (dz, dy, dx),
            psf_cfg.oversample_xy,
        )
        if os_z > 1:                       # average the z sub-planes
            vol = vol.reshape(nz, os_z, nxy, nxy).mean(axis=1)
        return vol

    psf_em = build(emission_nm)

    if psf_cfg.mode == "confocal":
        if excitation_nm is None:
            raise ValueError("confocal mode requires an excitation wavelength")
        radius = psf_cfg.pinhole_radius_um
        if radius is None:
            radius = 0.5 * psf_cfg.pinhole_airy_units * _airy_unit_um(
                emission_nm, optics.numerical_aperture
            )
        kernel = _pinhole_kernel(radius, dy, dx)
        detection = np.empty_like(psf_em)
        from scipy.signal import fftconvolve

        for i in range(psf_em.shape[0]):
            detection[i] = fftconvolve(psf_em[i], kernel, mode="same")
        np.clip(detection, 0.0, None, out=detection)
        psf_total = build(excitation_nm) * detection
        logger.debug(
            "confocal PSF: ex=%.0f nm, em=%.0f nm, back-projected pinhole radius %.3f um",
            excitation_nm, emission_nm, radius,
        )
    else:
        psf_total = psf_em

    peak_plane = int(np.argmax(psf_total.max(axis=(1, 2))))
    offset = peak_plane - nz // 2
    if offset and abs(offset) > max(1, nz // 8):
        logger.warning(
            "PSF peak sits %d planes (%.2f um) off the kernel centre at em=%.0f nm. "
            "This means strong spherical aberration for particle_depth_um=%.1f with "
            "sample_ri=%.3f; consider increasing psf.z_size or checking optics.sample_ri.",
            offset, offset * dz, emission_nm, depth, optics_eff.sample_ri,
        )
    elif offset:
        logger.debug(
            "PSF peak %+d plane(s) off centre at em=%.0f nm (residual aberration)",
            offset, emission_nm,
        )
    if psf_cfg.center_on_peak and offset:
        psf_total = np.roll(psf_total, -offset, axis=0)
        # Rolled-in planes are wrapped garbage; zero them.
        if offset > 0:
            psf_total[-offset:] = 0.0
        else:
            psf_total[:-offset] = 0.0

    total = psf_total.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(
            f"PSF computation produced a degenerate kernel (sum={total}); "
            "check the optical parameters"
        )
    psf_total = psf_total / total

    _save_cached(psf_cfg.cache_dir, key, psf_total)
    logger.debug(
        "PSF built: em=%.0f nm shape=%s voxel=(%.4f, %.4f, %.4f) um key=%s",
        emission_nm, psf_total.shape, dz, dy, dx, key,
    )
    return PSFResult(psf_total, (dz, dy, dx), params, key)


def psf_for_channel(
    cfg: Config,
    channel: ChannelConfig,
    voxel_size_um: tuple[float, float, float],
    stack_shape: tuple[int, int, int] | None = None,
) -> PSFResult:
    """Convenience wrapper: build the PSF for one configured channel."""
    return compute_psf(
        voxel_size_um=voxel_size_um,
        emission_nm=channel.emission_nm,
        excitation_nm=channel.excitation_nm,
        optics=cfg.optics,
        psf_cfg=cfg.psf,
        stack_shape=stack_shape,
    )


def _load_cached(cache_dir: str | None, key: str) -> np.ndarray | None:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"psf_{key}.npy"
    if not path.is_file():
        return None
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        logger.warning("ignoring unreadable PSF cache %s (%s)", path, exc)
        return None


def _save_cached(cache_dir: str | None, key: str, data: np.ndarray) -> None:
    if not cache_dir:
        return
    try:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"psf_{key}.npy", data)
    except OSError as exc:
        logger.warning("could not write PSF cache in %s (%s)", cache_dir, exc)


def measure_fwhm(psf: np.ndarray, voxel_size_um: tuple[float, float, float]) -> dict:
    """FWHM of the PSF along z and x, in micrometres (linear interpolation).

    Reported in the log as a sanity check against the theoretical resolution.
    """
    dz, _, dx = voxel_size_um
    cz, cy, cx = (s // 2 for s in psf.shape)

    def fwhm(profile: np.ndarray, spacing: float, centre: int) -> float:
        peak = profile[centre]
        if peak <= 0:
            return float("nan")
        half = peak / 2.0
        edges = []
        for direction in (-1, 1):
            i = centre
            while 0 < i < len(profile) - 1 and profile[i] > half:
                i += direction
            if profile[i] > half:            # never crossed within the kernel
                return float("nan")
            prev = i - direction
            span = profile[prev] - profile[i]
            frac = (profile[prev] - half) / span if span else 0.0
            edges.append(prev + direction * frac)
        return abs(edges[1] - edges[0]) * spacing

    return {
        "fwhm_z_um": fwhm(psf[:, cy, cx], dz, cz),
        "fwhm_x_um": fwhm(psf[cz, cy, :], dx, cx),
    }
