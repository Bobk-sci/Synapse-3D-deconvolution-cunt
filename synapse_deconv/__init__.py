"""3D deconvolution pipeline for confocal z-stacks (Olympus FV1000).

Channel-by-channel Richardson-Lucy deconvolution with a theoretical
Gibson-Lanni PSF, on anisotropic voxel grids, driven by a single config file.
"""

from .config import Config, ConfigError, load_config
from .deconvolution import richardson_lucy
from .pipeline import BatchSummary, StackResult, process_stack, run_batch
from .psf import compute_psf
from .readers import ImageStack, ReadError, read_stack
from .writers import write_ome_tiff

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "richardson_lucy",
    "compute_psf",
    "read_stack",
    "ImageStack",
    "ReadError",
    "write_ome_tiff",
    "process_stack",
    "run_batch",
    "StackResult",
    "BatchSummary",
    "__version__",
]
