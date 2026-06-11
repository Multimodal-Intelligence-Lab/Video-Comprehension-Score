
# Version from package metadata (shared with the per-result Config string)
from ._utils import _resolve_version

__version__ = _resolve_version()

__author__ = "Harsh Dubey"
__email__ = "had7143@gmail.com"

# Main scoring functions
from .scorer import compute_vcs_score, compute_vcs_from_embeddings

# Configuration constants
from ._config import (
    DEFAULT_CONTEXT_CUTOFF_VALUE,
    DEFAULT_CONTEXT_WINDOW_CONTROL,
    DEFAULT_Rn,
    DEFAULT_CHUNK_SIZE,
)

__all__ = [
    # Main functions
    "compute_vcs_score",
    "compute_vcs_from_embeddings",

    # Version and metadata
    "__version__",
    "__author__",
    "__email__",

    # Configuration constants
    "DEFAULT_CONTEXT_CUTOFF_VALUE",
    "DEFAULT_CONTEXT_WINDOW_CONTROL",
    "DEFAULT_Rn",
    "DEFAULT_CHUNK_SIZE",
]

# Package metadata for programmatic access
__package_name__ = "video-comprehension-score"
__description__ = "Video Comprehension Score (VCS) - A comprehensive metric for evaluating narrative similarity"
__url__ = "https://github.com/Multimodal-Intelligence-Lab/Video-Comprehension-Score"
__license__ = "MIT"