
# Import version from package metadata
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("video-comprehension-score")
except PackageNotFoundError:
    # Running from a source tree without an installed distribution
    __version__ = "0.0.0+unknown"

__author__ = "Harsh Dubey"
__email__ = "had7143@gmail.com"

# Main scoring function
from .scorer import compute_vcs_score

# Visualization functions
from ._visualize_vcs import (
    visualize_config,
    visualize_text_chunks,
    visualize_similarity_matrix,
    visualize_mapping_windows,
    visualize_best_match,
    visualize_local_nas,
    visualize_local_nas_precision_calculations,
    visualize_local_nas_recall_calculations,  
    visualize_global_nas,
    visualize_las,
    visualize_las_load_sharing,
    visualize_metrics_summary,
    create_vcs_pdf_report
)

# Configuration constants
from ._config import (
    DEFAULT_CONTEXT_CUTOFF_VALUE,
    DEFAULT_CONTEXT_WINDOW_CONTROL,
    DEFAULT_Rn,
    DEFAULT_CHUNK_SIZE,
)

__all__ = [
    # Main function
    "compute_vcs_score", 
    
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    
    # Visualization functions
    "visualize_config",
    "visualize_text_chunks",
    "visualize_similarity_matrix",
    "visualize_mapping_windows",
    "visualize_best_match",
    "visualize_local_nas",
    "visualize_local_nas_precision_calculations",
    "visualize_local_nas_recall_calculations",
    "visualize_global_nas",
    "visualize_las",
    "visualize_las_load_sharing",
    "visualize_metrics_summary",
    "create_vcs_pdf_report",
    
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