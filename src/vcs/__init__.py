
try:
    from ._version import version as __version__
except ImportError:
    # Fallback version if setuptools_scm version file is not available
    __version__ = "1.0.0"

__author__ = "Harsh Dubey"
__email__ = "chulwoo.pack@sdstate.edu"

# Main scoring function
from .scorer import compute_vcs_score

# Visualization functions
from ._visualize_vcs import (
    visualize_config,
    visualize_text_chunks,
    visualize_similarity_matrix,
    visualize_mapping_windows,
    visualize_best_match,
    visualize_line_nas,
    visualize_line_nas_precision_calculations,
    visualize_line_nas_recall_calculations,  
    visualize_distance_nas,
    visualize_las,
    visualize_window_regularizer,
    visualize_metrics_summary,
    create_vcs_pdf_report
)

# Configuration constants
from ._config import (
    DEFAULT_CONTEXT_CUTOFF_VALUE,
    DEFAULT_CONTEXT_WINDOW_CONTROL,
    DEFAULT_LCT,
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
    "visualize_line_nas",
    "visualize_line_nas_precision_calculations",
    "visualize_line_nas_recall_calculations",
    "visualize_distance_nas",
    "visualize_las",
    "visualize_window_regularizer",
    "visualize_metrics_summary",
    "create_vcs_pdf_report",
    
    # Configuration constants
    "DEFAULT_CONTEXT_CUTOFF_VALUE",
    "DEFAULT_CONTEXT_WINDOW_CONTROL",
    "DEFAULT_LCT",
    "DEFAULT_CHUNK_SIZE",
]

# Package metadata for programmatic access
__package_name__ = "vcs-metrics"
__description__ = "Video Comprehension Score (VCS) - A comprehensive metric for evaluating narrative similarity"
__url__ = "https://github.com/yourusername/vcs-metrics"
__license__ = "MIT"