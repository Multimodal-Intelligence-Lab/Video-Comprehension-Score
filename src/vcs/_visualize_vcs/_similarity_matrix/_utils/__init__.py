from .matrix_scaling import (
    determine_matrix_size, calculate_figure_size, calculate_tick_steps,
    setup_axis_ticks
)
from .annotation_handler import create_similarity_heatmap
from .match_highlighting import (
    should_show_matches, highlight_precision_matches, highlight_recall_matches,
    create_matrix_title, highlight_all_matches
)

__all__ = [
    # Matrix scaling
    "determine_matrix_size",
    "calculate_figure_size", 
    "calculate_tick_steps",
    "setup_axis_ticks",
    
    # Annotation handling
    "create_similarity_heatmap",
    
    # Match highlighting
    "should_show_matches",
    "highlight_precision_matches",
    "highlight_recall_matches", 
    "create_matrix_title",
    "highlight_all_matches"
]