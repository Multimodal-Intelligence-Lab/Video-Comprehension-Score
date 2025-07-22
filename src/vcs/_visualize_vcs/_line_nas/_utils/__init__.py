from .pagination import paginate_segments, should_paginate, calculate_total_pages
from .figure_creation import create_base_calculation_figure, finalize_calculation_figure, create_empty_segments_figure
from .summary_generation import generate_summary_text, generate_calculation_method_text, generate_lct_note
from .table_formatting import create_segment_table, format_segment_row
from .path_plotting import plot_floor_path, plot_ceiling_path, plot_actual_path
from .precision_plotting import setup_precision_plot, draw_precision_content
from .recall_plotting import setup_recall_plot, draw_recall_content
from .window_drawing import draw_mapping_windows

__all__ = [
    # Pagination
    "paginate_segments",
    "should_paginate", 
    "calculate_total_pages",
    
    # Figure creation
    "create_base_calculation_figure",
    "finalize_calculation_figure",
    "create_empty_segments_figure",
    
    # Summary generation
    "generate_summary_text",
    "generate_calculation_method_text", 
    "generate_lct_note",
    
    # Table formatting
    "create_segment_table",
    "format_segment_row",
    
    # Path plotting
    "plot_floor_path",
    "plot_ceiling_path", 
    "plot_actual_path",
    
    # Plot setup
    "setup_precision_plot",
    "draw_precision_content",
    "setup_recall_plot",
    "draw_recall_content",
    
    # Window drawing
    "draw_mapping_windows"
]