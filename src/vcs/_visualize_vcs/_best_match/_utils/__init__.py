from .precision_details import create_precision_details_section
from .recall_details import create_recall_details_section
from .summary_table import create_summary_table_section
from .text_formatting import (
    create_section_header, format_segment_details, format_candidate_details,
    create_table_header, create_table_row, create_table_footer,
    truncate_text
)
# Import the new figure generator functions
from .figure_generators import (
    create_precision_details_figure,
    create_recall_details_figure, 
    create_summary_table_figure
)

__all__ = [
    # Section creators (existing functions)
    "create_precision_details_section",
    "create_recall_details_section", 
    "create_summary_table_section",
    
    # Text formatting utilities (streamlined for 5-column format)
    "create_section_header",
    "format_segment_details",
    "format_candidate_details",
    "create_table_header",
    "create_table_row", 
    "create_table_footer",
    "truncate_text",
    
    # New figure generators for separate displays
    "create_precision_details_figure",
    "create_recall_details_figure",
    "create_summary_table_figure"
]