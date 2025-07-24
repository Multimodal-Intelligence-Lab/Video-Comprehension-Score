from .page_generators import (
    generate_best_match_pages, 
    generate_content_pages,
    generate_text_chunks_pages,
    generate_line_nas_pages
)
from .match_details import create_precision_match_details_page, create_recall_match_details_page
from .summary_tables import create_precision_matches_summary_page, create_recall_matches_summary_page
from .empty_states import create_empty_precision_matches_page, create_empty_recall_matches_page
from .layout_pages import create_title_page, create_metrics_page, create_toc, generate_front_matter
from .pdf_helper import (
    pdf_matplotlib_context,
    setup_pdf_metadata, 
    normalize_metrics_list, 
    extract_key_metrics,
    create_section_structure,
    estimate_pages_for_metric,
    estimate_best_match_pages,
    estimate_paginated_content_pages,
    validate_metrics_list,
    calculate_content_layout,
    filter_sections_and_calculate_pages,
    determine_layout_config,
    add_page_number
)
from .style import setup_matplotlib_style

__all__ = [
    # Page generators
    "generate_best_match_pages",
    "generate_content_pages", 
    "generate_text_chunks_pages",
    "generate_line_nas_pages",
    
    # Match details
    "create_precision_match_details_page",
    "create_recall_match_details_page",
    
    # Summary tables
    "create_precision_matches_summary_page", 
    "create_recall_matches_summary_page",
    
    # Empty states
    "create_empty_precision_matches_page",
    "create_empty_recall_matches_page",
    
    # Layout pages
    "create_title_page",
    "create_metrics_page", 
    "create_toc",
    "generate_front_matter",

    # PDF Utilities
    "pdf_matplotlib_context",
    "setup_pdf_metadata",
    "normalize_metrics_list",
    "extract_key_metrics", 
    "create_section_structure",
    "estimate_pages_for_metric",
    "estimate_best_match_pages",
    "estimate_paginated_content_pages",
    "validate_metrics_list", 
    "calculate_content_layout",
    "filter_sections_and_calculate_pages",
    "determine_layout_config",
    "add_page_number"

    #Styling
    "setup_matplotlib_style"
]