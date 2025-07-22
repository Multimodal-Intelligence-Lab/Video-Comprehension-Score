from .pagination import split_text_chunks_for_display, should_paginate_chunks, calculate_chunk_pages
from .text_formatting import (
    create_chunk_header, format_chunk_content, create_chunk_separators,
    format_page_info, wrap_chunk_text
)
from .figure_creation import create_text_chunk_figure, setup_chunk_figure

__all__ = [
    # Pagination
    "split_text_chunks_for_display",
    "should_paginate_chunks", 
    "calculate_chunk_pages",
    
    # Text formatting
    "create_chunk_header",
    "format_chunk_content",
    "create_chunk_separators",
    "format_page_info",
    "wrap_chunk_text",
    
    # Figure creation
    "create_text_chunk_figure",
    "setup_chunk_figure"
]