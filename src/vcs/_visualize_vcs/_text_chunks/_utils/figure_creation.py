import matplotlib.pyplot as plt
from typing import List, Tuple
from .text_formatting import (
    create_chunk_header, format_chunk_content, create_content_footer, format_page_info
)
from .pagination import get_page_range_info

def setup_chunk_figure(figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, plt.Axes]:
    """Set up basic figure and axis for chunk display."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    return fig, ax

def create_text_chunk_figure(title: str, chunks_with_indices: List[Tuple[int, str]], 
                            total_chunks: int, page_num: int, total_pages: int,
                            chunk_size: int = 1) -> plt.Figure:
    """Create a figure for displaying text chunks in structured text format."""
    fig, ax = setup_chunk_figure()
    
    # Create structured text content
    text_content = create_chunk_header(title, total_chunks, chunk_size, page_num, total_pages)
    
    # Add formatted chunk content
    text_content += format_chunk_content(chunks_with_indices)
    
    # Add footer
    text_content += create_content_footer()
    
    # Add page information at the bottom if multiple pages
    if total_pages > 1:
        start_chunk, end_chunk = get_page_range_info(chunks_with_indices)
        text_content += format_page_info(page_num, total_pages, start_chunk, end_chunk, total_chunks)
    
    # Display the structured text - adjusted position to leave room for footer
    ax.text(0.01, 0.95, text_content, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    # Add title to the figure
    fig.suptitle(f'{title} (Structured Text View)', fontsize=16, y=0.98)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings with text-heavy content
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    
    return fig