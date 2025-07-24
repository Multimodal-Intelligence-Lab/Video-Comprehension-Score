import matplotlib.pyplot as plt
from typing import Dict, Any, List

def create_base_calculation_figure(title: str, page_num: int, total_pages: int) -> tuple:
    """Create base figure and axis for line NAS calculation details."""
    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.axis('off')
    
    # Add title with page information
    page_title = f"{title} (Page {page_num} of {total_pages})" if total_pages > 1 else title
    fig.suptitle(page_title, fontsize=16, y=0.98)
    
    return fig, ax

def finalize_calculation_figure(fig: plt.Figure, page_num: int, total_pages: int) -> None:
    """Apply final layout and formatting to calculation figure."""
    # Add footer with page info if multiple pages
    if total_pages > 1:
        fig.text(0.5, 0.03, f"Page {page_num} of {total_pages}", ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

def create_empty_segments_figure(title: str, metric_type: str) -> plt.Figure:
    """Create a figure for when no segment data is available."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.text(0.5, 0.5, f"No segment data available for {metric_type} Line NAS", 
           ha='center', va='center', fontsize=14)
    fig.suptitle(title, fontsize=16)
    return fig