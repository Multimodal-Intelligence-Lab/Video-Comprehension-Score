import matplotlib.pyplot as plt
from typing import Dict, Any

def create_load_sharing_details_figure(las_internals: Dict[str, Any]) -> plt.Figure:
    """Create a detailed figure showing load sharing analysis for both directions."""
    
    precision_internals = las_internals.get('precision_internals', {})
    recall_internals = las_internals.get('recall_internals', {})
    
    # Create figure with 2 columns (precision and recall)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    fig.suptitle('LAS Load Sharing Analysis - Detailed Breakdown', fontsize=16, fontweight='bold')
    
    # Precision details (left panel)
    _create_direction_details_panel(ax1, precision_internals, "Precision")
    
    # Recall details (right panel)  
    _create_direction_details_panel(ax2, recall_internals, "Recall")
    
    plt.tight_layout()
    return fig

def _create_direction_details_panel(ax: plt.Axes, direction_internals: Dict[str, Any], direction: str) -> None:
    """Create a detailed panel showing load sharing for one direction."""
    from .text_formatting import create_load_sharing_header, format_load_sharing_details, format_direction_summary
    
    load_sharing_cases = direction_internals.get('load_sharing_details', [])
    
    if not load_sharing_cases:
        ax.text(0.5, 0.5, f"No load sharing detected in {direction.lower()} direction\n\n"
                          f"All chunks have unique best matches.", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.set_title(f'{direction} Direction - No Load Sharing', fontweight='bold')
        ax.axis('off')
        return
    
    # Create comprehensive text display
    text_content = create_load_sharing_header("SEMANTIC LOAD SHARING ANALYSIS", direction.lower())
    
    # Add details for each load sharing case (limit to first 3 for readability)
    display_cases = load_sharing_cases[:3]
    for i, case in enumerate(display_cases):
        text_content += f"CASE {i+1}:\n"
        text_content += format_load_sharing_details(case, direction.lower())
    
    if len(load_sharing_cases) > 3:
        remaining = len(load_sharing_cases) - 3
        text_content += f"... and {remaining} more cases\n\n"
    
    # Add direction summary
    text_content += format_direction_summary(direction_internals, direction.lower())
    
    # Display the text
    ax.text(0.01, 0.99, text_content,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=9,
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.1))
    
    ax.set_title(f'{direction} Direction - {len(load_sharing_cases)} Load Sharing Cases', 
                 fontweight='bold')
    ax.axis('off')