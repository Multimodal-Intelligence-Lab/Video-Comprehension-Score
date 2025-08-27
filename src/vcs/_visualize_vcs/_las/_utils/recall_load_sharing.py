import matplotlib.pyplot as plt
from typing import Dict, Any
from .text_formatting import create_las_section_header, format_load_sharing_case, format_direction_summary

def create_recall_load_sharing_section(ax: plt.Axes, recall_internals: Dict[str, Any]) -> None:
    """Create the recall load sharing details section."""
    load_sharing_cases = recall_internals.get('load_sharing_details', [])
    
    if not load_sharing_cases:
        _display_no_load_sharing_message(ax, "recall")
        return
    
    # Create simple text display (limit to first 4 cases for readability)
    recall_text = create_las_section_header("RECALL LOAD SHARING DETAILS (Reference → Generated)")
    
    display_cases = load_sharing_cases[:4]
    for i, case in enumerate(display_cases, 1):
        recall_text += format_load_sharing_case(case, i, "recall")
    
    if len(load_sharing_cases) > 4:
        remaining = len(load_sharing_cases) - 4
        recall_text += f"... and {remaining} more cases\n\n"
    
    # Add summary
    recall_text += format_direction_summary(recall_internals, "recall")
    
    # Display the text with monospace font like best match
    ax.text(0.01, 0.99, recall_text, 
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='left',
           fontsize=9,
           fontfamily='monospace')
    
    ax.set_title(f'Recall Load Sharing - {len(load_sharing_cases)} Cases', fontweight='bold')
    ax.axis('off')

def _display_no_load_sharing_message(ax: plt.Axes, direction: str) -> None:
    """Display a message when no load sharing is detected."""
    message = f"No load sharing detected in {direction} direction.\n\n"
    if direction == "recall":
        message += "Each reference chunk has a unique best match generated chunk.\n"
    else:
        message += "Each generated chunk has a unique best match reference chunk.\n"
    message += "No semantic load sharing penalty applied."
    
    ax.text(0.5, 0.5, message,
           transform=ax.transAxes,
           verticalalignment='center',
           horizontalalignment='center',
           fontsize=12)
    
    ax.set_title(f'{direction.title()} Direction - No Load Sharing', fontweight='bold')
    ax.axis('off')