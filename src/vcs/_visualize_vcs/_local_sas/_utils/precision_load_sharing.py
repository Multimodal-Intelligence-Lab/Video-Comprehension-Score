import matplotlib.pyplot as plt
from typing import Dict, Any
from .text_formatting import create_las_section_header, format_load_sharing_case, format_direction_summary

def create_precision_load_sharing_section(ax: plt.Axes, precision_internals: Dict[str, Any]) -> None:
    """Create the precision load sharing details section."""
    load_sharing_cases = precision_internals.get('load_sharing_details', [])
    
    if not load_sharing_cases:
        _display_no_load_sharing_message(ax, "precision")
        return
    
    # Create simple text display (limit to first 4 cases for readability)
    precision_text = create_las_section_header("PRECISION LOAD SHARING DETAILS (Generation → Reference)")
    
    display_cases = load_sharing_cases[:4]
    for i, case in enumerate(display_cases, 1):
        precision_text += format_load_sharing_case(case, i, "precision")
    
    if len(load_sharing_cases) > 4:
        remaining = len(load_sharing_cases) - 4
        precision_text += f"... and {remaining} more cases\n\n"
    
    # Add summary
    precision_text += format_direction_summary(precision_internals, "precision")
    
    # Display the text with monospace font like best match
    ax.text(0.01, 0.99, precision_text, 
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='left',
           fontsize=9,
           fontfamily='monospace')
    
    ax.set_title(f'Precision Load Sharing - {len(load_sharing_cases)} Cases', fontweight='bold')
    ax.axis('off')

def _display_no_load_sharing_message(ax: plt.Axes, direction: str) -> None:
    """Display a message when no load sharing is detected."""
    message = f"No load sharing detected in {direction} direction.\n\n"
    message += "Each generated chunk has a unique best match reference chunk.\n"
    message += "No semantic load sharing penalty applied."
    
    ax.text(0.5, 0.5, message,
           transform=ax.transAxes,
           verticalalignment='center',
           horizontalalignment='center',
           fontsize=12)
    
    ax.set_title(f'{direction.title()} Direction - No Load Sharing', fontweight='bold')
    ax.axis('off')