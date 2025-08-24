import matplotlib.pyplot as plt
from typing import Dict, Any
from .text_formatting import create_load_sharing_header, format_load_sharing_details

def create_precision_load_sharing_section(ax: plt.Axes, precision_internals: Dict[str, Any]) -> None:
    """Create the precision load sharing details section."""
    load_sharing_cases = precision_internals.get('load_sharing_details', [])
    
    if not load_sharing_cases:
        _display_no_load_sharing_message(ax, "precision")
        return
    
    # Create structured text display
    precision_text = create_load_sharing_header(
        "SEMANTIC LOAD SHARING ANALYSIS - PRECISION", 
        "precision"
    )
    
    # Add load sharing case details (limit to first 4 for readability)
    display_cases = load_sharing_cases[:4]
    for i, case in enumerate(display_cases):
        precision_text += f"CASE {i+1}:\n"
        precision_text += format_load_sharing_details(case, "precision")
    
    if len(load_sharing_cases) > 4:
        remaining = len(load_sharing_cases) - 4
        precision_text += f"... and {remaining} more cases (see summary for complete statistics)\n\n"
    
    # Add overall impact summary
    original_las = precision_internals.get('original_las', 0.0)
    adjusted_las = precision_internals.get('adjusted_las', 0.0)
    reduction = original_las - adjusted_las
    reduction_pct = (reduction / original_las * 100) if original_las > 0 else 0
    
    precision_text += "OVERALL IMPACT:\n"
    precision_text += f"Original Precision LAS: {original_las:.4f}\n"
    precision_text += f"Adjusted Precision LAS: {adjusted_las:.4f}\n"
    precision_text += f"Total reduction: {reduction:.4f} ({reduction_pct:.1f}%)\n"
    
    # Display the text
    ax.text(0.01, 0.99, precision_text, 
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='left',
           fontsize=9,
           fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.1))
    
    ax.set_title(f'Precision Load Sharing - {len(load_sharing_cases)} Cases Detected', 
                 fontweight='bold')
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
           fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    ax.set_title(f'{direction.title()} Direction - No Load Sharing', fontweight='bold')
    ax.axis('off')