import matplotlib.pyplot as plt
from typing import Dict, Any
from .text_formatting import create_section_header, format_segment_details

def create_precision_details_section(ax: plt.Axes, precision_match_details: Dict[str, Any],
                                    context_cutoff_value: float, context_window_control: float) -> None:
    """Create the precision matching details section."""
    if not precision_match_details or 'segments' not in precision_match_details:
        _display_no_data_message(ax, "No detailed precision match data available")
        return
    
    precision_segments = precision_match_details.get('segments', [])
    
    # Only process the first 5 segments to keep it readable
    display_segments = precision_segments[:5]
    
    # Create structured text display
    precision_text = create_section_header(
        "PRECISION MATCHING DETAILS (Generation → Reference)",
        context_cutoff_value, 
        context_window_control
    )
    
    # Add segment details
    for segment in display_segments:
        precision_text += format_segment_details(segment, "generation")
    
    # Display the text
    ax.text(0.01, 0.99, precision_text, 
           transform=ax.transAxes,
           fontsize=9, family='monospace',
           verticalalignment='top', horizontalalignment='left')
    
    # Add a note if there are more items
    if len(precision_segments) > 5:
        _add_truncation_note(ax, len(precision_segments), "generation segments")

def _display_no_data_message(ax: plt.Axes, message: str) -> None:
    """Display a message when no data is available."""
    ax.text(0.5, 0.5, message, 
           ha='center', va='center', fontsize=14, 
           transform=ax.transAxes)

def _add_truncation_note(ax: plt.Axes, total_count: int, item_type: str) -> None:
    """Add a note about truncated display."""
    note_text = f"Note: Showing first 5 of {total_count} {item_type}"
    ax.text(0.5, 0.01, note_text, 
           transform=ax.transAxes,
           ha='center', va='bottom', fontsize=9, fontweight='bold')