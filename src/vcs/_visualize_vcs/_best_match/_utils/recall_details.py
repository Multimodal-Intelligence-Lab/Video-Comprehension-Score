import matplotlib.pyplot as plt
from typing import Dict, Any
from .text_formatting import create_section_header, format_segment_details

def create_recall_details_section(ax: plt.Axes, recall_match_details: Dict[str, Any],
                                 context_cutoff_value: float, context_window_control: float) -> None:
    """Create the recall matching details section."""
    if not recall_match_details or 'segments' not in recall_match_details:
        _display_no_data_message(ax, "No detailed recall match data available")
        return
    
    recall_segments = recall_match_details.get('segments', [])
    
    # Only process the first 5 segments to keep it readable
    display_segments = recall_segments[:5]
    
    # Create structured text display
    recall_text = create_section_header(
        "RECALL MATCHING DETAILS (Reference → Generation)",
        context_cutoff_value, 
        context_window_control
    )
    
    # Add segment details
    for segment in display_segments:
        recall_text += format_segment_details(segment, "reference")
    
    # Display the text
    ax.text(0.01, 0.99, recall_text, 
           transform=ax.transAxes,
           fontsize=9, family='monospace',
           verticalalignment='top', horizontalalignment='left')
    
    # Add a note if there are more items
    if len(recall_segments) > 5:
        _add_truncation_note(ax, len(recall_segments), "reference segments")

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