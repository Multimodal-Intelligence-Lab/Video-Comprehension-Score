import numpy as np
from typing import Dict, Any, List, Tuple

def create_section_header(title: str, context_cutoff_value: float, context_window_control: float) -> str:
    """Create a section header with title and configuration parameters."""
    header_text = f"{title}\n"
    header_text += "=" * 80 + "\n\n"
    header_text += f"Context cutoff value: {context_cutoff_value}  |  Context window control: {context_window_control}\n\n"
    return header_text

def format_segment_details(segment: Dict[str, Any], segment_type: str = "generation") -> str:
    """Format detailed information for a single segment."""
    idx = segment.get('index', -1)
    if idx < 0:
        return ""
        
    # Header for each segment
    index_label = "GENERATION INDEX" if segment_type == "generation" else "REFERENCE INDEX"
    target_label = "Reference Index" if segment_type == "generation" else "Generation Index"
    target_prefix = "Ref" if segment_type == "generation" else "Gen"
    selected_label = "Selected Reference" if segment_type == "generation" else "Selected Generation"
    
    segment_text = f"{index_label}: {idx}\n"
    segment_text += "-" * 80 + "\n"
    
    is_valid = segment.get('valid', False)
    if not is_valid:
        reason = segment.get('reason', "Unknown")
        segment_text += f"Status: No valid matches found. Reason: {reason}\n\n"
        return segment_text
    
    mapping_window = segment.get('mapping_window', {})
    window_str = f"[{mapping_window.get('start', '?')}, {mapping_window.get('end', '?')})"
    
    selection = segment.get('selection', {})
    max_val = selection.get('max_value', 0)
    max_idx = selection.get('max_index', -1)
    
    # Context window calculation
    context_range = selection.get('context_range', 0)
    context_window = selection.get('context_window', 0)
    context_threshold = selection.get('context_threshold', 0)
    context_applied = selection.get('context_window_applied', False)
    selected_idx = selection.get('selected_index', -1)
    selection_reason = selection.get('selection_reason', "")
    
    # Add basic information
    segment_text += f"Mapping Window: {window_str}\n"
    segment_text += f"Max Similarity: {max_val:.4f} at {target_label}: {max_idx}\n"
    
    # Add context window calculation details
    segment_text += f"Context Range: {context_range:.4f}\n"
    
    if context_applied:
        segment_text += f"Context Window Applied: Yes\n"
        segment_text += f"Context Window Size: {context_window:.4f}\n"
        segment_text += f"Context Threshold: {context_threshold:.4f}\n"
    else:
        segment_text += f"Context Window Applied: No\n"
        segment_text += f"Context Threshold: Max Value ({max_val:.4f})\n"
    
    # Add candidate information
    candidates = selection.get('candidates', [])
    segment_text += f"Number of Candidates: {len(candidates)}\n"
    
    if candidates:
        segment_text += "\nCandidate Details:\n"
        segment_text += format_candidate_details(candidates, target_prefix)
    
    # Final selection information
    segment_text += f"\n{selected_label}: {selected_idx}\n"
    segment_text += f"Selection Reason: {selection_reason}\n"
    segment_text += "\n" + "-" * 80 + "\n\n"
    
    return segment_text

def format_candidate_details(candidates: List[Dict], index_prefix: str) -> str:
    """Format candidate details for display."""
    candidate_text = ""
    for i, candidate in enumerate(candidates):
        c_idx = candidate.get('index', -1)
        c_sim = candidate.get('similarity', 0)
        c_in_window = candidate.get('in_window', False)
        c_distance = candidate.get('distance', 0)
        c_selected = candidate.get('is_selected', False)
        
        # Format each candidate with indentation
        candidate_text += f"  {i+1}. {index_prefix} Index: {c_idx}\n"
        candidate_text += f"     Similarity: {c_sim:.4f}\n"
        candidate_text += f"     In Window: {'Yes' if c_in_window else 'No'}\n"
        candidate_text += f"     Distance: {c_distance}\n"
        candidate_text += f"     Selected: {'Yes' if c_selected else 'No'}\n"
    
    return candidate_text

def create_table_header(table_title: str, headers: List[str]) -> str:
    """Create table header with title and column headers optimized for 5-column format."""
    header_text = f"{table_title}\n"
    header_text += "-" * 120 + "\n"  # Reduced width for 5 columns
    header_text += "┌" + "─" * 118 + "┐\n"
    # Updated format for 5 columns with more space for text content
    header_text += "│ {:^10} │ {:^10} │ {:^15} │ {:^35} │ {:^35} │\n".format(*headers)
    header_text += "├" + "─" * 118 + "┤\n"
    return header_text

def create_table_row(col1: Any, col2: Any, col3: Any, col4: str, col5: str) -> str:
    """Create a single table row optimized for 5-column format without notes."""
    # Updated to handle 5 columns instead of 6, with expanded text space
    return "│ {:^10} │ {:^10} │ {:^15.4f} │ {:<35} │ {:<35} │\n".format(
        col1, col2, col3, col4, col5)

def create_table_footer() -> str:
    """Create table footer optimized for 5-column format."""
    return "└" + "─" * 118 + "┘\n"  # Reduced width for 5 columns

def create_empty_table_row(message: str) -> str:
    """Create an empty table row with centered message for 5-column format."""
    return "│ {:^114} │\n".format(message)  # Adjusted width for 5 columns

def truncate_text(text: str, max_length: int = 35) -> str:
    """Truncate text if it's too long. Increased default length since we have more space without notes."""
    if len(text) > max_length + 3:  # +3 for "..."
        return text[:max_length] + "..."
    return text