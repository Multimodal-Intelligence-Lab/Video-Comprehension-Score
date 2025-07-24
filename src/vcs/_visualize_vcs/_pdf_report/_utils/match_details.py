import matplotlib.pyplot as plt
from typing import Dict, List

def create_precision_match_details_page(
    batch_segments: List[Dict], 
    context_cutoff_value: float, 
    context_window_control: float,
    page_num: int,
    total_pages: int
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create header text
    header_text = _create_precision_header(context_cutoff_value, context_window_control)
    
    # Process segments and create content
    precision_text, y_positions = _process_precision_segments(batch_segments, header_text)
    
    # Apply alternating background shading for better readability
    _apply_background_shading(ax, y_positions)
    
    # Display the text
    ax.text(0.01, 0.99, precision_text, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    # Set title
    page_info = f"Precision Matching Details (Page {page_num} of {total_pages})"
    fig.suptitle(page_info, fontsize=14, y=0.99)
    
    return fig


def create_recall_match_details_page(
    batch_segments: List[Dict], 
    context_cutoff_value: float, 
    context_window_control: float,
    page_num: int,
    total_pages: int
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create header text
    header_text = _create_recall_header(context_cutoff_value, context_window_control)
    
    # Process segments and create content
    recall_text, y_positions = _process_recall_segments(batch_segments, header_text)
    
    # Apply alternating background shading
    _apply_background_shading(ax, y_positions)
    
    # Display the text
    ax.text(0.01, 0.99, recall_text, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    # Set title
    page_info = f"Recall Matching Details (Page {page_num} of {total_pages})"
    fig.suptitle(page_info, fontsize=14, y=0.99)
    
    return fig


def _create_precision_header(context_cutoff_value: float, context_window_control: float) -> str:
    header_text = "PRECISION MATCHING DETAILS (Generation → Reference)\n"
    header_text += "=" * 80 + "\n\n"
    header_text += f"Context cutoff value: {context_cutoff_value}  |  Context window control: {context_window_control}\n\n"
    return header_text


def _create_recall_header(context_cutoff_value: float, context_window_control: float) -> str:
    header_text = "RECALL MATCHING DETAILS (Reference → Generation)\n"
    header_text += "=" * 80 + "\n\n"
    header_text += f"Context cutoff value: {context_cutoff_value}  |  Context window control: {context_window_control}\n\n"
    return header_text


def _process_precision_segments(batch_segments: List[Dict], header_text: str) -> tuple:
    precision_text = header_text
    y_positions = []
    
    for segment_idx, segment in enumerate(batch_segments):
        idx = segment.get('index', -1)
        if idx < 0:
            continue
        
        # Remember starting y position for this segment (approximate)
        y_start = 0.99 - (segment_idx * 0.22)  # Adjusted based on content size
        y_positions.append(y_start)
            
        # Header for each segment - with improved formatting
        segment_header = f"GENERATION INDEX: {idx}\n"
        segment_header += "-" * 80 + "\n"
        precision_text += segment_header
        
        is_valid = segment.get('valid', False)
        if not is_valid:
            reason = segment.get('reason', "Unknown")
            precision_text += f"Status: No valid matches found. Reason: {reason}\n\n"
            continue
        
        # Process segment details
        precision_text += _format_precision_segment_details(segment)
    
    return precision_text, y_positions


def _process_recall_segments(batch_segments: List[Dict], header_text: str) -> tuple:
    recall_text = header_text
    y_positions = []
    
    for segment_idx, segment in enumerate(batch_segments):
        idx = segment.get('index', -1)
        if idx < 0:
            continue
            
        # Remember starting y position for this segment
        y_start = 0.99 - (segment_idx * 0.22)
        y_positions.append(y_start)
        
        # Header for each segment
        recall_text += f"REFERENCE INDEX: {idx}\n"
        recall_text += "-" * 80 + "\n"
        
        is_valid = segment.get('valid', False)
        if not is_valid:
            reason = segment.get('reason', "Unknown")
            recall_text += f"Status: No valid matches found. Reason: {reason}\n\n"
            continue
        
        # Process segment details
        recall_text += _format_recall_segment_details(segment)
    
    return recall_text, y_positions


def _format_precision_segment_details(segment: Dict) -> str:
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
    
    # Formatted key-value pairs with consistent alignment using string formatting
    details_text = f"{'Mapping Window:':<25} {window_str}\n"
    details_text += f"{'Max Similarity:':<25} {max_val:.4f} at Reference Index: {max_idx}\n"
    details_text += f"{'Context Range:':<25} {context_range:.4f}\n"
    
    if context_applied:
        details_text += f"{'Context Window Applied:':<25} Yes\n"
        details_text += f"{'Context Window Size:':<25} {context_window:.4f}\n"
        details_text += f"{'Context Threshold:':<25} {context_threshold:.4f}\n"
    else:
        details_text += f"{'Context Window Applied:':<25} No\n"
        details_text += f"{'Context Threshold:':<25} Max Value ({max_val:.4f})\n"
    
    candidates = selection.get('candidates', [])
    details_text += f"{'Number of Candidates:':<25} {len(candidates)}\n"
    
    if candidates:
        details_text += "\nCandidate Details:\n"
        details_text += _format_candidates(candidates, "Ref Index")
    
    details_text += f"\n{'Selected Reference:':<25} {selected_idx}\n"
    details_text += f"{'Selection Reason:':<25} {selection_reason}\n"
    details_text += "\n" + "-" * 80 + "\n\n"
    
    return details_text


def _format_recall_segment_details(segment: Dict) -> str:
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
    
    # Consistent key-value formatting
    details_text = f"{'Mapping Window:':<25} {window_str}\n"
    details_text += f"{'Max Similarity:':<25} {max_val:.4f} at Generation Index: {max_idx}\n"
    details_text += f"{'Context Range:':<25} {context_range:.4f}\n"
    
    if context_applied:
        details_text += f"{'Context Window Applied:':<25} Yes\n"
        details_text += f"{'Context Window Size:':<25} {context_window:.4f}\n"
        details_text += f"{'Context Threshold:':<25} {context_threshold:.4f}\n"
    else:
        details_text += f"{'Context Window Applied:':<25} No\n"
        details_text += f"{'Context Threshold:':<25} Max Value ({max_val:.4f})\n"
    
    candidates = selection.get('candidates', [])
    details_text += f"{'Number of Candidates:':<25} {len(candidates)}\n"
    
    if candidates:
        details_text += "\nCandidate Details:\n"
        details_text += _format_candidates(candidates, "Gen Index")
    
    details_text += f"\n{'Selected Generation:':<25} {selected_idx}\n"
    details_text += f"{'Selection Reason:':<25} {selection_reason}\n"
    details_text += "\n" + "-" * 80 + "\n\n"
    
    return details_text


def _format_candidates(candidates: List[Dict], index_label: str) -> str:
    candidates_text = ""
    for i, candidate in enumerate(candidates):
        c_idx = candidate.get('index', -1)
        c_sim = candidate.get('similarity', 0)
        c_in_window = candidate.get('in_window', False)
        c_distance = candidate.get('distance', 0)
        c_selected = candidate.get('is_selected', False)
        
        # Improved candidate formatting with consistent indentation
        candidates_text += f"  {i+1}. {index_label + ':':<15} {c_idx}\n"
        candidates_text += f"     {'Similarity:':<15} {c_sim:.4f}\n"
        candidates_text += f"     {'In Window:':<15} {'Yes' if c_in_window else 'No'}\n"
        candidates_text += f"     {'Distance:':<15} {c_distance}\n"
        candidates_text += f"     {'Selected:':<15} {'Yes' if c_selected else 'No'}\n"
    
    return candidates_text


def _apply_background_shading(ax, y_positions: List[float]) -> None:
    for i in range(0, len(y_positions), 2):
        if i < len(y_positions):
            # Height approximation based on content
            height = 0.22 if i + 1 < len(y_positions) else 0.3
            rect = plt.Rectangle((0.01, y_positions[i] - height), 0.98, height, 
                                 fill=True, color='#f5f5f5', alpha=0.5, transform=ax.transAxes)
            ax.add_patch(rect)