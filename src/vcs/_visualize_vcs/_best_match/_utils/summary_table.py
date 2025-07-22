import matplotlib.pyplot as plt
from typing import Dict, Any, List
from .text_formatting import (
    create_table_header, create_table_row, create_table_footer, create_empty_table_row,
    truncate_text
    # Note: determine_match_notes removed completely since we no longer need notes
)

def create_summary_table_section(ax: plt.Axes, internals: Dict[str, Any]) -> None:
    """Create the comprehensive summary table section without notes or statistics."""
    # Get text chunks for reference in the summary
    ref_chunks = internals['texts']['reference_chunks']
    gen_chunks = internals['texts']['generated_chunks']
    
    # Extract precision and recall matches
    precision_matches = internals['alignment']['precision']['matches']
    precision_sim_values = internals['alignment']['precision']['similarity_values']
    
    recall_matches = internals['alignment']['recall']['matches']
    recall_sim_values = internals['alignment']['recall']['similarity_values']
    
    # Note: We no longer need best_match_info for notes since we're not displaying them
    
    # Create the summary table text with streamlined headers
    summary_text = "MATCHING SUMMARY TABLE\n"
    summary_text += "=" * 120 + "\n\n"  # Reduced width since we have fewer columns
    
    # Create precision matches table (no notes)
    summary_text += _create_precision_matches_table(
        precision_matches, precision_sim_values, gen_chunks, ref_chunks
    )
    
    summary_text += "\n"
    
    # Create recall matches table (no notes)
    summary_text += _create_recall_matches_table(
        recall_matches, recall_sim_values, ref_chunks, gen_chunks
    )
    
    # Display the summary text
    ax.text(0.01, 0.99, summary_text, 
           transform=ax.transAxes,
           fontsize=9, family='monospace',
           verticalalignment='top', horizontalalignment='left')
    


def _create_precision_matches_table(precision_matches: List, precision_sim_values: List,
                                   gen_chunks: List, ref_chunks: List) -> str:
    """Create the precision matches table without notes column."""
    # Updated header without notes column - now only 5 columns instead of 6
    table_text = create_table_header(
        "PRECISION MATCHES (Generation → Reference)",
        ["Gen Index", "Ref Index", "Similarity", "Generation Text", "Reference Text"]
    )
    
    # Add rows for each precision match
    if precision_matches:
        for g_idx, r_idx in precision_matches:
            # Get similarity value
            sim_value = precision_sim_values[g_idx] if g_idx < len(precision_sim_values) else 0.0
            
            # Get text snippets - truncate if too long
            # Note: We can now allow longer text since we have more space without the notes column
            gen_text = truncate_text(gen_chunks[g_idx] if g_idx < len(gen_chunks) else "", 35)
            ref_text = truncate_text(ref_chunks[r_idx] if r_idx < len(ref_chunks) else "", 35)
            
            # Create row without notes - now only 5 values instead of 6
            table_text += _create_table_row_no_notes(g_idx, r_idx, sim_value, gen_text, ref_text)
    else:
        table_text += create_empty_table_row("No precision matches found")
    
    table_text += create_table_footer()
    return table_text

def _create_recall_matches_table(recall_matches: List, recall_sim_values: List,
                                ref_chunks: List, gen_chunks: List) -> str:
    """Create the recall matches table without notes column."""
    # Updated header without notes column
    table_text = create_table_header(
        "RECALL MATCHES (Reference → Generation)",
        ["Ref Index", "Gen Index", "Similarity", "Reference Text", "Generation Text"]
    )
    
    # Add rows for each recall match
    if recall_matches:
        for g_idx, r_idx in recall_matches:
            # Get similarity value
            sim_value = recall_sim_values[r_idx] if r_idx < len(recall_sim_values) else 0.0
            
            # Get text snippets - truncate if too long
            # Note: More space available for text content without notes
            ref_text = truncate_text(ref_chunks[r_idx] if r_idx < len(ref_chunks) else "", 35)
            gen_text = truncate_text(gen_chunks[g_idx] if g_idx < len(gen_chunks) else "", 35)
            
            # Create row without notes
            table_text += _create_table_row_no_notes(r_idx, g_idx, sim_value, ref_text, gen_text)
    else:
        table_text += create_empty_table_row("No recall matches found")
    
    table_text += create_table_footer()
    return table_text

def _create_table_row_no_notes(col1: Any, col2: Any, col3: Any, col4: str, col5: str) -> str:
    """Create a table row without the notes column - streamlined 5-column format."""
    # Updated formatting for 5 columns instead of 6, with more space for text content
    return "│ {:^10} │ {:^10} │ {:^15.4f} │ {:<35} │ {:<35} │\n".format(
        col1, col2, col3, col4, col5)