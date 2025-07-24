import matplotlib.pyplot as plt
from typing import List, Tuple


def create_precision_matches_summary_page(
    page_matches: List[Tuple],
    ref_chunks: List[str],
    gen_chunks: List[str],
    precision_sim_values: List[float],
    page_num: int,
    total_pages: int
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create the table text for this page
    summary_text = _create_table_header("PRECISION MATCHES (Generation → Reference)")
    summary_text += _create_precision_table_content(page_matches, ref_chunks, gen_chunks, precision_sim_values)
    summary_text += _close_table()
    
    # Display the summary text
    ax.text(0.01, 0.99, summary_text, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    page_title = f"Precision Matches Summary (Page {page_num} of {total_pages})"
    fig.suptitle(page_title, fontsize=14, y=0.99)
    
    return fig


def create_recall_matches_summary_page(
    page_matches: List[Tuple],
    ref_chunks: List[str],
    gen_chunks: List[str],
    recall_sim_values: List[float],
    page_num: int,
    total_pages: int
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create the table text for this page
    summary_text = _create_table_header("RECALL MATCHES (Reference → Generation)")
    summary_text += _create_recall_table_content(page_matches, ref_chunks, gen_chunks, recall_sim_values)
    summary_text += _close_table()
    
    # Display the summary text
    ax.text(0.01, 0.99, summary_text, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    page_title = f"Recall Matches Summary (Page {page_num} of {total_pages})"
    fig.suptitle(page_title, fontsize=14, y=0.99)
    
    return fig


def _create_table_header(title: str) -> str:
    summary_text = "MATCHING SUMMARY TABLE\n"
    summary_text += "=" * 110 + "\n\n"
    summary_text += title + "\n"
    summary_text += "-" * 110 + "\n"
    summary_text += "┌" + "─" * 108 + "┐\n"
    return summary_text


def _close_table() -> str:
    return "└" + "─" * 108 + "┘\n"


def _create_precision_table_content(
    page_matches: List[Tuple],
    ref_chunks: List[str],
    gen_chunks: List[str],
    precision_sim_values: List[float]
) -> str:
    content = "│ {:^10} │ {:^10} │ {:^15} │ {:^30} │ {:^30} │\n".format(
        "Gen Index", "Ref Index", "Similarity", "Generation Text", "Reference Text")
    content += "├" + "─" * 108 + "┤\n"
    
    # Add rows for this page's matches
    for g_idx, r_idx in page_matches:
        # Get similarity value
        sim_value = precision_sim_values[g_idx] if g_idx < len(precision_sim_values) else 0.0
        
        # Get text snippets - truncate if too long
        gen_text = _truncate_text(gen_chunks, g_idx, 30)
        ref_text = _truncate_text(ref_chunks, r_idx, 30)
        
        # Add the row
        content += "│ {:^10} │ {:^10} │ {:^15.4f} │ {:<30} │ {:<30} │\n".format(
            g_idx, r_idx, sim_value, gen_text, ref_text)
    
    return content


def _create_recall_table_content(
    page_matches: List[Tuple],
    ref_chunks: List[str],
    gen_chunks: List[str],
    recall_sim_values: List[float]
) -> str:
    content = "│ {:^10} │ {:^10} │ {:^15} │ {:^30} │ {:^30} │\n".format(
        "Ref Index", "Gen Index", "Similarity", "Reference Text", "Generation Text")
    content += "├" + "─" * 108 + "┤\n"
    
    # Add rows for this page's matches
    for g_idx, r_idx in page_matches:
        # Get similarity value
        sim_value = recall_sim_values[r_idx] if r_idx < len(recall_sim_values) else 0.0
        
        # Get text snippets - truncate if too long
        ref_text = _truncate_text(ref_chunks, r_idx, 30)
        gen_text = _truncate_text(gen_chunks, g_idx, 30)
        
        # Add the row
        content += "│ {:^10} │ {:^10} │ {:^15.4f} │ {:<30} │ {:<30} │\n".format(
            r_idx, g_idx, sim_value, ref_text, gen_text)
    
    return content


def _truncate_text(chunks: List[str], index: int, max_length: int) -> str:
    if index >= len(chunks):
        return ""
    
    text = chunks[index]
    if len(text) > max_length + 3:  # +3 for "..."
        return text[:max_length] + "..."
    return text