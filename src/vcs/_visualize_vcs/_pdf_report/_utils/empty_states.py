import matplotlib.pyplot as plt

def create_empty_las_load_sharing_page() -> plt.Figure:
    """Create an empty LAS load sharing page when data is unavailable."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    
    ax.text(0.5, 0.5, 
           "LAS Load Sharing Analysis\n\n"
           "No load sharing data available or\n"
           "an error occurred generating the visualization.",
           ha='center', va='center', transform=ax.transAxes,
           fontsize=16, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
    
    ax.set_title('LAS Load Sharing - Data Unavailable', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    return fig


def create_empty_precision_matches_page() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    summary_text = _create_empty_table(
        "PRECISION MATCHES (Generation → Reference)",
        ["Gen Index", "Ref Index", "Similarity", "Generation Text", "Reference Text"],
        "No precision matches found"
    )
    
    ax.text(0.01, 0.99, summary_text, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    fig.suptitle("Precision Matches Summary", fontsize=14, y=0.99)
    return fig


def create_empty_recall_matches_page() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    summary_text = _create_empty_table(
        "RECALL MATCHES (Reference → Generation)",
        ["Ref Index", "Gen Index", "Similarity", "Reference Text", "Generation Text"],
        "No recall matches found"
    )
    
    ax.text(0.01, 0.99, summary_text, 
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    fig.suptitle("Recall Matches Summary", fontsize=14, y=0.99)
    return fig


def _create_empty_table(title: str, headers: list, empty_message: str) -> str:
    summary_text = "MATCHING SUMMARY TABLE\n"
    summary_text += "=" * 110 + "\n\n"
    
    summary_text += title + "\n"
    summary_text += "-" * 110 + "\n"
    summary_text += "┌" + "─" * 108 + "┐\n"
    
    # Create header row
    if len(headers) == 5:
        summary_text += "│ {:^10} │ {:^10} │ {:^15} │ {:^30} │ {:^30} │\n".format(*headers)
    else:
        # Fallback for different header structures
        header_format = "│ " + " │ ".join(["{:^10}"] * len(headers)) + " │\n"
        summary_text += header_format.format(*headers)
    
    summary_text += "├" + "─" * 108 + "┤\n"
    summary_text += "│ {:^108} │\n".format(empty_message)
    summary_text += "└" + "─" * 108 + "┘\n"
    
    return summary_text