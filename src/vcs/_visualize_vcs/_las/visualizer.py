import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def visualize_las(internals: Dict[str, Any]) -> plt.Figure:
    """Create a visualization of Local Alignment Score (LAS) precision and recall components.
    
    Displays LAS precision and recall as side-by-side bar charts showing similarity
    values for each matched segment pair, with averages and match details.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain LAS metrics and alignment data.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure with two subplots showing precision and recall LAS components
        with similarity values, averages, and match information.
    
    Examples
    --------
    **Basic Usage:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True,
            return_all_metrics=True
        )
        fig = visualize_las(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_best_match : See detailed match analysis
    visualize_similarity_matrix : See underlying similarity computations
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    precision_sim_values = np.array(internals['alignment']['precision']['similarity_values'])
    recall_sim_values = np.array(internals['alignment']['recall']['similarity_values'])
    precision_matches = internals['alignment']['precision']['matches']
    recall_matches = internals['alignment']['recall']['matches']
    
    las_metrics = internals['metrics']['las']
    precision_las = las_metrics['precision']
    recall_las = las_metrics['recall']
    f1_las = las_metrics['f1']
    
    ax_precision = axes[0]
    
    x_indices = np.arange(len(precision_sim_values))
    
    bars = ax_precision.bar(x_indices, precision_sim_values, alpha=0.7, color='skyblue')
    
    for i, sim in enumerate(precision_sim_values):
        if sim > 0.5:
            ax_precision.annotate(f"{sim:.2f}", 
                                xy=(i, sim), 
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
    
    # Add the average line with improved positioning and visibility
    ax_precision.axhline(y=precision_las, color='red', linestyle='--')
    
    # Add a text annotation for the average in a more visible position
    # If the average is close to 1.0, place it slightly lower to ensure visibility
    if precision_las > 0.95:
        avg_text_y = 0.9  # Position text lower when average is near top
    else:
        avg_text_y = min(precision_las + 0.07, 0.95)  # Place above line but not too high
    
    ax_precision.text(len(precision_sim_values) * 0.5, avg_text_y, 
                     f'Average: {precision_las:.4f}',
                     ha='center', va='bottom', color='red',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.3'))
    
    ax_precision.set_xlabel('Generation Index')
    ax_precision.set_ylabel('Similarity Value')
    ax_precision.set_title(f'Precision LAS: {precision_las:.4f}')
    ax_precision.set_ylim(0, 1.05)
    
    match_text = "Matches:\n"
    for g_idx, r_idx in precision_matches[:10]:
        match_text += f"Gen {g_idx} → Ref {r_idx}\n"
    if len(precision_matches) > 10:
        match_text += f"... and {len(precision_matches) - 10} more"
    
    ax_precision.text(0.05, 0.95, match_text, 
                    transform=ax_precision.transAxes, 
                    va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)
    
    ax_recall = axes[1]
    
    x_indices = np.arange(len(recall_sim_values))
    
    bars = ax_recall.bar(x_indices, recall_sim_values, alpha=0.7, color='salmon')
    
    for i, sim in enumerate(recall_sim_values):
        if sim > 0.5:
            ax_recall.annotate(f"{sim:.2f}", 
                            xy=(i, sim), 
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
    
    # Add the average line with improved visibility
    ax_recall.axhline(y=recall_las, color='blue', linestyle='--')
    
    # Add a text annotation for the average in a more visible position
    # If the average is close to 1.0, place it slightly lower to ensure visibility
    if recall_las > 0.95:
        avg_text_y = 0.9  # Position text lower when average is near top
    else:
        avg_text_y = min(recall_las + 0.07, 0.95)  # Place above line but not too high
    
    ax_recall.text(len(recall_sim_values) * 0.5, avg_text_y, 
                  f'Average: {recall_las:.4f}',
                  ha='center', va='bottom', color='blue',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue', boxstyle='round,pad=0.3'))
    
    ax_recall.set_xlabel('Reference Index')
    ax_recall.set_ylabel('Similarity Value')
    ax_recall.set_title(f'Recall LAS: {recall_las:.4f}')
    ax_recall.set_ylim(0, 1.05)
    
    match_text = "Matches:\n"
    for g_idx, r_idx in recall_matches[:10]:
        match_text += f"Ref {r_idx} → Gen {g_idx}\n"
    if len(recall_matches) > 10:
        match_text += f"... and {len(recall_matches) - 10} more"
    
    ax_recall.text(0.05, 0.95, match_text, 
                 transform=ax_recall.transAxes, 
                 va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=8)
    
    fig.suptitle(f'Local Alignment Score (LAS): {f1_las:.4f}', fontsize=16)
    fig.tight_layout()
    return fig