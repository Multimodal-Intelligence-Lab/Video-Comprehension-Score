import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from .._design_system import (
    VCSColors, VCSTypography, create_professional_figure,
    style_metric_visualization, add_professional_legend
)
from ._utils import (
    create_precision_load_sharing_figure,
    create_recall_load_sharing_figure
)

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
    # Create professional figure
    fig, ax = create_professional_figure(
        title="Local Alignment Score (LAS) Analysis",
        subtitle="Precision and Recall Component Breakdown"
    )
    
    # Create subplots manually for better control
    ax.remove()  # Remove the default axis
    axes = fig.subplots(1, 2)
    
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
    
    bars = ax_precision.bar(x_indices, precision_sim_values, 
                           alpha=0.8, color=VCSColors.PRECISION_COLOR,
                           edgecolor=VCSColors.GRAY_LIGHT, linewidth=0.5)
    
    for i, sim in enumerate(precision_sim_values):
        if sim > 0.5:
            ax_precision.annotate(f"{sim:.2f}", 
                                xy=(i, sim), 
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
    
    # Add the average line with professional styling
    ax_precision.axhline(y=precision_las, color=VCSColors.ACCENT, linestyle='--', linewidth=2)
    
    # Add a text annotation for the average with professional styling
    if precision_las > 0.95:
        avg_text_y = 0.9
    else:
        avg_text_y = min(precision_las + 0.07, 0.95)
    
    ax_precision.text(len(precision_sim_values) * 0.5, avg_text_y, 
                     f'Average: {precision_las:.4f}',
                     ha='center', va='bottom', color=VCSColors.ACCENT,
                     fontsize=VCSTypography.BODY_SIZE,
                     fontweight=VCSTypography.BOLD,
                     bbox=dict(facecolor=VCSColors.WHITE, alpha=0.9, 
                              edgecolor=VCSColors.ACCENT, 
                              boxstyle='round,pad=0.4',
                              linewidth=1.5))
    
    ax_precision.set_xlabel('Generation Index')
    ax_precision.set_ylabel('Similarity Value')
    ax_precision.set_title(f'Precision LAS: {precision_las:.4f}')
    ax_precision.set_ylim(0, 1.05)
    
    match_text = "Key Matches:\n"
    for g_idx, r_idx in precision_matches[:8]:  # Show fewer for cleaner look
        match_text += f"Gen {g_idx} → Ref {r_idx}\n"
    if len(precision_matches) > 8:
        match_text += f"... +{len(precision_matches) - 8} more"
    
    ax_precision.text(0.05, 0.95, match_text, 
                    transform=ax_precision.transAxes, 
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor=VCSColors.GRAY_BG, 
                             alpha=0.9,
                             edgecolor=VCSColors.GRAY_LIGHT),
                    fontsize=VCSTypography.CAPTION_SIZE,
                    color=VCSColors.GRAY_DARK)
    
    ax_recall = axes[1]
    
    x_indices = np.arange(len(recall_sim_values))
    
    bars = ax_recall.bar(x_indices, recall_sim_values, 
                        alpha=0.8, color=VCSColors.RECALL_COLOR,
                        edgecolor=VCSColors.GRAY_LIGHT, linewidth=0.5)
    
    for i, sim in enumerate(recall_sim_values):
        if sim > 0.5:
            ax_recall.annotate(f"{sim:.2f}", 
                            xy=(i, sim), 
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
    
    # Add the average line with professional styling
    ax_recall.axhline(y=recall_las, color=VCSColors.PRIMARY, linestyle='--', linewidth=2)
    
    # Add a text annotation for the average with professional styling
    if recall_las > 0.95:
        avg_text_y = 0.9
    else:
        avg_text_y = min(recall_las + 0.07, 0.95)
    
    ax_recall.text(len(recall_sim_values) * 0.5, avg_text_y, 
                  f'Average: {recall_las:.4f}',
                  ha='center', va='bottom', color=VCSColors.PRIMARY,
                  fontsize=VCSTypography.BODY_SIZE,
                  fontweight=VCSTypography.BOLD,
                  bbox=dict(facecolor=VCSColors.WHITE, alpha=0.9, 
                           edgecolor=VCSColors.PRIMARY, 
                           boxstyle='round,pad=0.4',
                           linewidth=1.5))
    
    ax_recall.set_xlabel('Reference Index')
    ax_recall.set_ylabel('Similarity Value')
    ax_recall.set_title(f'Recall LAS: {recall_las:.4f}')
    ax_recall.set_ylim(0, 1.05)
    
    match_text = "Key Matches:\n"
    for g_idx, r_idx in recall_matches[:8]:  # Show fewer for cleaner look
        match_text += f"Ref {r_idx} → Gen {g_idx}\n"
    if len(recall_matches) > 8:
        match_text += f"... +{len(recall_matches) - 8} more"
    
    ax_recall.text(0.05, 0.95, match_text, 
                 transform=ax_recall.transAxes, 
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', 
                          facecolor=VCSColors.GRAY_BG, 
                          alpha=0.9,
                          edgecolor=VCSColors.GRAY_LIGHT),
                 fontsize=VCSTypography.CAPTION_SIZE,
                 color=VCSColors.GRAY_DARK)
    
    # Update the main title to use professional styling (already set in create_professional_figure)
    # Add final F1 score as subtitle if not already included
    if not fig._suptitle or 'F1' not in fig._suptitle.get_text():
        current_title = fig._suptitle.get_text() if fig._suptitle else "Local Alignment Score (LAS) Analysis"
        fig.suptitle(f"{current_title}\nOverall F1 Score: {f1_las:.4f}", 
                    fontsize=VCSTypography.TITLE_SIZE,
                    fontweight=VCSTypography.BOLD,
                    color=VCSColors.GRAY_DARK)
    
    plt.tight_layout()
    return fig

def visualize_las_load_sharing(internals: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """Create simple load sharing visualizations matching best match style.
    
    Returns separate precision and recall figures like best match does.
    """
    precision_fig = create_precision_load_sharing_figure(internals)
    recall_fig = create_recall_load_sharing_figure(internals)
    
    return {
        'precision_details': precision_fig,
        'recall_details': recall_fig
    }