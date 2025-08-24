import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from ._utils import (
    create_load_sharing_details_figure,
    create_precision_load_sharing_section,
    create_recall_load_sharing_section,
    create_load_sharing_summary_table_figure
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

def visualize_las_load_sharing(internals: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """Create detailed visualizations of LAS load sharing analysis for precision and recall.
    
    Generates comprehensive visualizations showing semantic load sharing cases,
    penalties applied, and before/after comparisons for both precision and recall directions.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain LAS internals with load sharing data.
    
    Returns
    -------
    dict
        Dictionary containing multiple matplotlib figures:
        
        * ``'load_sharing_details'`` : Figure showing detailed breakdown for both directions
        * ``'precision_load_sharing'`` : Figure showing precision load sharing details
        * ``'recall_load_sharing'`` : Figure showing recall load sharing details  
        * ``'summary_table'`` : Figure with load sharing summary statistics
    
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
        figs = visualize_las_load_sharing(result['internals'])
        figs['load_sharing_details'].show()
        figs['summary_table'].show()
    
    See Also
    --------
    visualize_las : See basic LAS visualization
    visualize_best_match : See matching analysis
    """
    las_internals = internals['metrics']['las']
    
    # Create comprehensive load sharing details figure
    load_sharing_details_fig = create_load_sharing_details_figure(las_internals)
    
    # Create individual direction figures
    precision_fig = _create_precision_load_sharing_figure(las_internals)
    recall_fig = _create_recall_load_sharing_figure(las_internals)
    
    # Create summary statistics table
    summary_fig = create_load_sharing_summary_table_figure(las_internals)
    
    return {
        'load_sharing_details': load_sharing_details_fig,
        'precision_load_sharing': precision_fig,
        'recall_load_sharing': recall_fig,
        'summary_table': summary_fig
    }

def _create_precision_load_sharing_figure(las_internals: Dict[str, Any]) -> plt.Figure:
    """Create a dedicated figure for precision load sharing analysis."""
    precision_internals = las_internals.get('precision_internals', {})
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle('LAS Precision Load Sharing Analysis', fontsize=16, fontweight='bold')
    
    create_precision_load_sharing_section(ax, precision_internals)
    
    plt.tight_layout()
    return fig

def _create_recall_load_sharing_figure(las_internals: Dict[str, Any]) -> plt.Figure:
    """Create a dedicated figure for recall load sharing analysis."""
    recall_internals = las_internals.get('recall_internals', {})
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle('LAS Recall Load Sharing Analysis', fontsize=16, fontweight='bold')
    
    create_recall_load_sharing_section(ax, recall_internals)
    
    plt.tight_layout()
    return fig