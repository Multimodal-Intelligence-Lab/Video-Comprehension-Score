import matplotlib.pyplot as plt
from typing import Dict, Any
from ._utils import setup_precision_plot, draw_precision_content, setup_recall_plot, draw_recall_content

def visualize_line_nas(internals: Dict[str, Any]) -> plt.Figure:
    """Visualize Line-based Narrative Alignment Score (NAS-L) calculations.
    
    Shows the actual alignment paths compared to ideal narrative lines for both
    precision and recall directions. Displays floor (shortest) and ceiling (longest)
    ideal paths along with the actual path taken through the alignment space.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'alignment' and 'metrics' sections
        with line-based NAS calculations.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure with two subplots showing precision and recall line-based analysis,
        including actual paths, ideal boundaries, and calculated path lengths.
    
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
        fig = visualize_line_nas(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_distance_nas : Compare with distance-based narrative analysis
    visualize_line_nas_precision_calculations : Detailed precision calculations
    visualize_line_nas_recall_calculations : Detailed recall calculations
    """
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # PRECISION PLOT
    ax_precision = axes[0]
    setup_precision_plot(ax_precision, ref_len, gen_len)
    draw_precision_content(ax_precision, internals, ref_len, gen_len)
    
    # RECALL PLOT
    ax_recall = axes[1]
    setup_recall_plot(ax_recall, ref_len, gen_len)
    draw_recall_content(ax_recall, internals, ref_len, gen_len)
    
    # Set overall title and layout
    fig.suptitle('Line-based NAS Metrics', fontsize=16)
    fig.tight_layout()
    
    return fig