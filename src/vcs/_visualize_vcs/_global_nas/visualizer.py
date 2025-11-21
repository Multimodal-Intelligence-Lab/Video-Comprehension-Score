import matplotlib.pyplot as plt
from typing import Dict, Any
from ._utils import (
    setup_precision_mapping_plot, draw_precision_mapping_content, draw_precision_penalty_plot,
    setup_recall_mapping_plot, draw_recall_mapping_content, draw_recall_penalty_plot
)

def visualize_global_nas(internals: Dict[str, Any]) -> plt.Figure:
    """Create visualization of Global Narrative Alignment Score components.
    
    Shows precision and recall mapping windows with penalty calculations that measure
    how well the chronological ordering of segments is preserved between texts.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain Global NAS metrics and mapping data.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure with four subplots showing precision mapping, recall mapping,
        and their respective penalty calculations for Global NAS.
    
    Examples
    --------
    **Basic Usage:**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_global_sas=your_embedder,
            return_internals=True,
            return_all_metrics=True
        )
        fig = visualize_global_nas(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_local_nas : See local NAS visualization
    visualize_mapping_windows : See mapping window details
    """
    # Extract basic data
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    Rn = internals['config']['Rn']

    # Get NAS data for window heights
    prec_nas_data = internals['metrics']['global_nas']['precision']
    rec_nas_data = internals['metrics']['global_nas']['recall']
    prec_window_height = prec_nas_data['alignment_window_height']
    rec_window_height = rec_nas_data['alignment_window_height']
    
    # Create figure with 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PRECISION MAPPING PLOT (top-left)
    ax_prec_mapping = axes[0, 0]
    setup_precision_mapping_plot(ax_prec_mapping, gen_len, ref_len, Rn, prec_window_height)
    draw_precision_mapping_content(ax_prec_mapping, internals, ref_len, gen_len, Rn)
    
    # RECALL MAPPING PLOT (top-right)
    ax_rec_mapping = axes[0, 1]
    setup_recall_mapping_plot(ax_rec_mapping, ref_len, gen_len, Rn, rec_window_height)
    draw_recall_mapping_content(ax_rec_mapping, internals, ref_len, gen_len, Rn)
    
    # PRECISION PENALTY PLOT (bottom-left)
    ax_prec_penalties = axes[1, 0]
    draw_precision_penalty_plot(ax_prec_penalties, internals, Rn)
    
    # RECALL PENALTY PLOT (bottom-right)
    ax_rec_penalties = axes[1, 1]
    draw_recall_penalty_plot(ax_rec_penalties, internals, Rn)
    
    # Set overall title and layout
    fig.suptitle('Global NAS Metrics', fontsize=16)
    fig.tight_layout()
    
    return fig