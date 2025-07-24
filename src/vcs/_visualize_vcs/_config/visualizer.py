import matplotlib.pyplot as plt
from typing import Dict, Any

def visualize_config(internals: Dict[str, Any]) -> plt.Figure:
    """Create a visualization of VCS configuration parameters and text lengths.
    
    Displays all configuration parameters used for VCS computation in a clear
    text box format. Useful for understanding what settings were used to
    produce specific results and for documentation purposes.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'config' and 'texts' sections.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure showing configuration parameters in a formatted text box
        including chunk size, context parameters, LCT, and text lengths.
    
    Examples
    --------
    **Basic Usage:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True
        )
        fig = visualize_config(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_metrics_summary : See the results produced by this configuration
    compute_vcs_score : The main function that uses these parameters
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    config = internals['config']
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    
    config_text = "VCS Configuration Parameters\n\n"
    config_text += f"Chunk Size: {config['chunk_size']}\n"
    config_text += f"Context Cutoff Value: {config['context_cutoff_value']:.4f}\n"
    config_text += f"Context Window Control: {config['context_window_control']:.4f}\n"
    config_text += f"Local Chronology Tolerance (LCT): {config['lct']}\n\n"
    config_text += f"Reference Length: {ref_len} | Generated Length: {gen_len}"
    
    ax.text(0.5, 0.5, config_text,
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.3),
            transform=ax.transAxes)
    
    plt.tight_layout()
    return fig