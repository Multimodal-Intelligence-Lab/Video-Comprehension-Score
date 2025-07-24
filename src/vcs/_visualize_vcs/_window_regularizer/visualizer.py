import matplotlib.pyplot as plt
from typing import Dict, Any

def visualize_window_regularizer(internals: Dict[str, Any]) -> plt.Figure:
    """Visualize the Window Regularizer calculation and its components.
    
    Shows how the window regularizer adjusts the final NAS score based on the
    overlap between mapping windows and the total timeline area. The regularizer
    prevents inflated scores when mapping windows cover too much of the alignment
    space.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'metrics' section with regularizer
        calculations.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the regularizer components as a bar chart with calculation
        formula and final regularizer value prominently displayed.
    
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
        fig = visualize_window_regularizer(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_mapping_windows : See the windows being regularized
    visualize_metrics_summary : See impact on final NAS score
    """
    regularizer_data = internals['metrics']['nas']['regularizer']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total_window_area = regularizer_data['total_mapping_window_area']
    timeline_area = regularizer_data['timeline_area']
    min_area = regularizer_data['min_area']
    regularizer_value = regularizer_data['value']
    
    labels = ['Total Window Area', 'Timeline Area', 'Window/Timeline Ratio']
    values = [total_window_area, timeline_area, total_window_area / timeline_area]
    
    bars = ax.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
               f'{value:.4f}', ha='center', va='bottom')
    
    ax.text(0.5, 0.9, f"Window Regularizer = {regularizer_value:.4f}", 
           transform=ax.transAxes, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
    
    formula = (f"Calculation: (({total_window_area:.2f} / {timeline_area:.2f}) - {min_area:.4f}) / "
              f"(0.5 - {min_area:.4f}) = {regularizer_value:.4f}")
    ax.text(0.5, 0.8, formula, transform=ax.transAxes, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
    
    ax.set_ylabel('Area Value')
    ax.set_title('Window Regularizer Components')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    return fig