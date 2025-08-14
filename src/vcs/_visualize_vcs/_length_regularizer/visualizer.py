import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def visualize_length_regularizer(internals: Dict[str, Any]) -> plt.Figure:
    """Visualize the Length Regularizer calculation and its penalty components.
    
    Shows how the length regularizer adjusts the final NAS score based on two
    penalty types: window penalty (wp) for coverage patterns and imbalance penalty 
    (i_p) for length differences. These are blended using a noisy-OR approach to
    create the final regularization factor.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'metrics' section with regularizer
        calculations and 'config' section with blend factor and coverage cutoff.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the penalty components, blending visualization, and final 
        regularizer value with calculation details.
    
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
        fig = visualize_length_regularizer(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_mapping_windows : See the windows being analyzed for coverage
    visualize_metrics_summary : See impact on final NAS score
    """
    regularizer_data = internals['metrics']['nas']['regularizer']
    config = internals.get('config', {})
    
    # Extract penalty values (need to calculate wp and i_p from the regularizer data)
    total_window_area = regularizer_data['total_mapping_window_area']
    timeline_area = regularizer_data['timeline_area']
    min_area = regularizer_data['min_area']
    regularizer_value = regularizer_data['value']
    
    # Get text lengths from internals to calculate i_p
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    
    # Calculate coverage ratio and window penalty (wp)
    coverage_ratio = (total_window_area / timeline_area) if timeline_area > 0 else 0.0
    nas_coverage_cutoff = config.get('nas_coverage_cutoff', 0.5)
    
    if timeline_area > 0 and min_area < nas_coverage_cutoff and (nas_coverage_cutoff - min_area) > 1e-9:
        wp = max(0, min(1, (coverage_ratio - min_area) / (nas_coverage_cutoff - min_area)))
    elif timeline_area > 0 and min_area >= nas_coverage_cutoff:
        wp = 0.0 if coverage_ratio <= (min_area + 1e-9) else 1.0
    else:
        wp = 0.0
    
    # Calculate imbalance penalty (i_p)
    i_p = 0.0 if max(ref_len, gen_len) == 0 else (1.0 - min(ref_len, gen_len) / max(ref_len, gen_len))
    i_p = max(0, min(1, i_p))
    
    nas_blend_factor = config.get('nas_blend_factor', 0.0)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Penalty Components Bar Chart
    penalties = ['Window Penalty (wp)', 'Imbalance Penalty (i_p)', 'Final Regularizer']
    penalty_values = [wp, i_p, regularizer_value]
    colors = ['lightblue', 'lightcoral', 'gold']
    
    bars1 = ax1.bar(penalties, penalty_values, color=colors)
    ax1.set_ylabel('Penalty Value')
    ax1.set_title('Length Regularizer Penalty Components')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars1, penalty_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Coverage Analysis
    coverage_components = ['Total Window Area', 'Timeline Area', 'Coverage Ratio', 'Min Area', 'Coverage Cutoff']
    coverage_values = [total_window_area, timeline_area, coverage_ratio, min_area, nas_coverage_cutoff]
    
    bars2 = ax2.bar(coverage_components, coverage_values, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'])
    ax2.set_ylabel('Area Value')
    ax2.set_title('Window Coverage Analysis')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars2, coverage_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(coverage_values),
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Length Imbalance Analysis
    length_components = ['Reference Length', 'Generated Length', 'Length Ratio']
    length_ratio = min(ref_len, gen_len) / max(ref_len, gen_len) if max(ref_len, gen_len) > 0 else 1.0
    length_values = [ref_len, gen_len, length_ratio]
    
    bars3 = ax3.bar(length_components, length_values, color=['lightsteelblue', 'lightpink', 'lightgray'])
    ax3.set_ylabel('Length/Ratio Value')
    ax3.set_title('Length Imbalance Analysis')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars3, length_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(length_values),
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Blending Visualization
    x = np.linspace(-1, 1, 100)
    # Simplified visualization of blend factor effect
    wp_influence = (1 + x) / 2  # More positive = more wp influence
    ip_influence = (1 - x) / 2  # More negative = more ip influence
    
    ax4.plot(x, wp_influence, label='Window Penalty Influence', color='blue', linewidth=2)
    ax4.plot(x, ip_influence, label='Imbalance Penalty Influence', color='red', linewidth=2)
    ax4.axvline(nas_blend_factor, color='green', linestyle='--', linewidth=2, 
                label=f'Current Blend Factor: {nas_blend_factor:.2f}')
    ax4.set_xlabel('Blend Factor')
    ax4.set_ylabel('Penalty Influence')
    ax4.set_title('Penalty Blending Visualization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(0, 1)
    
    # Add calculation summary
    fig.suptitle(f'Length Regularizer = {regularizer_value:.4f}\n'
                f'Blend Factor: {nas_blend_factor:.2f}, Coverage Cutoff: {nas_coverage_cutoff:.2f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig