import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from typing import Dict, Any

def visualize_mapping_windows(internals: Dict[str, Any]) -> plt.Figure:
    """Visualize the mapping windows used for constraining alignment between texts.
    
    Shows the allowable alignment regions (mapping windows) that constrain which
    reference chunks can be matched to which generated chunks. This helps maintain
    chronological ordering while allowing some flexibility. Also displays Local
    Chronology Tolerance (LCT) zones when applicable.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'mapping_windows', 'alignment',
        'config', and 'metrics' sections.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure with two subplots showing precision and recall mapping windows,
        including LCT padding zones and actual matches found.
    
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
            lct=1
        )
        fig = visualize_mapping_windows(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_distance_nas : See how mapping windows affect NAS-D penalties
    visualize_similarity_matrix : Compare with unconstrained similarity
    """
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    precision_windows = internals['mapping_windows']['precision']
    recall_windows = internals['mapping_windows']['recall']
    
    # Get LCT value from config
    lct = internals['config']['lct']
    
    # Get the LCT window heights from the NAS calculations
    prec_window_height = internals['metrics']['nas']['nas_d']['precision']['mapping_window_height']
    rec_window_height = internals['metrics']['nas']['nas_d']['recall']['mapping_window_height'] 
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    is_ref_longer = ref_len >= gen_len
    
    ax_precision = axes[0]
    for g_idx, (r_start, r_end) in enumerate(precision_windows):
        if g_idx < gen_len:
            height = r_end - r_start
            # Draw the original mapping window
            rect = Rectangle((g_idx, r_start), 1, height, 
                            edgecolor='blue', facecolor='lightblue', alpha=0.5)
            ax_precision.add_patch(rect)
            
            # Draw LCT padding if lct > 0
            if lct > 0:
                # Use the centralized LCT window height from NAS calculation
                lct_padding = lct * prec_window_height
                
                # Create expanded window with LCT padding
                expanded_start = max(0, r_start - lct_padding)
                expanded_end = min(ref_len, r_end + lct_padding)
                
                # Draw the LCT padding zone (if it extends beyond the original window)
                if expanded_start < r_start:
                    top_padding = Rectangle((g_idx, expanded_start), 1, r_start - expanded_start,
                                        edgecolor='green', facecolor='lightgreen', alpha=0.3, linestyle='--')
                    ax_precision.add_patch(top_padding)
                
                if expanded_end > r_end:
                    bottom_padding = Rectangle((g_idx, r_end), 1, expanded_end - r_end,
                                        edgecolor='green', facecolor='lightgreen', alpha=0.3, linestyle='--')
                    ax_precision.add_patch(bottom_padding)
            
            # Add text label for window
            ax_precision.text(g_idx + 0.5, r_start + height/2, f"P({g_idx})",
                            ha='center', va='center', fontsize=8)
    
    ax_precision.set_xlim(-0.5, gen_len + 0.5)
    ax_precision.set_ylim(-0.5, ref_len + 0.5)
    ax_precision.set_xlabel('Generation Index')
    ax_precision.set_ylabel('Reference Index')
    title = f'Precision Mapping Windows (ref→gen)'
    if lct > 0:
        title += f' (LCT={lct}, LCT Window={prec_window_height})'
    ax_precision.set_title(title)
    ax_precision.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend for LCT padding if applicable
    if lct > 0:
        lct_patch = Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', alpha=0.3, linestyle='--')
        ax_precision.legend([lct_patch], [f'LCT Padding (LCT={lct})'])
    
    precision_matches = internals['alignment']['precision']['matches']
    for g_idx, r_idx in precision_matches:
        if 0 <= g_idx < gen_len and 0 <= r_idx < ref_len:
            ax_precision.plot(g_idx + 0.5, r_idx + 0.5, 'ro', ms=6)
    
    ax_recall = axes[1]
    for r_idx, (g_start, g_end) in enumerate(recall_windows):
        if r_idx < ref_len:
            height = g_end - g_start
            # Draw the original mapping window
            rect = Rectangle((r_idx, g_start), 1, height, 
                            edgecolor='red', facecolor='mistyrose', alpha=0.5)
            ax_recall.add_patch(rect)
            
            # Draw LCT padding if lct > 0
            if lct > 0:
                # Use the centralized LCT window height from NAS calculation
                lct_padding = lct * rec_window_height
                
                # Create expanded window with LCT padding
                expanded_start = max(0, g_start - lct_padding)
                expanded_end = min(gen_len, g_end + lct_padding)
                
                # Draw the LCT padding zone (if it extends beyond the original window)
                if expanded_start < g_start:
                    top_padding = Rectangle((r_idx, expanded_start), 1, g_start - expanded_start,
                                        edgecolor='green', facecolor='lightgreen', alpha=0.3, linestyle='--')
                    ax_recall.add_patch(top_padding)
                
                if expanded_end > g_end:
                    bottom_padding = Rectangle((r_idx, g_end), 1, expanded_end - g_end,
                                        edgecolor='green', facecolor='lightgreen', alpha=0.3, linestyle='--')
                    ax_recall.add_patch(bottom_padding)
            
            # Add text label for window
            ax_recall.text(r_idx + 0.5, g_start + height/2, f"R({r_idx})",
                          ha='center', va='center', fontsize=8)
    
    ax_recall.set_xlim(-0.5, ref_len + 0.5)
    ax_recall.set_ylim(-0.5, gen_len + 0.5)
    ax_recall.set_xlabel('Reference Index')
    ax_recall.set_ylabel('Generation Index')
    title = f'Recall Mapping Windows (gen→ref)'
    if lct > 0:
        title += f' (LCT={lct}, LCT Window={rec_window_height})'
    ax_recall.set_title(title)
    ax_recall.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend for LCT padding if applicable
    if lct > 0:
        lct_patch = Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', alpha=0.3, linestyle='--')
        ax_recall.legend([lct_patch], [f'LCT Padding (LCT={lct})'])
    
    recall_matches = internals['alignment']['recall']['matches']
    for g_idx, r_idx in recall_matches:
        if 0 <= g_idx < gen_len and 0 <= r_idx < ref_len:
            ax_recall.plot(r_idx + 0.5, g_idx + 0.5, 'bo', ms=6)
    
    fig.suptitle(f'Mapping Windows (ref_len={ref_len}, gen_len={gen_len})', fontsize=16)
    fig.tight_layout()
    return fig