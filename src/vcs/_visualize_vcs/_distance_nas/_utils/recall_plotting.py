import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from .lct_handling import (
    draw_mapping_windows_with_lct, determine_point_color, 
    should_draw_distance_line, draw_distance_line, create_lct_legend_elements
)
from .penalty_visualization import (
    setup_penalty_plot, draw_penalty_bars, add_penalty_annotations,
    add_penalty_metrics_text, add_penalty_legend, create_penalty_title
)

def setup_recall_mapping_plot(ax: plt.Axes, ref_len: int, gen_len: int, 
                             lct: int, window_height: float) -> None:
    """Set up recall mapping plot with basic styling."""
    ax.set_xlim(-0.5, ref_len + 0.5)
    ax.set_ylim(-0.5, gen_len + 0.5)
    ax.set_xlabel('Reference Index')
    ax.set_ylabel('Generation Index')
    
    title = 'Recall Mapping with Distances'
    if lct > 0:
        title += f' (LCT={lct}, LCT Window={window_height})'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

def draw_recall_mapping_content(ax: plt.Axes, internals: Dict[str, Any],
                               ref_len: int, gen_len: int, lct: int) -> None:
    """Draw all content for recall mapping plot."""
    recall_indices = internals['alignment']['recall']['indices']
    recall_windows = internals['mapping_windows']['recall']
    rec_nas_data = internals['metrics']['nas']['nas_d']['recall']
    rec_window_height = rec_nas_data['mapping_window_height']
    
    # Draw mapping windows with LCT
    draw_mapping_windows_with_lct(
        ax, recall_windows, recall_indices, lct, 
        rec_window_height, gen_len, 'recall'
    )
    
    # Draw points and distance lines
    for r_idx, g_idx in enumerate(recall_indices):
        if g_idx >= 0 and r_idx < len(recall_windows):
            start, end = recall_windows[r_idx]
            
            # Determine point color
            color = determine_point_color(r_idx, g_idx, start, end, lct, rec_nas_data)
            ax.plot([r_idx], [g_idx], 'o', color=color, ms=6)
            
            # Draw distance lines if needed
            if should_draw_distance_line(r_idx, g_idx, start, end, lct, rec_nas_data):
                draw_distance_line(ax, r_idx, g_idx, start, end, 'recall')
    
    # Add legend if LCT is used
    if lct > 0:
        legend_elements = create_lct_legend_elements()
        ax.legend(handles=legend_elements, loc='best')

def draw_recall_penalty_plot(ax: plt.Axes, internals: Dict[str, Any], lct: int) -> None:
    """Draw recall penalty visualization."""
    rec_nas_data = internals['metrics']['nas']['nas_d']['recall']
    rec_window_height = rec_nas_data['mapping_window_height']
    
    # Get penalty data
    penalties = np.array(rec_nas_data['penalties'])
    in_window = np.array(rec_nas_data['in_window'])
    in_lct_zone = rec_nas_data.get('in_lct_zone', [False] * len(penalties))
    
    # Set up plot
    title = create_penalty_title('Recall NAS-D Penalties', rec_nas_data["total_penalty"])
    setup_penalty_plot(ax, title, 'Reference Index')
    
    # Draw bars
    draw_penalty_bars(ax, penalties, in_window, in_lct_zone, 'lightsalmon')
    
    # Add annotations
    add_penalty_annotations(ax, penalties, in_window)
    
    # Add legend and metrics text
    add_penalty_legend(ax, lct)
    add_penalty_metrics_text(ax, rec_nas_data, 'Recall', lct, rec_window_height)