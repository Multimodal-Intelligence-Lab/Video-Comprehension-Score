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

def setup_precision_mapping_plot(ax: plt.Axes, gen_len: int, ref_len: int, 
                                lct: int, window_height: float) -> None:
    """Set up precision mapping plot with basic styling."""
    ax.set_xlim(-0.5, gen_len + 0.5)
    ax.set_ylim(-0.5, ref_len + 0.5)
    ax.set_xlabel('Generation Index')
    ax.set_ylabel('Reference Index')
    
    title = 'Precision Mapping with Distances'
    if lct > 0:
        title += f' (LCT={lct}, LCT Window={window_height})'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

def draw_precision_mapping_content(ax: plt.Axes, internals: Dict[str, Any],
                                  ref_len: int, gen_len: int, lct: int) -> None:
    """Draw all content for precision mapping plot."""
    precision_indices = internals['alignment']['precision']['indices']
    precision_windows = internals['mapping_windows']['precision']
    prec_nas_data = internals['metrics']['nas']['nas_d']['precision']
    prec_window_height = prec_nas_data['mapping_window_height']
    
    # Draw mapping windows with LCT
    draw_mapping_windows_with_lct(
        ax, precision_windows, precision_indices, lct, 
        prec_window_height, ref_len, 'precision'
    )
    
    # Draw points and distance lines
    for g_idx, r_idx in enumerate(precision_indices):
        if r_idx >= 0 and g_idx < len(precision_windows):
            start, end = precision_windows[g_idx]
            
            # Determine point color
            color = determine_point_color(g_idx, r_idx, start, end, lct, prec_nas_data)
            ax.plot([g_idx], [r_idx], 'o', color=color, ms=6)
            
            # Draw distance lines if needed
            if should_draw_distance_line(g_idx, r_idx, start, end, lct, prec_nas_data):
                draw_distance_line(ax, g_idx, r_idx, start, end, 'precision')
    
    # Add legend if LCT is used
    if lct > 0:
        legend_elements = create_lct_legend_elements()
        ax.legend(handles=legend_elements, loc='best')

def draw_precision_penalty_plot(ax: plt.Axes, internals: Dict[str, Any], lct: int) -> None:
    """Draw precision penalty visualization."""
    prec_nas_data = internals['metrics']['nas']['nas_d']['precision']
    prec_window_height = prec_nas_data['mapping_window_height']
    
    # Get penalty data
    penalties = np.array(prec_nas_data['penalties'])
    in_window = np.array(prec_nas_data['in_window'])
    in_lct_zone = prec_nas_data.get('in_lct_zone', [False] * len(penalties))
    
    # Set up plot
    title = create_penalty_title('Precision NAS-D Penalties', prec_nas_data["total_penalty"])
    setup_penalty_plot(ax, title, 'Generation Index')
    
    # Draw bars
    draw_penalty_bars(ax, penalties, in_window, in_lct_zone, 'skyblue')
    
    # Add annotations
    add_penalty_annotations(ax, penalties, in_window)
    
    # Add legend and metrics text
    add_penalty_legend(ax, lct)
    add_penalty_metrics_text(ax, prec_nas_data, 'Precision', lct, prec_window_height)