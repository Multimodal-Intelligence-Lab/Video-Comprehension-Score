import matplotlib.pyplot as plt
from typing import Dict, Any
from .window_drawing import draw_mapping_windows, setup_plot_limits_and_labels
from .path_plotting import plot_actual_path, plot_floor_path, plot_ceiling_path, add_metrics_text_box

def setup_precision_plot(ax: plt.Axes, ref_len: int, gen_len: int) -> None:
    """Set up the precision plot with proper limits and labels."""
    setup_plot_limits_and_labels(
        ax, gen_len, ref_len,
        'Generation Index', 'Reference Index', 'Precision Line NAS'
    )

def draw_precision_content(ax: plt.Axes, internals: Dict[str, Any], 
                          ref_len: int, gen_len: int) -> None:
    """Draw all content for the precision plot."""
    # Draw mapping windows
    precision_windows = internals['mapping_windows']['precision']
    draw_mapping_windows(ax, precision_windows, gen_len, 'precision')
    
    # Get alignment data
    aligned_precision = internals['alignment']['precision']['aligned_segments']
    
    if aligned_precision:
        # Plot actual line
        plot_actual_path(ax, aligned_precision, color='b', label='Actual Line')
        
        # Get line data
        precision_line_data = internals['metrics']['nas']['nas_l']['precision']
        
        # Plot floor path if it exists
        floor_path = precision_line_data.get('floor_path', [])
        if floor_path:
            plot_floor_path(ax, floor_path)
        
        # Plot ceiling path if it exists
        ceil_path = precision_line_data.get('ceil_path', [])
        if ceil_path:
            plot_ceiling_path(ax, ceil_path)
        
        # Add metrics text box
        add_metrics_text_box(ax, precision_line_data, 'Precision')
        
        # Add legend after all labeled content is drawn
        ax.legend(loc='best')