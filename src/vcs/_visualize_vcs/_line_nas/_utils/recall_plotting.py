import matplotlib.pyplot as plt
from typing import Dict, Any
from .window_drawing import draw_mapping_windows, setup_plot_limits_and_labels
from .path_plotting import plot_actual_path, plot_floor_path, plot_ceiling_path, add_metrics_text_box

def setup_recall_plot(ax: plt.Axes, ref_len: int, gen_len: int) -> None:
    """Set up the recall plot with proper limits and labels."""
    setup_plot_limits_and_labels(
        ax, ref_len, gen_len,
        'Reference Index', 'Generation Index', 'Recall Line NAS'
    )

def draw_recall_content(ax: plt.Axes, internals: Dict[str, Any], 
                       ref_len: int, gen_len: int) -> None:
    """Draw all content for the recall plot."""
    # Draw mapping windows
    recall_windows = internals['mapping_windows']['recall']
    draw_mapping_windows(ax, recall_windows, ref_len, 'recall')
    
    # Get alignment data
    aligned_recall = internals['alignment']['recall']['aligned_segments']
    
    if aligned_recall:
        # Plot actual line (with swapped coordinates for recall)
        plot_actual_path(ax, aligned_recall, color='r', label='Actual Line', swap_coords=True)
        
        # Get line data
        recall_line_data = internals['metrics']['nas']['nas_l']['recall']
        
        # Plot floor path if it exists
        floor_path = recall_line_data.get('floor_path', [])
        if floor_path:
            plot_floor_path(ax, floor_path, swap_coords=True)
        
        # Plot ceiling path if it exists
        ceil_path = recall_line_data.get('ceil_path', [])
        if ceil_path:
            plot_ceiling_path(ax, ceil_path, swap_coords=True)
        
        # Add metrics text box
        add_metrics_text_box(ax, recall_line_data, 'Recall')
        
        # Add legend after all labeled content is drawn
        ax.legend(loc='best')