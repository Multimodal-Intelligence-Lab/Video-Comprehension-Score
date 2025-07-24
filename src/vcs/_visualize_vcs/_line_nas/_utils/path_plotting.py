import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Dict

def plot_actual_path(ax: plt.Axes, aligned_segments: List[Tuple], color: str = 'b', 
                    label: str = 'Actual Line', swap_coords: bool = False) -> None:
    """Plot the actual alignment path."""
    if not aligned_segments:
        return
        
    if swap_coords:
        # For recall plots: swap coordinates (r_idx, g_idx)
        x_indices = [point[1] for point in aligned_segments]  # r_idx
        y_indices = [point[0] for point in aligned_segments]  # g_idx
    else:
        # For precision plots: normal coordinates (g_idx, r_idx)
        x_indices = [point[0] for point in aligned_segments]  # g_idx
        y_indices = [point[1] for point in aligned_segments]  # r_idx
    
    ax.plot(x_indices, y_indices, f'{color}-o', label=label)

def plot_floor_path(ax: plt.Axes, floor_path: List[Tuple], 
                   label: str = 'Floor Ideal (Shortest Path)', swap_coords: bool = False) -> None:
    """Plot the floor (shortest) ideal path."""
    if not floor_path:
        return
        
    if swap_coords:
        # For recall plots
        x_indices = [point[0] for point in floor_path]  # r_idx
        y_indices = [point[1] for point in floor_path]  # g_idx
    else:
        # For precision plots
        x_indices = [point[0] for point in floor_path]  # g_idx
        y_indices = [point[1] for point in floor_path]  # r_idx
    
    ax.plot(x_indices, y_indices, 'g--^', label=label)

def plot_ceiling_path(ax: plt.Axes, ceil_path: List[Tuple], 
                     label: str = 'Ceiling Ideal (Longest Path)', swap_coords: bool = False) -> None:
    """Plot the ceiling (longest) ideal path."""
    if not ceil_path:
        return
        
    if swap_coords:
        # For recall plots
        x_indices = [point[0] for point in ceil_path]  # r_idx
        y_indices = [point[1] for point in ceil_path]  # g_idx
    else:
        # For precision plots
        x_indices = [point[0] for point in ceil_path]  # g_idx
        y_indices = [point[1] for point in ceil_path]  # r_idx
    
    ax.plot(x_indices, y_indices, 'r--s', label=label)

def add_metrics_text_box(ax: plt.Axes, line_data: Dict[str, Any], 
                        metric_name: str, position: Tuple[float, float] = (0.05, 0.95)) -> None:
    """Add metrics information text box to the plot."""
    actual_length = line_data['actual_line_length']
    floor_line = line_data['floor_ideal_line_length']
    ceil_line = line_data['ceil_ideal_line_length']
    nas_value = line_data['value']
    
    text_content = (f"Actual Length: {actual_length:.2f}\n"
                   f"Floor Ideal (Min): {floor_line:.2f}\n"
                   f"Ceil Ideal (Max): {ceil_line:.2f}\n"
                   f"NAS-L {metric_name}: {nas_value:.4f}")
    
    ax.text(position[0], position[1], text_content,
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))