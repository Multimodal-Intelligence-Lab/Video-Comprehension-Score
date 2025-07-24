import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Any

def draw_mapping_windows_with_lct(ax: plt.Axes, windows: List[Tuple[int, int]], 
                                 indices: List[int], lct: int, window_height: float,
                                 max_len: int, window_type: str = 'precision') -> None:
    """Draw mapping windows with LCT padding zones."""
    color_map = {'precision': 'blue', 'recall': 'red'}
    edge_color = color_map.get(window_type, 'blue')
    
    for idx, r_idx in enumerate(indices):
        if r_idx >= 0 and idx < len(windows):
            start, end = windows[idx]
            
            # Main mapping window
            rect = Rectangle((idx - 0.4, start), 0.8, end - start, 
                           fill=False, edgecolor=edge_color, alpha=0.3)
            ax.add_patch(rect)
            
            # LCT padding zone (if lct > 0)
            if lct > 0:
                lct_padding = lct * window_height
                expanded_start = max(0, start - lct_padding)
                expanded_end = min(max_len, end + lct_padding)
                
                # Only show if there's actual padding added
                if expanded_start < start or expanded_end > end:
                    lct_rect = Rectangle((idx - 0.4, expanded_start), 0.8, expanded_end - expanded_start,
                                       fill=False, edgecolor='green', linestyle='--', alpha=0.3)
                    ax.add_patch(lct_rect)

def determine_point_color(idx: int, target_idx: int, start: int, end: int, 
                         lct: int, nas_data: Dict[str, Any]) -> str:
    """Determine the color for a point based on window and LCT zone."""
    in_original_window = start <= target_idx < end
    in_lct_window = False
    
    if lct > 0 and 'in_lct_zone' in nas_data:
        in_lct_window = nas_data['in_lct_zone'][idx]
    
    if in_original_window:
        return 'green'
    elif in_lct_window:
        return 'orange'  # Different color for points in LCT zone
    else:
        return 'red'

def should_draw_distance_line(idx: int, target_idx: int, start: int, end: int, 
                             lct: int, nas_data: Dict[str, Any]) -> bool:
    """Determine if a distance line should be drawn."""
    in_original_window = start <= target_idx < end
    in_lct_window = False
    
    if lct > 0 and 'in_lct_zone' in nas_data:
        in_lct_window = nas_data['in_lct_zone'][idx]
    
    return not in_original_window and not in_lct_window

def draw_distance_line(ax: plt.Axes, idx: int, target_idx: int, start: int, end: int,
                      window_type: str = 'precision') -> None:
    """Draw distance lines for points outside both zones."""
    if window_type == 'precision':
        if target_idx < start:
            ax.plot([idx, idx], [target_idx, start], 'r-', alpha=0.5)
        else:
            ax.plot([idx, idx], [target_idx, end-1], 'r-', alpha=0.5)
    else:  # recall
        if target_idx < start:
            ax.plot([idx, idx], [target_idx, start], 'r-', alpha=0.5)
        else:
            ax.plot([idx, idx], [target_idx, end-1], 'r-', alpha=0.5)

def create_lct_legend_elements() -> List[Line2D]:
    """Create legend elements for LCT visualization."""
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='In Window'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='In LCT Zone'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Outside')
    ]