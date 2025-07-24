import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List

def setup_penalty_plot(ax: plt.Axes, title: str, x_label: str) -> None:
    """Set up basic penalty plot styling."""
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Penalty')
    ax.set_title(title)

def draw_penalty_bars(ax: plt.Axes, penalties: np.ndarray, in_window: np.ndarray, 
                     in_lct_zone: List[bool], base_color: str = 'skyblue') -> List:
    """Draw penalty bars with appropriate colors based on window status."""
    x_indices = np.arange(len(penalties))
    bars = ax.bar(x_indices, penalties, alpha=0.7, color=base_color, label='Penalty')
    
    # Color bars based on window status
    for i, (penalty, is_in, is_lct) in enumerate(zip(penalties, in_window, in_lct_zone)):
        if is_in:
            bars[i].set_color('green')
        elif is_lct:
            bars[i].set_color('orange')
        elif penalty > 0:
            bars[i].set_color('red')
    
    return bars

def add_penalty_annotations(ax: plt.Axes, penalties: np.ndarray, in_window: np.ndarray) -> None:
    """Add value annotations to penalty bars."""
    for i, (penalty, is_in) in enumerate(zip(penalties, in_window)):
        if penalty > 0:
            ax.annotate(f"{penalty:.3f}", 
                       xy=(i, penalty), 
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

def add_penalty_legend(ax: plt.Axes, lct: int) -> None:
    """Add legend for penalty plot if LCT is being used."""
    if lct > 0:
        ax.legend(['Zero Line', 'Normal Penalty', 'In Window (No Penalty)', 
                  'In LCT Zone (No Penalty)'])

def add_penalty_metrics_text(ax: plt.Axes, nas_data: Dict[str, Any], 
                           metric_name: str, lct: int, window_height: float) -> None:
    """Add metrics text box to penalty plot."""
    text_content = (f"Max Penalty: {nas_data['max_penalty']:.4f}\n"
                   f"NAS-D {metric_name}: {nas_data['value']:.4f}\n")
    
    if lct > 0:
        text_content += f"LCT: {lct}, LCT Window: {window_height}"
    
    ax.text(0.05, 0.95, text_content,
           transform=ax.transAxes, 
           va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_penalty_title(base_title: str, total_penalty: float) -> str:
    """Create penalty plot title with total penalty information."""
    return f"{base_title} (Total: {total_penalty:.4f})"