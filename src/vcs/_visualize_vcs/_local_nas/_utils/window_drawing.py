import matplotlib.pyplot as plt
from typing import List, Tuple

def draw_mapping_windows(ax: plt.Axes, windows: List[Tuple[int, int]], 
                        total_length: int, window_type: str = 'precision') -> None:

    for idx, (start, end) in enumerate(windows):
        if window_type == 'precision':
            # Precision: g_idx -> r_idx mapping
            if idx < total_length:
                rect = plt.Rectangle((idx + 0.5, start), 1, end - start, 
                                   fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
                ax.add_patch(rect)
        else:
            # Recall: r_idx -> g_idx mapping
            if idx < total_length:
                rect = plt.Rectangle((idx + 0.5, start), 1, end - start, 
                                   fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
                ax.add_patch(rect)

def setup_plot_limits_and_labels(ax: plt.Axes, x_max: int, y_max: int, 
                                 x_label: str, y_label: str, title: str) -> None:
    """Set up plot limits, labels, and styling."""
    ax.set_xlim(0, x_max + 1)
    ax.set_ylim(0, y_max + 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)