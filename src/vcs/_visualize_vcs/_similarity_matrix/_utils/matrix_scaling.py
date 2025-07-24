import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, NamedTuple

class MatrixSize(NamedTuple):
    """Container for matrix size classification."""
    is_large: bool
    is_very_large: bool

def determine_matrix_size(ref_len: int, gen_len: int) -> MatrixSize:
    """Determine if this is a large matrix based on dimensions."""
    is_large = ref_len > 50 or gen_len > 50
    is_very_large = ref_len > 100 or gen_len > 100
    return MatrixSize(is_large=is_large, is_very_large=is_very_large)

def calculate_figure_size(ref_len: int, gen_len: int) -> Tuple[float, float]:
    """Calculate appropriate figure size based on matrix dimensions."""
    fig_width = min(12, 8 + gen_len/20)
    fig_height = min(10, 6 + ref_len/20)
    return fig_width, fig_height

def calculate_tick_steps(ref_len: int, gen_len: int, matrix_size: MatrixSize) -> Tuple[int, int]:
    """Calculate appropriate tick step sizes based on matrix size."""
    if matrix_size.is_very_large:
        # For very large matrices, show fewer ticks
        x_step = max(1, gen_len // 10)
        y_step = max(1, ref_len // 10)
    elif matrix_size.is_large:
        # For large matrices, reduce tick density
        x_step = max(1, gen_len // 20)
        y_step = max(1, ref_len // 20)
    else:
        # For smaller matrices, use more ticks
        x_step = max(1, gen_len // 40)
        y_step = max(1, ref_len // 40)
    
    return x_step, y_step

def setup_axis_ticks(ax: plt.Axes, ref_len: int, gen_len: int, 
                    x_step: int, y_step: int) -> None:
    """Set up x and y axis ticks and labels."""
    # Set x-axis ticks and labels
    x_ticks = np.arange(0, gen_len, x_step)
    ax.set_xticks(x_ticks + 0.5)
    ax.set_xticklabels(x_ticks)
    
    # Set y-axis ticks and labels
    y_indices = np.arange(0, ref_len, y_step)
    y_ticks = np.array([ref_len - 1 - y for y in y_indices])  # Flip for visualization
    ax.set_yticks(y_ticks + 0.5)
    ax.set_yticklabels(y_indices)