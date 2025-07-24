import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple
from .matrix_scaling import MatrixSize

def should_show_matches(matrix_size: MatrixSize, precision_matches: List, 
                       recall_matches: List) -> bool:
    """Determine if matches should be highlighted based on matrix size and match count."""
    total_matches = len(precision_matches) + len(recall_matches)
    return not matrix_size.is_very_large and total_matches < 100

def highlight_precision_matches(ax: plt.Axes, precision_matches: List[Tuple], 
                               ref_len: int, gen_len: int) -> None:
    """Add red rectangles to highlight precision matches."""
    for g_idx, r_idx in precision_matches:
        if 0 <= r_idx < ref_len and 0 <= g_idx < gen_len:
            flipped_r_idx = ref_len - 1 - r_idx
            rect = Rectangle((g_idx, flipped_r_idx), 1, 1, fill=False, 
                           edgecolor='red', lw=1.5, alpha=0.6)
            ax.add_patch(rect)

def highlight_recall_matches(ax: plt.Axes, recall_matches: List[Tuple], 
                            ref_len: int, gen_len: int) -> None:
    """Add blue rectangles to highlight recall matches."""
    for g_idx, r_idx in recall_matches:
        if 0 <= r_idx < ref_len and 0 <= g_idx < gen_len:
            flipped_r_idx = ref_len - 1 - r_idx
            rect = Rectangle((g_idx, flipped_r_idx), 1, 1, fill=False, 
                           edgecolor='blue', lw=1.5, alpha=0.6)
            ax.add_patch(rect)

def create_matrix_title(ref_len: int, gen_len: int, precision_matches: List, 
                       recall_matches: List, show_matches: bool) -> str:
    """Create descriptive title for the similarity matrix."""
    match_count = len(precision_matches) + len(recall_matches)
    title = f'Similarity Matrix ({ref_len}x{gen_len})'
    
    if show_matches:
        title += f' with Precision (red) and Recall (blue) Matches ({match_count} total)'
    else:
        title += f' ({match_count} matches not shown due to size)'
    
    return title

def highlight_all_matches(ax: plt.Axes, precision_matches: List[Tuple], 
                         recall_matches: List[Tuple], ref_len: int, gen_len: int,
                         show_matches: bool) -> None:
    """Highlight both precision and recall matches if conditions are met."""
    if show_matches:
        highlight_precision_matches(ax, precision_matches, ref_len, gen_len)
        highlight_recall_matches(ax, recall_matches, ref_len, gen_len)