import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .matrix_scaling import MatrixSize

def create_similarity_heatmap(ax: plt.Axes, sim_matrix: np.ndarray, 
                             matrix_size: MatrixSize) -> None:
    """Create similarity heatmap with appropriate annotations based on matrix size."""
    # Flip the matrix for proper visualization (0 at the top)
    sim_matrix_flipped = sim_matrix[::-1]
    
    if matrix_size.is_very_large:
        # For very large matrices, use a heatmap without annotations
        _create_heatmap_no_annotations(ax, sim_matrix_flipped)
    elif matrix_size.is_large:
        # For large matrices, reduce annotation density by using smaller font
        _create_heatmap_reduced_annotations(ax, sim_matrix_flipped)
    else:
        # For smaller matrices, use full annotations
        _create_heatmap_full_annotations(ax, sim_matrix_flipped)

def _create_heatmap_no_annotations(ax: plt.Axes, sim_matrix_flipped: np.ndarray) -> None:
    """Create heatmap without annotations for very large matrices."""
    sns.heatmap(sim_matrix_flipped, ax=ax, cmap='viridis', vmin=0, vmax=1,
                annot=False,  # No annotations for very large matrices
                cbar_kws={'label': 'Cosine Similarity'})

def _create_heatmap_reduced_annotations(ax: plt.Axes, sim_matrix_flipped: np.ndarray) -> None:
    """Create heatmap with reduced annotations for large matrices."""
    sns.heatmap(sim_matrix_flipped, ax=ax, cmap='viridis', vmin=0, vmax=1,
                annot=True, fmt='.1f', annot_kws={"size": 6},
                cbar_kws={'label': 'Cosine Similarity'})

def _create_heatmap_full_annotations(ax: plt.Axes, sim_matrix_flipped: np.ndarray) -> None:
    """Create heatmap with full annotations for smaller matrices."""
    sns.heatmap(sim_matrix_flipped, ax=ax, cmap='viridis', vmin=0, vmax=1, 
                annot=True, fmt='.2f', annot_kws={"size": 8},
                cbar_kws={'label': 'Cosine Similarity'})