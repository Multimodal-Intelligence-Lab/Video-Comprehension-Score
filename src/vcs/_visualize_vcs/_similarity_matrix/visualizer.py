import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from ._utils import (
    determine_matrix_size, calculate_figure_size, calculate_tick_steps, setup_axis_ticks,
    create_similarity_heatmap, should_show_matches, highlight_all_matches, create_matrix_title
)

def visualize_similarity_matrix(internals: Dict[str, Any]) -> plt.Figure:
    """Create a heatmap visualization of the similarity matrix between text chunks.
    
    Displays the cosine similarity values between all reference and generated text
    chunks as a color-coded matrix. Optionally highlights the best matches found
    during precision and recall alignment. Essential for understanding the semantic
    relationships discovered by the algorithm.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'similarity', 'alignment', and 
        'texts' sections.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the similarity matrix heatmap with optional match 
        highlighting. The matrix shows reference chunks on the y-axis and 
        generated chunks on the x-axis.
    
    Examples
    --------
    **Basic Usage:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True
        )
        fig = visualize_similarity_matrix(result['internals'])
        fig.show()
    
    See Also
    --------
    visualize_text_chunks : See the actual text content being compared
    visualize_best_match : Detailed analysis of matching decisions
    visualize_mapping_windows : See alignment constraints applied
    """
    # Extract data
    sim_matrix = np.array(internals['similarity']['matrix'])
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    precision_matches = internals['alignment']['precision']['matches']
    recall_matches = internals['alignment']['recall']['matches']
    
    # Determine matrix characteristics
    matrix_size = determine_matrix_size(ref_len, gen_len)
    fig_size = calculate_figure_size(ref_len, gen_len)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create the heatmap with appropriate annotations
    create_similarity_heatmap(ax, sim_matrix, matrix_size)
    
    # Set up axis ticks
    x_step, y_step = calculate_tick_steps(ref_len, gen_len, matrix_size)
    setup_axis_ticks(ax, ref_len, gen_len, x_step, y_step)
    
    # Highlight matches if appropriate
    show_matches = should_show_matches(matrix_size, precision_matches, recall_matches)
    highlight_all_matches(ax, precision_matches, recall_matches, ref_len, gen_len, show_matches)
    
    # Set labels and title
    ax.set_xlabel('Generated Text Segments')
    ax.set_ylabel('Reference Text Segments')
    
    title = create_matrix_title(ref_len, gen_len, precision_matches, recall_matches, show_matches)
    ax.set_title(title)
    
    # Apply layout
    fig.tight_layout()
    
    return fig