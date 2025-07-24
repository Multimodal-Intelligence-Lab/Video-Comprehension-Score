import matplotlib.pyplot as plt
from typing import Dict, Any
from ._utils import (
    create_precision_details_figure,
    create_recall_details_figure, 
    create_summary_table_figure
)

def visualize_best_match(internals: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """Create detailed visualizations of best match analysis for precision and recall.
    
    Generates three complementary visualizations showing how segments were matched
    between reference and generated texts, including detailed match information
    and summary statistics.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain alignment and best match data.
    
    Returns
    -------
    dict
        Dictionary containing three matplotlib figures:
        
        * ``'precision_details'`` : Figure showing precision matching details
        * ``'recall_details'`` : Figure showing recall matching details  
        * ``'summary_table'`` : Figure with summary statistics table
    
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
        figures = visualize_best_match(result['internals'])
        figures['precision_details'].show()
        figures['recall_details'].show()
        figures['summary_table'].show()
    
    See Also
    --------
    visualize_similarity_matrix : See similarity matrix used for matching
    visualize_las : See LAS metrics computed from these matches
    """

    precision_fig = create_precision_details_figure(internals)
    recall_fig = create_recall_details_figure(internals)
    summary_fig = create_summary_table_figure(internals)
    
    # Return as a dictionary for organized access
    # This pattern follows the established convention used by visualize_text_chunks
    return {
        'precision_details': precision_fig,
        'recall_details': recall_fig,
        'summary_table': summary_fig
    }