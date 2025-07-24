import matplotlib.pyplot as plt
from typing import Dict, Any, List
from ._utils import (
    split_text_chunks_for_display, should_paginate_chunks, create_text_chunk_figure
)

def _visualize_chunks_generic(chunks: List[str], title: str, chunk_size: int) -> List[plt.Figure]:
    # If there are only a few chunks, display them all in one figure
    if not should_paginate_chunks(chunks):
        fig = create_text_chunk_figure(
            title,
            [(i+1, chunk) for i, chunk in enumerate(chunks)],
            len(chunks),
            1, 1,
            chunk_size
        )
        return [fig]
    
    # For many chunks, paginate
    pages = split_text_chunks_for_display(chunks)
    figures = []
    
    for i, page_chunks in enumerate(pages):
        fig = create_text_chunk_figure(
            title,
            page_chunks,
            len(chunks),
            i+1,
            len(pages),
            chunk_size
        )
        figures.append(fig)
    
    return figures

def visualize_reference_chunks(internals: Dict[str, Any]) -> List[plt.Figure]:
    """Create visualization of reference text chunks.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary from VCS computation.
    
    Returns
    -------
    list of matplotlib.figure.Figure
        One or more figures showing reference text chunks.
    
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
        figures = visualize_reference_chunks(result['internals'])
        for fig in figures:
            fig.show()
    """
    ref_chunks = internals['texts']['reference_chunks']
    chunk_size = internals['config'].get('chunk_size', 1)
    
    return _visualize_chunks_generic(ref_chunks, "Reference Text Chunks", chunk_size)

def visualize_generated_chunks(internals: Dict[str, Any]) -> List[plt.Figure]:
    """Create visualization of generated text chunks.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary from VCS computation.
    
    Returns
    -------
    list of matplotlib.figure.Figure
        One or more figures showing generated text chunks.
    
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
        figures = visualize_generated_chunks(result['internals'])
        for fig in figures:
            fig.show()
    """
    gen_chunks = internals['texts']['generated_chunks']
    chunk_size = internals['config'].get('chunk_size', 1)
    
    return _visualize_chunks_generic(gen_chunks, "Generated Text Chunks", chunk_size)

def visualize_text_chunks(internals: Dict[str, Any]) -> Dict[str, List[plt.Figure]]:
    """Visualize the segmented and chunked text content from both reference and generated texts.
    
    Creates structured text displays showing how the input texts were segmented and
    grouped into chunks for analysis. For large numbers of chunks, automatically 
    creates multiple pages for better readability. Essential for understanding how
    the algorithm processed the input texts.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain 'texts' section with chunk data.
    
    Returns
    -------
    dict
        Dictionary with keys 'reference_chunks' and 'generated_chunks', each 
        containing a list of matplotlib figures:
        
        * ``'reference_chunks'`` : list of plt.Figure
            One or more figures showing reference text chunks
        * ``'generated_chunks'`` : list of plt.Figure  
            One or more figures showing generated text chunks
    
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
        chunk_figs = visualize_text_chunks(result['internals'])
        
        # Display reference chunks
        for fig in chunk_figs['reference_chunks']:
            fig.show()
        
        # Display generated chunks  
        for fig in chunk_figs['generated_chunks']:
            fig.show()
    
    See Also
    --------
    visualize_similarity_matrix : See how chunks relate to each other
    visualize_mapping_windows : Understand chunk alignment windows
    """
    ref_figs = visualize_reference_chunks(internals)
    gen_figs = visualize_generated_chunks(internals)
    
    return {
        'reference_chunks': ref_figs,
        'generated_chunks': gen_figs
    }