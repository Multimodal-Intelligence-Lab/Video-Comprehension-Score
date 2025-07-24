import matplotlib.pyplot as plt
from typing import Dict, Any, List
from ._utils import (
    paginate_segments, should_paginate, create_base_calculation_figure, 
    finalize_calculation_figure, create_empty_segments_figure,
    generate_summary_text, generate_calculation_method_text, generate_lct_note,
    create_segment_table
)

def create_line_nas_calculation_figure(title: str, segments: List[Dict], 
                                     summary_data: Dict[str, float], 
                                     lct: int, page_num: int, total_pages: int) -> plt.Figure:
    """Create a figure for a page of Line NAS calculation details."""
    # Create base figure
    fig, ax = create_base_calculation_figure(title, page_num, total_pages)
    
    # Generate summary text
    summary_text = generate_summary_text(summary_data, segments)
    
    # Add calculation method explanation on first page only
    if page_num == 1:
        summary_text += generate_calculation_method_text(summary_data)
    
    # Display summary in a box
    plt.text(0.5, 0.85, summary_text, 
            bbox=dict(facecolor='#e6f3ff', alpha=0.5, boxstyle='round,pad=0.5'),
            ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Create and display segment table
    segments_text = create_segment_table(segments, lct)
    
    # Add LCT note if applicable
    segments_text += generate_lct_note(lct)
    
    # Display segment table
    plt.text(0.5, 0.4, segments_text, 
            ha='center', va='center', fontsize=8, family='monospace',
            transform=ax.transAxes)
    
    # Finalize figure
    finalize_calculation_figure(fig, page_num, total_pages)
    
    return fig

def visualize_line_nas_precision_calculations(internals: Dict[str, Any]) -> List[plt.Figure]:
    """Create detailed visualizations of precision Line-based NAS calculations.
    
    Provides comprehensive breakdown of how the precision NAS-L score was calculated,
    including segment-by-segment analysis, threshold applications, and calculation
    methods. Automatically creates multiple pages for large datasets.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain detailed line-based NAS calculations.
    
    Returns
    -------
    list of matplotlib.figure.Figure
        List of figures (one or more pages) showing detailed precision NAS-L 
        calculations, including:
        - Summary statistics and calculation methods
        - Segment-by-segment breakdown table
        - Threshold and LCT information
    
    Examples
    --------
    **Basic Usage:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True,
            return_all_metrics=True,
            lct=1
        )
        precision_figs = visualize_line_nas_precision_calculations(result['internals'])
        for fig in precision_figs:
            fig.show()
    * Indicates which calculation method was used for each segment
    * Displays LCT threshold applications when applicable
    * Essential for debugging unexpected NAS-L precision scores
    * Automatically paginated for datasets with >15 segments per page
    
    See Also
    --------
    visualize_line_nas : Overview of line-based analysis
    visualize_line_nas_recall_calculations : Corresponding recall analysis
    """
    precision_line_data = internals['metrics']['nas']['nas_l']['precision']
    lct = internals['config']['lct']
    segments = precision_line_data.get('segments', [])
    
    # Handle empty segments
    if not segments:
        return [create_empty_segments_figure(
            "Precision Line-based NAS Calculation Details", "Precision"
        )]
    
    # Check if pagination is needed
    if not should_paginate(segments):
        fig = create_line_nas_calculation_figure(
            "Precision Line-based NAS Calculation Details",
            segments,
            precision_line_data,
            lct,
            1, 1
        )
        return [fig]
    
    # Create paginated figures
    paginated_segments = paginate_segments(segments)
    figures = []
    
    for i, page_segments in enumerate(paginated_segments):
        fig = create_line_nas_calculation_figure(
            "Precision Line-based NAS Calculation Details",
            page_segments,
            precision_line_data,
            lct,
            i + 1,
            len(paginated_segments)
        )
        figures.append(fig)
    
    return figures

def visualize_line_nas_recall_calculations(internals: Dict[str, Any]) -> List[plt.Figure]:
    """Create detailed visualizations of recall Line-based NAS calculations.
    
    Provides comprehensive breakdown of how the recall NAS-L score was calculated,
    including segment-by-segment analysis, threshold applications, and calculation
    methods. Automatically creates multiple pages for large datasets.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain detailed line-based NAS calculations.
    
    Returns
    -------
    list of matplotlib.figure.Figure
        List of figures (one or more pages) showing detailed recall NAS-L 
        calculations, including:
        - Summary statistics and calculation methods
        - Segment-by-segment breakdown table  
        - Threshold and LCT information
    
    Examples
    --------
    **Basic Usage:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True,
            return_all_metrics=True,
            lct=1
        )
        recall_figs = visualize_line_nas_recall_calculations(result['internals'])
        for fig in recall_figs:
            fig.show()
    
    See Also
    --------
    visualize_line_nas : Overview of line-based analysis  
    visualize_line_nas_precision_calculations : Corresponding precision analysis
    """
    recall_line_data = internals['metrics']['nas']['nas_l']['recall']
    lct = internals['config']['lct']
    segments = recall_line_data.get('segments', [])
    
    # Handle empty segments
    if not segments:
        return [create_empty_segments_figure(
            "Recall Line-based NAS Calculation Details", "Recall"
        )]
    
    # Check if pagination is needed
    if not should_paginate(segments):
        fig = create_line_nas_calculation_figure(
            "Recall Line-based NAS Calculation Details",
            segments,
            recall_line_data,
            lct,
            1, 1
        )
        return [fig]
    
    # Create paginated figures
    paginated_segments = paginate_segments(segments)
    figures = []
    
    for i, page_segments in enumerate(paginated_segments):
        fig = create_line_nas_calculation_figure(
            "Recall Line-based NAS Calculation Details",
            page_segments,
            recall_line_data,
            lct,
            i + 1,
            len(paginated_segments)
        )
        figures.append(fig)
    
    return figures