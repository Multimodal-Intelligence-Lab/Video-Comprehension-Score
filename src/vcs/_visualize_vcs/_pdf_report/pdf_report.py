import os
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Any, List, Union

from ._utils import (
    pdf_matplotlib_context,
    setup_pdf_metadata,
    normalize_metrics_list,
    extract_key_metrics,
    create_section_structure,
    filter_sections_and_calculate_pages,
    determine_layout_config,
    generate_front_matter,
    generate_content_pages,
    setup_matplotlib_style
)

def create_vcs_pdf_report(
    internals: Dict[str, Any], 
    output_file: str,
    metrics_to_include: Union[str, List[str]] = "all"
) -> None:
    """Generate a comprehensive PDF report of the VCS analysis.
    
    Creates a professional, multi-page PDF report containing all relevant 
    visualizations, metrics, and analysis details. Perfect for documentation,
    sharing results, or creating analysis archives.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain complete analysis data.
    output_file : str
        Path where the PDF report should be saved. Directory will be created
        if it doesn't exist. Should end with '.pdf'.
    metrics_to_include : str or list of str, default="all"
        Controls which sections to include in the report:
        
        * ``"all"`` : Include all available visualizations and analyses
        * List of specific metrics: Choose from:
          
          - ``"Config"`` : Configuration parameters
          - ``"Overview"`` : Metrics summary  
          - ``"Text Chunks"`` : Segmented text display
          - ``"Similarity Matrix"`` : Similarity heatmap
          - ``"Mapping Windows"`` : Alignment windows
          - ``"Best Match"`` : Match analysis details
          - ``"LAS"`` : Local Alignment Score analysis
          - ``"NAS Distance"`` : Distance-based NAS analysis
          - ``"NAS Line"`` : Line-based NAS analysis  
          - ``"Window Regularizer"`` : Regularization analysis
    
    Returns
    -------
    None
        The function saves the PDF to the specified file path.
    
    Examples
    --------
    **Create Complete Report:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True,
            return_all_metrics=True
        )
        create_vcs_pdf_report(result['internals'], 'analysis_report.pdf')
        print("Complete PDF report saved to analysis_report.pdf")
    
    **Create Focused Report:**
    
    .. code-block:: python
    
        # Include only key metrics for a summary report
        create_vcs_pdf_report(
            result['internals'], 
            'summary_report.pdf',
            metrics_to_include=["Config", "Overview", "LAS", "NAS Distance"]
        )
    
    **Create Report for Specific Analysis:**
    
    .. code-block:: python
    
        # Focus on similarity and alignment analysis
        create_vcs_pdf_report(
            result['internals'],
            'alignment_analysis.pdf', 
            metrics_to_include=["Similarity Matrix", "Best Match", "Mapping Windows"]
        )
    
    
    Notes
    -----
    **Report Structure:**
    
    * Title page with generation date and library version
    * Optional metrics summary page (when multiple metrics included)
    * Optional table of contents (when multiple sections included)  
    * Individual analysis sections as specified
    * Automatic page numbering and professional formatting
    
    **Performance Notes:**
    
    * Large datasets may take several minutes to generate complete reports
    * Consider using specific metrics list for faster generation
    * Reports with many text chunks or detailed calculations will be longer
    * PDF generation temporarily disables matplotlib display to avoid memory issues
    
    **Best Practices:**
    
    * Always include "Config" and "Overview" for documentation
    * Use descriptive file names with dates/versions
    * Consider file size when including all visualizations
    * Test with smaller metric sets first for large datasets
    
    See Also
    --------
    visualize_metrics_summary : Quick overview without full report
    visualize_config : Just configuration display
    
    Raises
    ------
    ValueError
        If unknown metric names are provided in metrics_to_include list.
    IOError  
        If output directory cannot be created or file cannot be written.
    """
    
    with pdf_matplotlib_context():
        # Import visualization functions - these need to be imported here to avoid circular imports
        from .._best_match import visualize_best_match
        from .._config import visualize_config
        from .._text_chunks import visualize_text_chunks
        from .._similarity_matrix import visualize_similarity_matrix
        from .._mapping_windows import visualize_mapping_windows
        from .._line_nas import visualize_line_nas, visualize_line_nas_precision_calculations, visualize_line_nas_recall_calculations
        from .._distance_nas import visualize_distance_nas
        from .._las import visualize_las
        from .._window_regularizer import visualize_window_regularizer
        from .._metrics_summary import visualize_metrics_summary
        
        # Set up consistent styling
        setup_matplotlib_style()
        
        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract key metrics for summary pages
        key_metrics = extract_key_metrics(internals)
        
        # Normalize metrics_to_include and create section structure
        include_all, metrics_list = normalize_metrics_list(metrics_to_include)
        all_sections = create_section_structure(
            visualize_config, visualize_metrics_summary, visualize_text_chunks,
            visualize_similarity_matrix, visualize_mapping_windows, visualize_las,
            visualize_distance_nas, visualize_line_nas, 
            visualize_line_nas_precision_calculations, visualize_line_nas_recall_calculations,
            visualize_window_regularizer, internals
        )
        
        # Filter sections and calculate page layout
        sections_to_use, toc_data = filter_sections_and_calculate_pages(
            all_sections, include_all, metrics_list, internals
        )
        
        # Determine layout configuration
        layout_config = determine_layout_config(include_all, metrics_list, toc_data)
        
        # Create the PDF
        with PdfPages(output_file) as pdf:
            current_page = generate_front_matter(
                pdf, key_metrics, layout_config, toc_data
            )
            
            current_page = generate_content_pages(
                pdf, sections_to_use, internals, current_page, include_all, metrics_list
            )
            
            # Set PDF metadata
            setup_pdf_metadata(pdf)