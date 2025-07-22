import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Any, List, Tuple

from .match_details import create_precision_match_details_page, create_recall_match_details_page
from .summary_tables import create_precision_matches_summary_page, create_recall_matches_summary_page
from .empty_states import create_empty_precision_matches_page, create_empty_recall_matches_page


def generate_best_match_pages(internals: Dict[str, Any], pdf: PdfPages, start_page: int) -> int:
    """Generate best match pages with proper figure management to prevent console display."""
    page_count = 0
    current_page = start_page
    
    # Get best match information
    best_match_info = internals.get('best_match', {})
    precision_match_details = best_match_info.get('precision', {})
    recall_match_details = best_match_info.get('recall', {})
    
    # Get configuration parameters
    context_cutoff_value = internals['config'].get('context_cutoff_value', 0.6)
    context_window_control = internals['config'].get('context_window_control', 4.0)
    
    # 1. Generate precision matching details pages
    current_page += _generate_precision_detail_pages(
        precision_match_details, context_cutoff_value, context_window_control, 
        pdf, current_page
    )
    
    # 2. Generate recall matching details pages  
    current_page += _generate_recall_detail_pages(
        recall_match_details, context_cutoff_value, context_window_control,
        pdf, current_page
    )
    
    # Get text chunks and match information for summary tables
    text_data = _extract_text_and_match_data(internals)
    
    # 3. Generate precision matches summary tables
    current_page += _generate_precision_summary_pages(text_data, pdf, current_page)
    
    # 4. Generate recall matches summary tables
    current_page += _generate_recall_summary_pages(text_data, pdf, current_page)
    
    return current_page - start_page


def generate_content_pages(
    pdf: PdfPages, 
    sections_to_use: List[Tuple], 
    internals: Dict[str, Any], 
    current_page: int,
    include_all: bool,
    metrics_list: List[str]
) -> int:
    """Generate all content pages for the PDF report."""
    from ..._text_chunks import visualize_text_chunks
    from ..._line_nas import visualize_line_nas_precision_calculations, visualize_line_nas_recall_calculations
    
    for section_name, section_items in sections_to_use:
        for item_name, metric_key, item_generator in section_items:
            if metric_key == "Best Match" and (include_all or "Best Match" in metrics_list):
                # Special handling for Best Match section - memory efficient version
                best_match_pages = generate_best_match_pages(internals, pdf, current_page)
                current_page += best_match_pages
            elif item_generator == "PAGINATED_CONTENT":
                # Handle paginated content - generate ALL pages
                if item_name == "Reference Chunks":
                    current_page += generate_text_chunks_pages(pdf, internals, current_page, 'reference')
                elif item_name == "Generated Chunks":
                    current_page += generate_text_chunks_pages(pdf, internals, current_page, 'generated')
                elif item_name == "Line NAS Precision":
                    current_page += generate_line_nas_pages(pdf, internals, current_page, 'precision')
                elif item_name == "Line NAS Recall":
                    current_page += generate_line_nas_pages(pdf, internals, current_page, 'recall')
            else:
                # Generate figure on demand
                fig = item_generator()
                if fig is not None:
                    # Add page number
                    fig.text(0.5, 0.02, f"Page {current_page}", 
                            ha='center', va='bottom', fontsize=9, color='#555555')
                    
                    # Add to PDF
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)  # Important: close the figure immediately
                    current_page += 1
    
    return current_page


def generate_text_chunks_pages(pdf: PdfPages, internals: Dict[str, Any], start_page: int, chunk_type: str) -> int:
    """Generate all pages for text chunks without displaying them."""
    from ..._text_chunks import visualize_text_chunks
    
    # Get all text chunk figures
    text_chunks_figures = visualize_text_chunks(internals)
    
    # Select the appropriate chunk type
    if chunk_type == 'reference':
        figures_to_process = text_chunks_figures['reference_chunks']
        # Close generated chunks figures to prevent display
        for fig in text_chunks_figures['generated_chunks']:
            plt.close(fig)
    else:  # generated
        figures_to_process = text_chunks_figures['generated_chunks']
        # Close reference chunks figures to prevent display
        for fig in text_chunks_figures['reference_chunks']:
            plt.close(fig)
    
    pages_added = 0
    for fig in figures_to_process:
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page + pages_added}", 
                ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and close immediately
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        pages_added += 1
    
    return pages_added


def generate_line_nas_pages(pdf: PdfPages, internals: Dict[str, Any], start_page: int, calc_type: str) -> int:
    """Generate all pages for line NAS calculations without displaying them."""
    from ..._line_nas import visualize_line_nas_precision_calculations, visualize_line_nas_recall_calculations
    
    # Get all line NAS calculation figures
    if calc_type == 'precision':
        figures_to_process = visualize_line_nas_precision_calculations(internals)
    else:  # recall
        figures_to_process = visualize_line_nas_recall_calculations(internals)
    
    pages_added = 0
    for fig in figures_to_process:
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page + pages_added}", 
                ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and close immediately
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        pages_added += 1
    
    return pages_added


def _generate_precision_detail_pages(
    precision_match_details: Dict[str, Any],
    context_cutoff_value: float,
    context_window_control: float,
    pdf: PdfPages,
    start_page: int
) -> int:
    """Generate precision detail pages with immediate figure closure."""
    if not precision_match_details or 'segments' not in precision_match_details:
        return 0
        
    precision_segments = precision_match_details.get('segments', [])
    if not precision_segments:
        return 0
    
    # Calculate total pages
    total_pages = (len(precision_segments) + 3) // 4
    pages_added = 0
    
    # Generate pages one at a time with immediate closure
    for batch_idx in range(0, len(precision_segments), 4):
        batch_segments = precision_segments[batch_idx:batch_idx + 4]
        
        # Create figure
        fig = create_precision_match_details_page(
            batch_segments, 
            context_cutoff_value, 
            context_window_control,
            batch_idx // 4 + 1,
            total_pages
        )
        
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page + pages_added}", 
                 ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and immediately close to prevent display
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        pages_added += 1
    
    return pages_added


def _generate_recall_detail_pages(
    recall_match_details: Dict[str, Any],
    context_cutoff_value: float,
    context_window_control: float,
    pdf: PdfPages,
    start_page: int
) -> int:
    """Generate recall detail pages with immediate figure closure."""
    if not recall_match_details or 'segments' not in recall_match_details:
        return 0
        
    recall_segments = recall_match_details.get('segments', [])
    if not recall_segments:
        return 0
    
    # Calculate total pages
    total_pages = (len(recall_segments) + 3) // 4
    pages_added = 0
    
    # Generate pages one at a time with immediate closure
    for batch_idx in range(0, len(recall_segments), 4):
        batch_segments = recall_segments[batch_idx:batch_idx + 4]
        
        # Create figure
        fig = create_recall_match_details_page(
            batch_segments, 
            context_cutoff_value, 
            context_window_control,
            batch_idx // 4 + 1,
            total_pages
        )
        
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page + pages_added}", 
                 ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and immediately close to prevent display
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        pages_added += 1
    
    return pages_added


def _extract_text_and_match_data(internals: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text and match data for summary tables."""
    return {
        'ref_chunks': internals['texts']['reference_chunks'],
        'gen_chunks': internals['texts']['generated_chunks'],
        'precision_matches': internals['alignment']['precision']['matches'],
        'precision_sim_values': internals['alignment']['precision']['similarity_values'],
        'recall_matches': internals['alignment']['recall']['matches'],
        'recall_sim_values': internals['alignment']['recall']['similarity_values']
    }


def _generate_precision_summary_pages(text_data: Dict[str, Any], pdf: PdfPages, start_page: int) -> int:
    """Generate precision summary pages with proper figure management."""
    precision_matches = text_data['precision_matches']
    
    if precision_matches:
        return _generate_match_summary_pages(
            precision_matches,
            text_data['ref_chunks'],
            text_data['gen_chunks'],
            text_data['precision_sim_values'],
            create_precision_matches_summary_page,
            pdf,
            start_page
        )
    else:
        # Create a single page showing no precision matches
        fig = create_empty_precision_matches_page()
        
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page}", 
                 ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and immediately close
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        return 1


def _generate_recall_summary_pages(text_data: Dict[str, Any], pdf: PdfPages, start_page: int) -> int:
    """Generate recall summary pages with proper figure management."""
    recall_matches = text_data['recall_matches']
    
    if recall_matches:
        return _generate_match_summary_pages(
            recall_matches,
            text_data['ref_chunks'],
            text_data['gen_chunks'],
            text_data['recall_sim_values'],
            create_recall_matches_summary_page,
            pdf,
            start_page
        )
    else:
        # Create a single page showing no recall matches
        fig = create_empty_recall_matches_page()
        
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page}", 
                 ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and immediately close
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        return 1


def _generate_match_summary_pages(
    matches: List,
    ref_chunks: List[str],
    gen_chunks: List[str],
    sim_values: List[float],
    page_creator_func,
    pdf: PdfPages,
    start_page: int
) -> int:
    """Generate match summary pages with immediate figure closure."""
    max_rows_per_page = 25
    total_pages = (len(matches) + max_rows_per_page - 1) // max_rows_per_page
    pages_added = 0
    
    for page_idx in range(total_pages):
        start_idx = page_idx * max_rows_per_page
        end_idx = min(start_idx + max_rows_per_page, len(matches))
        page_matches = matches[start_idx:end_idx]
        
        # Create figure
        fig = page_creator_func(
            page_matches,
            ref_chunks,
            gen_chunks,
            sim_values,
            page_idx + 1,
            total_pages
        )
        
        # Add page number
        fig.text(0.5, 0.02, f"Page {start_page + pages_added}", 
                 ha='center', va='bottom', fontsize=9, color='#555555')
        
        # Save to PDF and immediately close to prevent display
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        pages_added += 1
    
    return pages_added