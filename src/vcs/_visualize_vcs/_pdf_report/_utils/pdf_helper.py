import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Union, Callable


@contextmanager
def pdf_matplotlib_context():
    """Context manager to set matplotlib for PDF generation without affecting global state."""
    # Store original settings with minimal interference
    original_interactive = matplotlib.is_interactive()
    original_show = plt.show
    
    # Keep track of existing figures before we start
    existing_figures = set(plt.get_fignums())
    
    try:
        # Only disable show function during PDF generation - don't touch backend or interactive mode
        plt.show = lambda *args, **kwargs: None
        
        yield
        
    finally:
        # Restoration phase - be very conservative
        try:
            # 1. Close only figures created during PDF generation
            current_figures = set(plt.get_fignums())
            new_figures = current_figures - existing_figures
            
            for fig_num in new_figures:
                try:
                    plt.close(fig_num)
                except Exception:
                    pass
            
            # 2. Restore the show function immediately
            plt.show = original_show
            
            # 3. Ensure interactive mode is restored if it was originally on
            if original_interactive and not matplotlib.is_interactive():
                plt.ion()
            elif not original_interactive and matplotlib.is_interactive():
                plt.ioff()
                
        except Exception:
            # Fallback: at minimum restore the show function
            plt.show = original_show
            if original_interactive:
                plt.ion()


def setup_pdf_metadata(pdf: PdfPages) -> None:
    d = pdf.infodict()
    d['Title'] = 'VCS Metrics Analysis Report'
    d['Author'] = 'VCS Library'
    d['Subject'] = 'Automatic analysis of narrative similarity metrics'
    d['Keywords'] = 'VCS, NAS, LAS, GAS, text similarity'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()


def normalize_metrics_list(metrics_to_include: Union[str, List[str]]) -> Tuple[bool, List[str]]:
    if metrics_to_include == "all":
        return True, []  # Not used when include_all is True
    elif isinstance(metrics_to_include, str):
        return False, [metrics_to_include]
    else:
        return False, metrics_to_include


def extract_key_metrics(internals: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from internals for summary pages."""
    return {
        'vcs_score': internals['metrics']['vcs']['value'],
        'gas_score': internals['metrics']['gas']['value'],
        'las_score': internals['metrics']['las']['f1'],
        'nas_score': internals['metrics']['nas']['regularized_nas'],
        'ref_len': internals['texts']['reference_length'],
        'gen_len': internals['texts']['generated_length']
    }


def create_section_structure(
    visualize_config: Callable,
    visualize_metrics_summary: Callable,
    visualize_text_chunks: Callable,
    visualize_similarity_matrix: Callable,
    visualize_mapping_windows: Callable,
    visualize_las: Callable,
    visualize_distance_nas: Callable,
    visualize_line_nas: Callable,
    visualize_line_nas_precision_calculations: Callable,
    visualize_line_nas_recall_calculations: Callable,
    visualize_window_regularizer: Callable,
    internals: Dict[str, Any]
) -> List[Tuple]:
    return [
        ("Overview", [
            ("Configuration", "Config", lambda: visualize_config(internals)),
            ("Metrics Summary", "Overview", lambda: visualize_metrics_summary(internals))
        ]),
        ("Text Analysis", [
            ("Reference Chunks", "Text Chunks", lambda: visualize_text_chunks(internals)['reference_chunks'][0]),
            ("Generated Chunks", "Text Chunks", lambda: visualize_text_chunks(internals)['generated_chunks'][0])
        ]),
        ("Similarity Analysis", [
            ("Similarity Matrix", "Similarity Matrix", lambda: visualize_similarity_matrix(internals)),
            ("Mapping Windows", "Mapping Windows", lambda: visualize_mapping_windows(internals))
        ]),
        ("Best Match Analysis", [
            ("Best Match Details", "Best Match", lambda: None)  # Special handling for paginated content
        ]),
        ("Local Alignment Score (LAS)", [
            ("LAS Visualization", "LAS", lambda: visualize_las(internals))
        ]),
        ("Narrative Alignment Score (NAS)", [
            ("Distance-based NAS", "NAS Distance", lambda: visualize_distance_nas(internals)),
            ("Line-based NAS", "NAS Line", lambda: visualize_line_nas(internals)),
            ("Line NAS Precision", "NAS Line", lambda: visualize_line_nas_precision_calculations(internals)[0]),
            ("Line NAS Recall", "NAS Line", lambda: visualize_line_nas_recall_calculations(internals)[0]),
            ("Window Regularization", "Window Regularizer", lambda: visualize_window_regularizer(internals))
        ])
    ]


def estimate_pages_for_metric(metric_key: str, internals: Dict[str, Any]) -> int:

    if metric_key == "Best Match":
        return estimate_best_match_pages(internals)
    elif metric_key == "Text Chunks":
        # Text chunks might be paginated
        ref_chunks = internals['texts']['reference_chunks']
        gen_chunks = internals['texts']['generated_chunks']
        ref_pages = max(1, (len(ref_chunks) + 39) // 40) if ref_chunks else 1
        gen_pages = max(1, (len(gen_chunks) + 39) // 40) if gen_chunks else 1
        return ref_pages + gen_pages
    elif metric_key == "NAS Line":
        # Line NAS calculations might be paginated
        precision_segments = internals.get('metrics', {}).get('nas', {}).get('nas_l', {}).get('precision', {}).get('segments', [])
        recall_segments = internals.get('metrics', {}).get('nas', {}).get('nas_l', {}).get('recall', {}).get('segments', [])
        precision_pages = max(1, (len(precision_segments) + 14) // 15) if precision_segments else 1
        recall_pages = max(1, (len(recall_segments) + 14) // 15) if recall_segments else 1
        return 1 + precision_pages + recall_pages  # +1 for main line NAS visualization
    else:
        # Most other metrics are single-page
        return 1


def estimate_best_match_pages(internals: Dict[str, Any]) -> int:
    """Estimate the number of pages needed for best match content."""
    precision_segments = internals.get('best_match', {}).get('precision', {}).get('segments', [])
    recall_segments = internals.get('best_match', {}).get('recall', {}).get('segments', [])
    precision_matches = internals['alignment']['precision']['matches']
    recall_matches = internals['alignment']['recall']['matches']
    
    # Estimate pages (4 segments per detail page, 25 matches per summary page)
    precision_detail_pages = max(1, (len(precision_segments) + 3) // 4) if precision_segments else 0
    recall_detail_pages = max(1, (len(recall_segments) + 3) // 4) if recall_segments else 0
    precision_match_pages = max(1, (len(precision_matches) + 24) // 25) if precision_matches else 1
    recall_match_pages = max(1, (len(recall_matches) + 24) // 25) if recall_matches else 1
    
    return precision_detail_pages + recall_detail_pages + precision_match_pages + recall_match_pages


def estimate_paginated_content_pages(item_name: str, internals: Dict[str, Any]) -> int:
    """Estimate the number of pages needed for paginated content."""
    if item_name == "Reference Chunks":
        ref_chunks = internals['texts']['reference_chunks']
        return max(1, (len(ref_chunks) + 39) // 40)  # 40 chunks per page
    elif item_name == "Generated Chunks":
        gen_chunks = internals['texts']['generated_chunks']
        return max(1, (len(gen_chunks) + 39) // 40)  # 40 chunks per page
    elif item_name == "Line NAS Precision":
        precision_segments = internals.get('metrics', {}).get('nas', {}).get('nas_l', {}).get('precision', {}).get('segments', [])
        return max(1, (len(precision_segments) + 14) // 15)  # 15 segments per page
    elif item_name == "Line NAS Recall":
        recall_segments = internals.get('metrics', {}).get('nas', {}).get('nas_l', {}).get('recall', {}).get('segments', [])
        return max(1, (len(recall_segments) + 14) // 15)  # 15 segments per page
    else:
        return 1


def validate_metrics_list(metrics_list: List[str]) -> List[str]:
    valid_metrics = {
        "Config", "Overview", "Text Chunks", "Similarity Matrix", 
        "Mapping Windows", "Best Match", "LAS", "NAS Distance", 
        "NAS Line", "Window Regularizer"
    }
    
    validated = []
    for metric in metrics_list:
        if metric in valid_metrics:
            validated.append(metric)
        else:
            print(f"Warning: Unknown metric '{metric}' ignored. Valid metrics: {sorted(valid_metrics)}")
    
    return validated


def calculate_content_layout(
    include_all: bool, 
    metrics_list: List[str], 
    toc_entries_count: int
) -> Dict[str, Any]:
    include_toc = toc_entries_count > 1
    include_metrics_page = include_all or len(metrics_list) > 1
    
    if include_toc:
        toc_page_num = 3 if include_metrics_page else 2
        content_start_page = toc_page_num + 1
    else:
        content_start_page = 3 if include_metrics_page else 2
    
    return {
        'include_toc': include_toc,
        'include_metrics_page': include_metrics_page,
        'toc_page_num': toc_page_num if include_toc else None,
        'content_start_page': content_start_page
    }


def add_page_number(fig, page_number: int, color: str = '#555555') -> None:
    fig.text(0.5, 0.02, f"Page {page_number}", 
             ha='center', va='bottom', fontsize=9, color=color)


def filter_sections_and_calculate_pages(
    all_sections: List[Tuple], 
    include_all: bool, 
    metrics_list: List[str], 
    internals: Dict[str, Any]
) -> Tuple[List[Tuple], List[Tuple]]:
    """Filter sections based on metrics and calculate page numbers."""
    current_page = 2  # Start after title page
    toc_data = []
    
    # Only include metrics page if more than one metric is being shown or all metrics
    if include_all or len(metrics_list) > 1:
        current_page += 1  # Add page for metrics summary
    
    sections_to_use = []
    
    for section_name, section_items in all_sections:
        if not include_all:
            # Filter items based on metrics_list
            section_items = [
                (item_name, metric_key, item_generator) 
                for item_name, metric_key, item_generator in section_items
                if metric_key in metrics_list
            ]
        
        if not section_items:
            continue
        
        # Mark paginated content for special handling
        updated_section_items = []
        for item_name, metric_key, item_generator in section_items:
            if item_name in ["Reference Chunks", "Generated Chunks", "Line NAS Precision", "Line NAS Recall"]:
                # Mark as paginated content - will be handled specially in _generate_content_pages
                updated_section_items.append((item_name, metric_key, "PAGINATED_CONTENT"))
            else:
                updated_section_items.append((item_name, metric_key, item_generator))
            
        section_entries = []
        
        for item_name, metric_key, item_generator in updated_section_items:
            if metric_key == "Best Match":
                # Special handling for Best Match - estimate pages needed
                if "Best Match" in metrics_list or include_all:
                    best_match_pages = estimate_best_match_pages(internals)
                    section_entries.append((item_name, current_page))
                    current_page += best_match_pages
            elif item_generator == "PAGINATED_CONTENT":
                # Estimate pages for paginated content
                pages_needed = estimate_paginated_content_pages(item_name, internals)
                section_entries.append((item_name, current_page))
                current_page += pages_needed
            else:
                # Regular single-page item
                section_entries.append((item_name, current_page))
                current_page += 1
        
        if section_entries:
            sections_to_use.append((section_name, updated_section_items))
            toc_data.append((section_name, section_entries))
    
    return sections_to_use, toc_data


def determine_layout_config(include_all: bool, metrics_list: List[str], toc_data: List[Tuple]) -> Dict[str, Any]:
    """Determine layout configuration for the PDF."""
    include_toc = len(toc_data) > 1
    include_metrics_page = include_all or len(metrics_list) > 1
    
    if include_toc:
        toc_page_num = 3 if include_metrics_page else 2
        content_start_page = toc_page_num + 1
    else:
        content_start_page = 3 if include_metrics_page else 2
    
    return {
        'include_toc': include_toc,
        'include_metrics_page': include_metrics_page,
        'toc_page_num': toc_page_num if include_toc else None,
        'content_start_page': content_start_page
    }