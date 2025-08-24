import matplotlib.pyplot as plt
from typing import Dict, List

def create_precision_load_sharing_page(
    batch_cases: List[Dict],
    page_num: int, 
    total_pages: int,
    direction_summary: Dict = None
) -> plt.Figure:
    """Create a precision load sharing page with batch of cases."""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create header
    header_text = "PRECISION LOAD SHARING DETAILS (Generation → Reference)\n"
    header_text += "=" * 80 + "\n\n"
    
    # Process load sharing cases
    content_text = header_text
    case_start_num = (page_num - 1) * 4 + 1
    
    for i, case in enumerate(batch_cases):
        case_num = case_start_num + i
        content_text += _format_load_sharing_case(case, case_num, "precision")
    
    # Add summary on last page
    if page_num == total_pages and direction_summary:
        content_text += _format_direction_summary(direction_summary, "precision")
    
    # Display the text
    ax.text(0.01, 0.99, content_text,
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    # Set title
    page_info = f"Precision Load Sharing (Page {page_num} of {total_pages})"
    ax.set_title(page_info, fontsize=14, fontweight='bold')
    
    return fig

def create_recall_load_sharing_page(
    batch_cases: List[Dict],
    page_num: int,
    total_pages: int, 
    direction_summary: Dict = None
) -> plt.Figure:
    """Create a recall load sharing page with batch of cases."""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create header
    header_text = "RECALL LOAD SHARING DETAILS (Reference → Generated)\n"
    header_text += "=" * 80 + "\n\n"
    
    # Process load sharing cases
    content_text = header_text
    case_start_num = (page_num - 1) * 4 + 1
    
    for i, case in enumerate(batch_cases):
        case_num = case_start_num + i
        content_text += _format_load_sharing_case(case, case_num, "recall")
    
    # Add summary on last page
    if page_num == total_pages and direction_summary:
        content_text += _format_direction_summary(direction_summary, "recall")
    
    # Display the text
    ax.text(0.01, 0.99, content_text,
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    # Set title
    page_info = f"Recall Load Sharing (Page {page_num} of {total_pages})"
    ax.set_title(page_info, fontsize=14, fontweight='bold')
    
    return fig

def _format_load_sharing_case(case: Dict, case_num: int, direction: str) -> str:
    """Format a single load sharing case."""
    if direction == "precision":
        target_idx = case.get('reference_chunk_idx', -1)
        num_competing = case.get('num_gens_shared', 0)
        target_label = f"Reference Chunk {target_idx}"
        competing_type = "Generation"
    else:
        target_idx = case.get('generated_chunk_idx', -1)
        num_competing = case.get('num_refs_shared', 0)
        target_label = f"Generated Chunk {target_idx}"
        competing_type = "Reference"
    
    semantic_coverage = case.get('semantic_coverage', 0.0)
    load_sharing_penalty = case.get('load_sharing_penalty', 0.0)
    chunk_details = case.get('chunk_details', [])
    
    text = f"LOAD SHARING CASE {case_num}: {target_label}\n"
    text += "-" * 80 + "\n"
    text += f"Competing {competing_type} chunks: {num_competing}\n"
    text += f"Semantic coverage: {semantic_coverage:.4f}\n"
    text += f"Load sharing penalty: {load_sharing_penalty:.4f}\n\n"
    
    # Show all affected chunks (not limited)
    for chunk in chunk_details:
        if direction == "precision":
            chunk_idx = chunk.get('generated_chunk_idx', -1)
            chunk_label = f"Gen[{chunk_idx}]"
        else:
            chunk_idx = chunk.get('reference_chunk_idx', -1)
            chunk_label = f"Ref[{chunk_idx}]"
            
        original_sim = chunk.get('original_similarity', 0.0)
        adjusted_sim = chunk.get('adjusted_similarity', 0.0)
        reduction = original_sim - adjusted_sim
        
        text += f"  {chunk_label}: {original_sim:.4f} → {adjusted_sim:.4f} (penalty: {reduction:.4f})\n"
    
    text += "\n"
    return text

def _format_direction_summary(direction_internals: Dict, direction: str) -> str:
    """Create a summary for one direction."""
    load_sharing_cases = direction_internals.get('load_sharing_details', [])
    original_las = direction_internals.get('original_las', 0.0)
    adjusted_las = direction_internals.get('adjusted_las', 0.0)
    
    num_cases = len(load_sharing_cases)
    total_chunks_affected = sum(len(case.get('chunk_details', [])) for case in load_sharing_cases)
    
    las_reduction = original_las - adjusted_las
    las_reduction_pct = (las_reduction / original_las * 100) if original_las > 0 else 0
    
    text = f"{direction.upper()} DIRECTION SUMMARY:\n"
    text += "-" * 30 + "\n"
    text += f"Load sharing cases: {num_cases}\n"
    text += f"Chunks affected: {total_chunks_affected}\n"
    text += f"Original LAS: {original_las:.4f}\n"
    text += f"Adjusted LAS: {adjusted_las:.4f}\n"
    text += f"Penalty impact: {las_reduction:.4f} ({las_reduction_pct:.1f}%)\n\n"
    
    return text