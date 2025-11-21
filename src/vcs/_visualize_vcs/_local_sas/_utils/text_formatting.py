from typing import Dict, Any, List

def create_las_section_header(title: str) -> str:
    """Create a section header for LAS load sharing analysis."""
    header_text = f"{title}\n"
    header_text += "=" * 80 + "\n\n"
    return header_text

def format_load_sharing_case(case: Dict[str, Any], case_num: int, direction: str) -> str:
    """Format a single load sharing case in simple text format."""
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
    
    # Show affected chunks (limit to first 3 for space)
    display_chunks = chunk_details[:3]
    for chunk in display_chunks:
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
    
    if len(chunk_details) > 3:
        remaining = len(chunk_details) - 3
        text += f"  ... and {remaining} more chunks\n"
    
    text += "\n"
    return text

def format_direction_summary(direction_internals: Dict[str, Any], direction: str) -> str:
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