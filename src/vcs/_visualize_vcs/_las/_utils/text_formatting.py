import numpy as np
from typing import Dict, Any, List

def create_load_sharing_header(title: str, direction: str) -> str:
    """Create a section header for load sharing analysis."""
    header_text = f"{title} - {direction.upper()} DIRECTION\n"
    header_text += "=" * 80 + "\n\n"
    header_text += f"Analyzing semantic load sharing penalty in {direction} direction:\n"
    if direction == "precision":
        header_text += "Multiple generated chunks competing for the same reference chunk\n\n"
    else:
        header_text += "Multiple reference chunks competing for the same generated chunk\n\n"
    return header_text

def format_load_sharing_details(load_sharing_case: Dict[str, Any], direction: str) -> str:
    """Format detailed information for a single load sharing case."""
    if direction == "precision":
        target_idx = load_sharing_case.get('reference_chunk_idx', -1)
        num_competing = load_sharing_case.get('num_gens_shared', 0)
        target_label = f"Reference Chunk {target_idx}"
        competing_label = f"{num_competing} Generated chunks"
    else:
        target_idx = load_sharing_case.get('generated_chunk_idx', -1) 
        num_competing = load_sharing_case.get('num_refs_shared', 0)
        target_label = f"Generated Chunk {target_idx}"
        competing_label = f"{num_competing} Reference chunks"
    
    semantic_coverage = load_sharing_case.get('semantic_coverage', 0.0)
    load_sharing_penalty = load_sharing_case.get('load_sharing_penalty', 0.0)
    chunk_details = load_sharing_case.get('chunk_details', [])
    
    text = f"┌─ LOAD SHARING CASE: {target_label} ─┐\n"
    text += f"│ Competing chunks: {competing_label}\n"
    text += f"│ Semantic coverage: {semantic_coverage:.4f}\n" 
    text += f"│ Load sharing penalty: {load_sharing_penalty:.4f}\n"
    text += f"└─────────────────────────────────────┘\n\n"
    
    text += "Chunk-level details:\n"
    for i, chunk in enumerate(chunk_details):
        if direction == "precision":
            chunk_idx = chunk.get('generated_chunk_idx', -1)
            chunk_label = f"Gen[{chunk_idx}]"
        else:
            chunk_idx = chunk.get('reference_chunk_idx', -1)
            chunk_label = f"Ref[{chunk_idx}]"
            
        original_sim = chunk.get('original_similarity', 0.0)
        adjusted_sim = chunk.get('adjusted_similarity', 0.0)
        reduction = original_sim - adjusted_sim
        reduction_pct = (reduction / original_sim * 100) if original_sim > 0 else 0
        
        text += f"  {chunk_label}: {original_sim:.4f} → {adjusted_sim:.4f} "
        text += f"(↓{reduction:.4f}, -{reduction_pct:.1f}%)\n"
    
    text += "\n" + "-" * 60 + "\n\n"
    return text

def format_direction_summary(direction_internals: Dict[str, Any], direction: str) -> str:
    """Create a summary of load sharing for one direction."""
    load_sharing_cases = direction_internals.get('load_sharing_details', [])
    original_similarities = direction_internals.get('original_similarities', [])
    adjusted_similarities = direction_internals.get('adjusted_similarities', [])
    original_las = direction_internals.get('original_las', 0.0)
    adjusted_las = direction_internals.get('adjusted_las', 0.0)
    
    num_cases = len(load_sharing_cases)
    total_chunks_affected = sum(len(case.get('chunk_details', [])) for case in load_sharing_cases)
    
    text = f"\n{direction.upper()} DIRECTION SUMMARY:\n"
    text += "-" * 30 + "\n"
    text += f"Load sharing cases detected: {num_cases}\n"
    text += f"Total chunks affected: {total_chunks_affected}\n"
    text += f"Original LAS: {original_las:.4f}\n"
    text += f"Adjusted LAS: {adjusted_las:.4f}\n"
    
    las_reduction = original_las - adjusted_las
    las_reduction_pct = (las_reduction / original_las * 100) if original_las > 0 else 0
    text += f"LAS reduction: {las_reduction:.4f} ({las_reduction_pct:.1f}%)\n\n"
    
    return text