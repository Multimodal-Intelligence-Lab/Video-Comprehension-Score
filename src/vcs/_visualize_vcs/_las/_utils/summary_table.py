import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def create_load_sharing_summary_table_figure(las_internals: Dict[str, Any]) -> plt.Figure:
    """Create a summary table showing load sharing statistics for both directions."""
    
    precision_internals = las_internals.get('precision_internals', {})
    recall_internals = las_internals.get('recall_internals', {})
    
    # Extract data for both directions
    precision_stats = _extract_load_sharing_stats(precision_internals, "Precision")
    recall_stats = _extract_load_sharing_stats(recall_internals, "Recall")
    
    # Create figure with summary table
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('LAS Load Sharing Summary Statistics', fontsize=16, fontweight='bold')
    
    # Create comprehensive summary table
    _create_summary_table(ax, precision_stats, recall_stats)
    
    return fig

def _extract_load_sharing_stats(direction_internals: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """Extract load sharing statistics for one direction."""
    load_sharing_cases = direction_internals.get('load_sharing_details', [])
    original_similarities = np.array(direction_internals.get('original_similarities', []))
    adjusted_similarities = np.array(direction_internals.get('adjusted_similarities', []))
    original_las = direction_internals.get('original_las', 0.0)
    adjusted_las = direction_internals.get('adjusted_las', 0.0)
    
    # Calculate statistics
    num_cases = len(load_sharing_cases)
    total_chunks = len(original_similarities)
    chunks_affected = sum(len(case.get('chunk_details', [])) for case in load_sharing_cases)
    
    # Calculate penalty statistics
    similarity_reductions = original_similarities - adjusted_similarities
    avg_reduction = np.mean(similarity_reductions[similarity_reductions > 0]) if np.any(similarity_reductions > 0) else 0.0
    max_reduction = np.max(similarity_reductions) if len(similarity_reductions) > 0 else 0.0
    
    las_reduction = original_las - adjusted_las
    las_reduction_pct = (las_reduction / original_las * 100) if original_las > 0 else 0.0
    
    # Coverage statistics
    semantic_coverages = [case.get('semantic_coverage', 0.0) for case in load_sharing_cases]
    avg_semantic_coverage = np.mean(semantic_coverages) if semantic_coverages else 1.0
    min_semantic_coverage = np.min(semantic_coverages) if semantic_coverages else 1.0
    
    return {
        'direction': direction,
        'num_cases': num_cases,
        'total_chunks': total_chunks,
        'chunks_affected': chunks_affected,
        'original_las': original_las,
        'adjusted_las': adjusted_las,
        'las_reduction': las_reduction,
        'las_reduction_pct': las_reduction_pct,
        'avg_reduction': avg_reduction,
        'max_reduction': max_reduction,
        'avg_semantic_coverage': avg_semantic_coverage,
        'min_semantic_coverage': min_semantic_coverage
    }

def _create_summary_table(ax: plt.Axes, precision_stats: Dict[str, Any], recall_stats: Dict[str, Any]) -> None:
    """Create a comprehensive summary table."""
    
    # Table data structure
    table_data = [
        ['Metric', 'Precision Direction', 'Recall Direction'],
        ['─' * 25, '─' * 18, '─' * 16],
        ['Load Sharing Cases', f"{precision_stats['num_cases']}", f"{recall_stats['num_cases']}"],
        ['Total Chunks', f"{precision_stats['total_chunks']}", f"{recall_stats['total_chunks']}"],
        ['Chunks Affected', f"{precision_stats['chunks_affected']}", f"{recall_stats['chunks_affected']}"],
        ['', '', ''],
        ['Original LAS', f"{precision_stats['original_las']:.4f}", f"{recall_stats['original_las']:.4f}"],
        ['Adjusted LAS', f"{precision_stats['adjusted_las']:.4f}", f"{recall_stats['adjusted_las']:.4f}"],
        ['LAS Reduction', f"{precision_stats['las_reduction']:.4f}", f"{recall_stats['las_reduction']:.4f}"],
        ['LAS Reduction %', f"{precision_stats['las_reduction_pct']:.1f}%", f"{recall_stats['las_reduction_pct']:.1f}%"],
        ['', '', ''],
        ['Avg Chunk Reduction', f"{precision_stats['avg_reduction']:.4f}", f"{recall_stats['avg_reduction']:.4f}"],
        ['Max Chunk Reduction', f"{precision_stats['max_reduction']:.4f}", f"{recall_stats['max_reduction']:.4f}"],
        ['', '', ''],
        ['Avg Semantic Coverage', f"{precision_stats['avg_semantic_coverage']:.4f}", f"{recall_stats['avg_semantic_coverage']:.4f}"],
        ['Min Semantic Coverage', f"{precision_stats['min_semantic_coverage']:.4f}", f"{recall_stats['min_semantic_coverage']:.4f}"],
    ]
    
    # Create table
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style separator row
    for j in range(3):
        table[(1, j)].set_facecolor('#E8F5E8')
    
    # Style section breaks (empty rows)
    for i in [5, 10, 13]:
        for j in range(3):
            table[(i, j)].set_facecolor('#F5F5F5')
    
    # Style metric names (first column)
    for i in range(2, len(table_data)):
        if table_data[i][0] and table_data[i][0] != '':
            table[(i, 0)].set_facecolor('#E3F2FD')
    
    ax.set_title('Load Sharing Impact Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add explanatory text
    explanation = ("Load sharing occurs when multiple chunks compete for the same best match target.\n"
                   "Semantic coverage measures the adequacy of shared semantic content.\n"
                   "The penalty reduces scores based on coverage and similarity strength.")
    
    ax.text(0.5, 0.05, explanation, 
           transform=ax.transAxes,
           ha='center', va='bottom',
           fontsize=9, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))