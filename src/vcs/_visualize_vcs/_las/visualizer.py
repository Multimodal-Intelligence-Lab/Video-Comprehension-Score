import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def visualize_las(internals: Dict[str, Any]) -> plt.Figure:
    """Create a visualization of Local Alignment Score (LAS) with semantic load sharing penalties.
    
    Displays LAS precision and recall components showing original vs adjusted similarity
    values, penalties applied, and detailed sharing group information.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain LAS metrics and alignment data.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure showing detailed LAS analysis with penalty visualization.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Get LAS internals and data
    las_internals = internals['metrics']['las']['internals']
    precision_sim_values = np.array(internals['alignment']['precision']['similarity_values'])
    recall_sim_values = np.array(internals['alignment']['recall']['similarity_values'])
    
    # Print detailed penalty analysis
    print("=" * 80)
    print("LOCAL ALIGNMENT SCORE (LAS) - SEMANTIC LOAD SHARING ANALYSIS")
    print("=" * 80)
    print()
    
    # Original vs Adjusted averages
    print(f"ORIGINAL AVERAGES:")
    print(f"  Precision LAS (before penalty): {las_internals['original_precision_average']:.4f}")
    print(f"  Recall LAS (before penalty):    {las_internals['original_recall_average']:.4f}")
    print()
    print(f"ADJUSTED AVERAGES:")
    print(f"  Precision LAS (after penalty):  {las_internals['adjusted_precision_average']:.4f}")
    print(f"  Recall LAS (after penalty):     {las_internals['adjusted_recall_average']:.4f}")
    print()
    
    # Precision direction analysis
    precision_internals = las_internals['precision']
    print("PRECISION DIRECTION (Gen → Ref) - Semantic Load Sharing Analysis")
    print("-" * 70)
    
    if precision_internals['total_sharing_groups'] > 0:
        print(f"Total sharing groups detected: {precision_internals['total_sharing_groups']}")
        print(f"Total penalties applied: {precision_internals['total_penalties_applied']}")
        print()
        
        for i, group in enumerate(precision_internals['sharing_groups']):
            target_ref = group['target_ref_idx']
            sharing_gens = group['sharing_gen_indices']
            demand = group['demand']
            supply = group['supply']
            adequacy = group['adequacy_factor']
            threshold = group['threshold']
            
            print(f"Sharing Group {i+1}:")
            print(f"  Target Reference Chunk: {target_ref}")
            print(f"  Sharing Generated Chunks: {sharing_gens}")
            print(f"  Demand (chunks sharing): {demand}")
            print(f"  Supply (sum of similarities): {supply:.4f}")
            print(f"  Adequacy Factor (α): {adequacy:.4f}")
            print(f"  Penalty Threshold (1-α): {threshold:.4f}")
            print(f"  Similarities and Penalties:")
            
            for sim_info in group['similarities']:
                gen_idx = sim_info['gen_idx']
                original = sim_info['original_similarity']
                adjusted = sim_info['adjusted_similarity']
                penalty = sim_info['penalty']
                applied = sim_info['penalty_applied']
                
                status = "PENALIZED" if applied else "NO PENALTY"
                print(f"    Gen {gen_idx}: {original:.4f} → {adjusted:.4f} (penalty: {penalty:.4f}) [{status}]")
            print()
    else:
        print("No semantic load sharing detected in precision direction.")
        print()
    
    # Recall direction analysis
    recall_internals = las_internals['recall']
    print("RECALL DIRECTION (Ref → Gen) - Semantic Load Sharing Analysis")
    print("-" * 70)
    
    if recall_internals['total_sharing_groups'] > 0:
        print(f"Total sharing groups detected: {recall_internals['total_sharing_groups']}")
        print(f"Total penalties applied: {recall_internals['total_penalties_applied']}")
        print()
        
        for i, group in enumerate(recall_internals['sharing_groups']):
            target_gen = group['target_gen_idx']
            sharing_refs = group['sharing_ref_indices']
            demand = group['demand']
            supply = group['supply']
            adequacy = group['adequacy_factor']
            threshold = group['threshold']
            
            print(f"Sharing Group {i+1}:")
            print(f"  Target Generated Chunk: {target_gen}")
            print(f"  Sharing Reference Chunks: {sharing_refs}")
            print(f"  Demand (chunks sharing): {demand}")
            print(f"  Supply (sum of similarities): {supply:.4f}")
            print(f"  Adequacy Factor (α): {adequacy:.4f}")
            print(f"  Penalty Threshold (1-α): {threshold:.4f}")
            print(f"  Similarities and Penalties:")
            
            for sim_info in group['similarities']:
                ref_idx = sim_info['ref_idx']
                original = sim_info['original_similarity']
                adjusted = sim_info['adjusted_similarity']
                penalty = sim_info['penalty']
                applied = sim_info['penalty_applied']
                
                status = "PENALIZED" if applied else "NO PENALTY"
                print(f"    Ref {ref_idx}: {original:.4f} → {adjusted:.4f} (penalty: {penalty:.4f}) [{status}]")
            print()
    else:
        print("No semantic load sharing detected in recall direction.")
        print()
    
    # Create visual representation
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Precision subplot
    ax_precision = fig.add_subplot(gs[0, 0])
    x_indices = np.arange(len(precision_sim_values))
    
    # Get adjusted precision values
    adjusted_precision = precision_sim_values.copy()
    for penalty_info in precision_internals['penalties_applied']:
        gen_idx = penalty_info['gen_idx']
        adjusted_precision[gen_idx] = penalty_info['adjusted_sim']
    
    # Plot original and adjusted bars
    bars_orig = ax_precision.bar(x_indices - 0.2, precision_sim_values, width=0.4, 
                                alpha=0.7, color='skyblue', label='Original')
    bars_adj = ax_precision.bar(x_indices + 0.2, adjusted_precision, width=0.4, 
                               alpha=0.7, color='orange', label='Adjusted')
    
    ax_precision.axhline(y=las_internals['original_precision_average'], 
                        color='blue', linestyle='--', alpha=0.7, label='Original Avg')
    ax_precision.axhline(y=las_internals['adjusted_precision_average'], 
                        color='red', linestyle='--', alpha=0.7, label='Adjusted Avg')
    
    ax_precision.set_xlabel('Generation Index')
    ax_precision.set_ylabel('Similarity Value')
    ax_precision.set_title('Precision LAS: Before vs After Penalty')
    ax_precision.set_ylim(0, 1.05)
    ax_precision.legend()
    
    # Recall subplot
    ax_recall = fig.add_subplot(gs[0, 1])
    x_indices = np.arange(len(recall_sim_values))
    
    # Get adjusted recall values
    adjusted_recall = recall_sim_values.copy()
    for penalty_info in recall_internals['penalties_applied']:
        ref_idx = penalty_info['ref_idx']
        adjusted_recall[ref_idx] = penalty_info['adjusted_sim']
    
    # Plot original and adjusted bars
    bars_orig = ax_recall.bar(x_indices - 0.2, recall_sim_values, width=0.4, 
                             alpha=0.7, color='salmon', label='Original')
    bars_adj = ax_recall.bar(x_indices + 0.2, adjusted_recall, width=0.4, 
                            alpha=0.7, color='purple', label='Adjusted')
    
    ax_recall.axhline(y=las_internals['original_recall_average'], 
                     color='blue', linestyle='--', alpha=0.7, label='Original Avg')
    ax_recall.axhline(y=las_internals['adjusted_recall_average'], 
                     color='red', linestyle='--', alpha=0.7, label='Adjusted Avg')
    
    ax_recall.set_xlabel('Reference Index')
    ax_recall.set_ylabel('Similarity Value')
    ax_recall.set_title('Recall LAS: Before vs After Penalty')
    ax_recall.set_ylim(0, 1.05)
    ax_recall.legend()
    
    # Summary text
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis('off')
    
    summary_text = f"""
SEMANTIC LOAD SHARING PENALTY SUMMARY

Algorithm: Bidirectional semantic adequacy penalty based on demand vs supply ratio
Gate Function: ψ_α(s) = max(0, (s-(1-α))/α) where α = supply/demand

Precision Direction: {precision_internals['total_sharing_groups']} sharing groups, {precision_internals['total_penalties_applied']} penalties applied
Recall Direction: {recall_internals['total_sharing_groups']} sharing groups, {recall_internals['total_penalties_applied']} penalties applied

Final LAS: {internals['metrics']['las']['f1']:.4f} (harmonic mean of adjusted precision and recall)
"""
    
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    fig.suptitle('Local Alignment Score (LAS) - Semantic Load Sharing Analysis', fontsize=16)
    fig.tight_layout()
    
    print("=" * 80)
    print()
    
    return fig