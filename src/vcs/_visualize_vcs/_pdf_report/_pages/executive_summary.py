"""
Executive Summary Page Generator
===============================

Creates a professional executive summary page that provides key insights
and high-level overview of VCS analysis results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple
import numpy as np

from ..._design_system import (
    VCSColors, VCSTypography, VCSSpacing, VCSVisualElements,
    create_professional_figure, style_metric_visualization
)

def create_executive_summary_page(internals: Dict[str, Any]) -> plt.Figure:
    """Create a professional executive summary page."""
    
    # Extract key metrics
    metrics = _extract_key_insights(internals)
    
    # Create figure with professional styling
    fig, ax = create_professional_figure(
        title="VCS Analysis Executive Summary",
        subtitle="Key Insights & Performance Overview",
        figsize=(VCSSpacing.FIGURE_WIDTH, VCSSpacing.FIGURE_HEIGHT)
    )
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 1. Key Metrics Dashboard (top section)
    _add_metrics_dashboard(ax, metrics)
    
    # 2. Performance Assessment (middle section)  
    _add_performance_assessment(ax, metrics)
    
    # 3. Key Findings (bottom section)
    _add_key_findings(ax, metrics)
    
    # 4. Recommendations (if applicable)
    _add_recommendations(ax, metrics)
    
    plt.tight_layout()
    return fig

def _extract_key_insights(internals: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics and insights from internals."""
    
    # Core metrics
    vcs_score = internals['metrics']['vcs']['value']
    gas_score = internals['metrics']['gas']['value'] 
    las_score = internals['metrics']['las']['f1']
    nas_score = internals['metrics']['nas']['regularized_nas']
    
    # Text statistics
    ref_len = internals['texts']['reference_length']
    gen_len = internals['texts']['generated_length']
    ref_chunks = len(internals['texts']['reference_chunks'])
    gen_chunks = len(internals['texts']['generated_chunks'])
    
    # Alignment statistics
    precision_matches = len(internals['alignment']['precision']['matches'])
    recall_matches = len(internals['alignment']['recall']['matches'])
    
    # Performance assessment
    overall_performance = _assess_overall_performance(vcs_score)
    component_balance = _assess_component_balance(gas_score, las_score, nas_score)
    alignment_quality = _assess_alignment_quality(precision_matches, recall_matches, ref_chunks, gen_chunks)
    
    # Key insights
    insights = _generate_key_insights(vcs_score, gas_score, las_score, nas_score, 
                                    ref_len, gen_len, precision_matches, recall_matches)
    
    return {
        'metrics': {
            'vcs': vcs_score,
            'gas': gas_score, 
            'las': las_score,
            'nas': nas_score
        },
        'text_stats': {
            'ref_length': ref_len,
            'gen_length': gen_len,
            'ref_chunks': ref_chunks,
            'gen_chunks': gen_chunks
        },
        'alignment_stats': {
            'precision_matches': precision_matches,
            'recall_matches': recall_matches
        },
        'assessments': {
            'overall_performance': overall_performance,
            'component_balance': component_balance,
            'alignment_quality': alignment_quality
        },
        'insights': insights
    }

def _add_metrics_dashboard(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """Add a professional metrics dashboard at the top."""
    
    # Dashboard background
    dashboard_bg = patches.Rectangle(
        (0.5, 7.5), 9, 2,
        facecolor=VCSColors.GRAY_BG,
        edgecolor=VCSColors.GRAY_LIGHT,
        linewidth=VCSVisualElements.BORDER_WIDTH,
        alpha=0.8
    )
    ax.add_patch(dashboard_bg)
    
    # Dashboard title
    ax.text(5, 9.2, "PERFORMANCE METRICS", 
           ha='center', va='center',
           fontsize=VCSTypography.SECTION_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.GRAY_DARK)
    
    # Metric cards
    metric_data = [
        ('VCS', metrics['metrics']['vcs'], VCSColors.VCS_COLOR),
        ('GAS', metrics['metrics']['gas'], VCSColors.GAS_COLOR),
        ('LAS', metrics['metrics']['las'], VCSColors.LAS_COLOR),
        ('NAS', metrics['metrics']['nas'], VCSColors.PRIMARY)
    ]
    
    x_positions = [1.5, 3.5, 5.5, 7.5]
    
    for i, (name, value, color) in enumerate(metric_data):
        x = x_positions[i]
        
        # Metric card background
        card_bg = patches.Rectangle(
            (x-0.4, 7.8), 0.8, 1.4,
            facecolor=VCSColors.WHITE,
            edgecolor=color,
            linewidth=2,
            alpha=0.95
        )
        ax.add_patch(card_bg)
        
        # Metric value (large)
        ax.text(x, 8.7, f"{value:.3f}",
               ha='center', va='center',
               fontsize=16, fontweight=VCSTypography.BOLD,
               color=color)
        
        # Metric name
        ax.text(x, 8.1, name,
               ha='center', va='center', 
               fontsize=VCSTypography.BODY_SIZE,
               color=VCSColors.GRAY_DARK)

def _add_performance_assessment(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """Add performance assessment section."""
    
    # Section title
    ax.text(0.5, 6.8, "PERFORMANCE ASSESSMENT",
           fontsize=VCSTypography.SECTION_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.GRAY_DARK)
    
    # Assessment content
    assessments = metrics['assessments']
    
    y_pos = 6.3
    
    # Overall performance
    overall = assessments['overall_performance']
    ax.text(0.5, y_pos, "Overall Performance:",
           fontsize=VCSTypography.BODY_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.GRAY_DARK)
    
    perf_color = _get_performance_color(overall['level'])
    ax.text(3.0, y_pos, f"{overall['level']} ({overall['description']})",
           fontsize=VCSTypography.BODY_SIZE,
           color=perf_color)
    
    # Component balance
    y_pos -= 0.4
    balance = assessments['component_balance']
    ax.text(0.5, y_pos, "Component Balance:",
           fontsize=VCSTypography.BODY_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.GRAY_DARK)
    
    ax.text(3.0, y_pos, balance['description'],
           fontsize=VCSTypography.BODY_SIZE,
           color=VCSColors.GRAY_DARK)
    
    # Alignment quality
    y_pos -= 0.4
    alignment = assessments['alignment_quality']
    ax.text(0.5, y_pos, "Alignment Quality:",
           fontsize=VCSTypography.BODY_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.GRAY_DARK)
    
    align_color = _get_performance_color(alignment['level'])
    ax.text(3.0, y_pos, f"{alignment['level']} - {alignment['description']}",
           fontsize=VCSTypography.BODY_SIZE,
           color=align_color)

def _add_key_findings(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """Add key findings section."""
    
    # Section title with visual separator
    ax.text(0.5, 4.5, "KEY FINDINGS & INSIGHTS",
           fontsize=VCSTypography.SECTION_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.GRAY_DARK)
    
    # Add separator line
    separator = patches.Rectangle(
        (0.5, 4.3), 9, 0.02,
        facecolor=VCSColors.PRIMARY,
        alpha=0.6
    )
    ax.add_patch(separator)
    
    # Key insights
    insights = metrics['insights']
    y_pos = 3.9
    
    for i, insight in enumerate(insights[:5]):  # Show top 5 insights
        # Bullet point
        ax.text(0.7, y_pos, "•",
               fontsize=VCSTypography.BODY_SIZE,
               color=VCSColors.PRIMARY,
               fontweight=VCSTypography.BOLD)
        
        # Insight text
        ax.text(1.0, y_pos, insight,
               fontsize=VCSTypography.BODY_SIZE,
               color=VCSColors.GRAY_DARK,
               wrap=True)
        
        y_pos -= 0.35

def _add_recommendations(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """Add recommendations section if applicable."""
    
    recommendations = _generate_recommendations(metrics)
    
    if not recommendations:
        return
    
    # Section title
    ax.text(0.5, 1.8, "RECOMMENDATIONS",
           fontsize=VCSTypography.SECTION_SIZE,
           fontweight=VCSTypography.BOLD,
           color=VCSColors.ACCENT)
    
    # Recommendations box
    rec_bg = patches.Rectangle(
        (0.5, 0.3), 9, 1.3,
        facecolor=VCSColors.ACCENT,
        alpha=0.1,
        edgecolor=VCSColors.ACCENT,
        linewidth=VCSVisualElements.BORDER_WIDTH
    )
    ax.add_patch(rec_bg)
    
    y_pos = 1.4
    for rec in recommendations[:3]:  # Show top 3 recommendations
        ax.text(0.7, y_pos, "→",
               fontsize=VCSTypography.BODY_SIZE,
               color=VCSColors.ACCENT,
               fontweight=VCSTypography.BOLD)
        
        ax.text(1.0, y_pos, rec,
               fontsize=VCSTypography.BODY_SIZE,
               color=VCSColors.GRAY_DARK)
        
        y_pos -= 0.3

def _assess_overall_performance(vcs_score: float) -> Dict[str, str]:
    """Assess overall performance based on VCS score."""
    if vcs_score >= 0.8:
        return {'level': 'Excellent', 'description': 'High semantic similarity achieved'}
    elif vcs_score >= 0.6:
        return {'level': 'Good', 'description': 'Satisfactory semantic alignment'}
    elif vcs_score >= 0.4:
        return {'level': 'Fair', 'description': 'Moderate semantic correspondence'}
    else:
        return {'level': 'Poor', 'description': 'Limited semantic similarity'}

def _assess_component_balance(gas: float, las: float, nas: float) -> Dict[str, str]:
    """Assess balance between VCS components."""
    scores = [gas, las, nas]
    score_range = max(scores) - min(scores)
    
    if score_range <= 0.2:
        return {'description': 'Well-balanced across all components'}
    elif score_range <= 0.4:
        return {'description': 'Moderate variance between components'}
    else:
        dominant = ['Global', 'Local', 'Narrative'][scores.index(max(scores))]
        return {'description': f'{dominant} alignment dominates performance'}

def _assess_alignment_quality(precision_matches: int, recall_matches: int, 
                            ref_chunks: int, gen_chunks: int) -> Dict[str, str]:
    """Assess alignment quality based on match statistics."""
    precision_rate = precision_matches / gen_chunks if gen_chunks > 0 else 0
    recall_rate = recall_matches / ref_chunks if ref_chunks > 0 else 0
    
    avg_rate = (precision_rate + recall_rate) / 2
    
    if avg_rate >= 0.8:
        return {'level': 'High', 'description': 'Strong bidirectional alignment'}
    elif avg_rate >= 0.6:
        return {'level': 'Good', 'description': 'Solid alignment coverage'}
    elif avg_rate >= 0.4:
        return {'level': 'Fair', 'description': 'Moderate alignment gaps'}
    else:
        return {'level': 'Low', 'description': 'Significant alignment issues'}

def _generate_key_insights(vcs: float, gas: float, las: float, nas: float,
                         ref_len: int, gen_len: int, precision_matches: int, 
                         recall_matches: int) -> List[str]:
    """Generate key insights based on metrics."""
    insights = []
    
    # VCS insight
    if vcs >= 0.7:
        insights.append(f"Strong overall semantic similarity (VCS: {vcs:.3f})")
    else:
        insights.append(f"Opportunity for improved semantic alignment (VCS: {vcs:.3f})")
    
    # Component analysis
    component_scores = [('Global', gas), ('Local', las), ('Narrative', nas)]
    component_scores.sort(key=lambda x: x[1], reverse=True)
    
    strongest = component_scores[0]
    insights.append(f"{strongest[0]} alignment is strongest component ({strongest[1]:.3f})")
    
    # Length analysis
    length_ratio = gen_len / ref_len if ref_len > 0 else 0
    if length_ratio < 0.8:
        insights.append("Generated text is significantly shorter than reference")
    elif length_ratio > 1.2:
        insights.append("Generated text is notably longer than reference")
    else:
        insights.append("Generated and reference texts have similar lengths")
    
    # Match coverage
    if precision_matches > 0 and recall_matches > 0:
        insights.append(f"Bidirectional alignment achieved ({precision_matches} precision, {recall_matches} recall matches)")
    else:
        insights.append("Limited bidirectional alignment detected")
    
    return insights

def _generate_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []
    
    vcs = metrics['metrics']['vcs']
    gas = metrics['metrics']['gas']
    las = metrics['metrics']['las']
    nas = metrics['metrics']['nas']
    
    if vcs < 0.6:
        recommendations.append("Consider improving semantic alignment through better content matching")
    
    if gas < 0.5:
        recommendations.append("Focus on global content coverage and thematic consistency")
    
    if las < 0.5:
        recommendations.append("Enhance local sentence-level alignment and coherence")
    
    if nas < 0.5:
        recommendations.append("Improve narrative flow and sequential structure alignment")
    
    return recommendations

def _get_performance_color(level: str) -> str:
    """Get color based on performance level."""
    color_map = {
        'Excellent': VCSColors.SUCCESS,
        'High': VCSColors.SUCCESS,
        'Good': VCSColors.INFO,
        'Fair': VCSColors.WARNING,
        'Poor': VCSColors.ERROR,
        'Low': VCSColors.ERROR
    }
    return color_map.get(level, VCSColors.GRAY_DARK)