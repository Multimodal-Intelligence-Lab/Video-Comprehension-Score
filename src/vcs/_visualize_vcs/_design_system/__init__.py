"""
VCS Design System
================

Professional visual design system for VCS library visualizations and reports.
"""

from .theme import (
    VCSColors,
    VCSTypography, 
    VCSSpacing,
    VCSVisualElements,
    apply_vcs_theme,
    create_professional_figure,
    style_metric_visualization,
    add_professional_legend
)

__all__ = [
    'VCSColors',
    'VCSTypography',
    'VCSSpacing', 
    'VCSVisualElements',
    'apply_vcs_theme',
    'create_professional_figure',
    'style_metric_visualization',
    'add_professional_legend'
]