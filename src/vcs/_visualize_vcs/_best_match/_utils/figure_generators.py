import matplotlib.pyplot as plt
from typing import Dict, Any
from .precision_details import create_precision_details_section
from .recall_details import create_recall_details_section
from .summary_table import create_summary_table_section


def create_precision_details_figure(internals: Dict[str, Any]) -> plt.Figure:

    # Create a single-axis figure optimized for precision details display
    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    ax.axis('off')
    
    # Extract the data needed for precision details
    best_match_info = internals.get('best_match', {})
    precision_match_details = best_match_info.get('precision', {})
    
    # Get configuration parameters for context calculations
    context_cutoff_value = internals['config'].get('context_cutoff_value', 0.6)
    context_window_control = internals['config'].get('context_window_control', 4.0)
    
    # Use existing utility function to populate the figure
    create_precision_details_section(
        ax, precision_match_details, context_cutoff_value, context_window_control
    )
    
    # Set descriptive title for the standalone figure
    fig.suptitle('Precision Matching Analysis (Generation → Reference)', 
                 fontsize=16, y=0.98)
    
    # Use subplots_adjust for better control with text-heavy content
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    
    return fig


def create_recall_details_figure(internals: Dict[str, Any]) -> plt.Figure:

    # Create a single-axis figure optimized for recall details display
    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    ax.axis('off')
    
    # Extract the data needed for recall details
    best_match_info = internals.get('best_match', {})
    recall_match_details = best_match_info.get('recall', {})
    
    # Get configuration parameters for context calculations
    context_cutoff_value = internals['config'].get('context_cutoff_value', 0.6)
    context_window_control = internals['config'].get('context_window_control', 4.0)
    
    # Use existing utility function to populate the figure
    create_recall_details_section(
        ax, recall_match_details, context_cutoff_value, context_window_control
    )
    
    # Set descriptive title for the standalone figure
    fig.suptitle('Recall Matching Analysis (Reference → Generation)', 
                 fontsize=16, y=0.98)
    
    # Use subplots_adjust for better control with text-heavy content
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    
    return fig


def create_summary_table_figure(internals: Dict[str, Any]) -> plt.Figure:

    # Create a single-axis figure optimized for table display
    fig = plt.figure(figsize=(20, 12))
    ax = plt.gca()
    ax.axis('off')
    
    # Use existing utility function to create the comprehensive summary
    create_summary_table_section(ax, internals)
    
    # Set descriptive title for the standalone figure
    fig.suptitle('Complete Matching Summary Table', fontsize=16, y=0.98)
    
    # Use subplots_adjust for optimal table layout
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    
    return fig