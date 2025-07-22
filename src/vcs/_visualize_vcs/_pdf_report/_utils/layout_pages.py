import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple, Dict, Any


def create_title_page() -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Add a subtle background gradient for modern look
    _add_background_gradient(ax)
    
    # Add title with modern styling - corrected to Video Comprehension Score
    _add_main_title(ax)
    
    # Add a decorative line with gradient
    _add_decorative_line(ax)
    
    # Add generation date and version info
    _add_metadata(ax)
    
    # Add footer
    _add_footer(ax)
    
    # Add decorative corner elements
    _add_corner_decorations(ax)
    
    return fig


def create_metrics_page(
    vcs_score: float, 
    gas_score: float, 
    las_score: float, 
    nas_score: float, 
    ref_len: int, 
    gen_len: int
) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Add a title with proper spacing and styling
    _add_page_title(ax, "VCS Analysis Summary")
    
    # Add a header for the metrics section
    ax.text(0.1, 0.82, "Key Metrics", fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Create metrics table
    _create_metrics_table(ax, vcs_score, gas_score, las_score, nas_score, ref_len, gen_len)
    
    # Add explanation section
    _add_metrics_explanation(ax)
    
    # Add decorative boxes
    _add_decorative_boxes(ax)
    
    # Add page number to footer
    fig.text(0.5, 0.02, "Page 2", ha='center', va='bottom', fontsize=9, color='#555555')
    
    return fig


def create_toc(sections_with_pages: List[Tuple]) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Add title
    _add_page_title(ax, "Table of Contents")
    
    # Calculate layout parameters
    total_entries = sum(len(pages) for section, pages in sections_with_pages)
    line_height = 0.75 / (total_entries + len(sections_with_pages) * 2)  # Height per line with spacing
    
    # Draw sections and entries
    _draw_toc_content(ax, sections_with_pages, line_height)
    
    # Add page number to footer
    page_num = 3  # Typically the TOC is page 3 after title and metrics
    fig.text(0.5, 0.02, f"Page {page_num}", ha='center', va='bottom', fontsize=9, color='#555555')
    
    return fig


def generate_front_matter(
    pdf: PdfPages, 
    key_metrics: Dict[str, Any], 
    layout_config: Dict[str, Any], 
    toc_data: List[Tuple]
) -> int:
    """Generate front matter pages (title, metrics, TOC) and return current page number."""
    current_page = 2
    
    # First page: Title page (always included)
    title_page = create_title_page()
    pdf.savefig(title_page, bbox_inches='tight')
    plt.close(title_page)
    
    # Optional metrics summary page
    if layout_config['include_metrics_page']:
        metrics_page = create_metrics_page(**key_metrics)
        pdf.savefig(metrics_page, bbox_inches='tight')
        plt.close(metrics_page)
        current_page += 1
    
    # Optional TOC page
    if layout_config['include_toc']:
        toc_page = create_toc(toc_data)
        pdf.savefig(toc_page, bbox_inches='tight')
        plt.close(toc_page)
        current_page += 1
    
    return current_page


def _add_background_gradient(ax) -> None:
    gradient = np.linspace(0, 1, 100)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', extent=[0, 1, 0, 1], origin='lower',
              cmap=plt.cm.Blues, alpha=0.15, transform=ax.transAxes)


def _add_main_title(ax) -> None:
    title_text = "Video Comprehension Score\n(VCS)\nAnalysis Report"
    ax.text(0.5, 0.6, title_text, 
            ha='center', va='center', 
            fontsize=28, fontweight='bold', 
            linespacing=1.5,
            color='#2c3e50')  # Dark blue for professional look


def _add_decorative_line(ax) -> None:
    for x in np.linspace(0.2, 0.8, 100):
        alpha = 1 - (abs(x - 0.5) * 2) * 0.5  # Gradient effect
        ax.axhline(y=0.45, xmin=x, xmax=x+0.005, color='#3498db', alpha=alpha, linewidth=3)


def _add_metadata(ax) -> None:
    date_text = f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}"
    ax.text(0.5, 0.35, date_text, ha='center', va='center', 
            fontsize=14, style='italic', color='#7f8c8d')
    
    version_text = "VCS Library v1.0"
    ax.text(0.5, 0.30, version_text, ha='center', va='center',
            fontsize=12, color='#7f8c8d')


def _add_footer(ax) -> None:
    ax.text(0.5, 0.05, "VCS Library - Automated Text Coherence Analysis", 
            ha='center', va='center', fontsize=10, color='#95a5a6')


def _add_corner_decorations(ax) -> None:
    corner_size = 0.05
    linewidth = 2
    color = '#3498db'
    
    # Top left
    ax.plot([0.05, 0.05, 0.05+corner_size], [0.9, 0.95, 0.95], 
            color=color, linewidth=linewidth)
    # Top right
    ax.plot([0.95-corner_size, 0.95, 0.95], [0.95, 0.95, 0.9], 
            color=color, linewidth=linewidth)
    # Bottom left
    ax.plot([0.05, 0.05, 0.05+corner_size], [0.1, 0.05, 0.05], 
            color=color, linewidth=linewidth)
    # Bottom right
    ax.plot([0.95-corner_size, 0.95, 0.95], [0.05, 0.05, 0.1], 
            color=color, linewidth=linewidth)


def _add_page_title(ax, title: str) -> None:
    """Add a page title with consistent styling."""
    ax.text(0.5, 0.92, title, 
            ha='center', va='top', fontsize=20, fontweight='bold', color='#2c3e50')
    
    # Add a subtle blue bar under the title
    ax.axhline(y=0.88, xmin=0.25, xmax=0.75, color='#3498db', linewidth=2, alpha=0.7)


def _create_metrics_table(ax, vcs_score: float, gas_score: float, las_score: float, 
                         nas_score: float, ref_len: int, gen_len: int) -> None:
    """Create and style the metrics table."""
    table_data = [
        ["VCS Score:", f"{vcs_score:.4f}"],
        ["GAS Score:", f"{gas_score:.4f}"],
        ["LAS Score:", f"{las_score:.4f}"],
        ["NAS Score:", f"{nas_score:.4f}"],
        ["", ""],  # Spacer row
        ["Reference Length:", f"{ref_len} segments"],
        ["Generated Length:", f"{gen_len} segments"]
    ]
    
    # Create and style the table
    table = plt.table(cellText=table_data, colWidths=[0.4, 0.3], 
                      cellLoc='left', bbox=[0.1, 0.55, 0.8, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)


def _add_metrics_explanation(ax) -> None:
    ax.text(0.1, 0.48, "Metrics Explanation", fontsize=16, fontweight='bold', color='#2c3e50')
    
    explanation_text = [
        ("VCS (Video Comprehension Score):", 
         "Overall narrative similarity between reference and generated text."),
        ("GAS (Global Alignment Score):", 
         "Measures semantic similarity at the full-text level."),
        ("LAS (Local Alignment Score):", 
         "Measures segment-by-segment semantic similarity."),
        ("NAS (Narrative Alignment Score):", 
         "Measures how well the narrative flow is preserved.")
    ]
    
    # Position and style the explanation text
    y_pos = 0.43
    for title, desc in explanation_text:
        # Title in bold
        ax.text(0.12, y_pos, title, fontsize=12, fontweight='bold', color='#2c3e50')
        # Description indented and in regular weight
        ax.text(0.15, y_pos - 0.03, desc, fontsize=11, color='#34495e')
        y_pos -= 0.08


def _add_decorative_boxes(ax) -> None:
    metrics_rect = plt.Rectangle((0.08, 0.5), 0.84, 0.35, 
                                 fill=True, facecolor='#ecf0f1', 
                                 edgecolor='#bdc3c7', linewidth=1, alpha=0.5)
    metrics_rect.set_zorder(-1)
    ax.add_patch(metrics_rect)
    
    explanation_rect = plt.Rectangle((0.08, 0.1), 0.84, 0.35, 
                                     fill=True, facecolor='#ecf0f1', 
                                     edgecolor='#bdc3c7', linewidth=1, alpha=0.5)
    explanation_rect.set_zorder(-1)
    ax.add_patch(explanation_rect)


def _draw_toc_content(ax, sections_with_pages: List[Tuple], line_height: float) -> None:
    y_pos = 0.85  # Starting position
    
    # Draw each section
    for section, pages in sections_with_pages:
        # Draw section heading
        ax.text(0.1, y_pos, section, fontweight='bold', fontsize=14, color='#2c3e50')
        y_pos -= line_height * 1.5  # Extra space after section heading
        
        # Draw divider line for section
        ax.axhline(y=y_pos + line_height * 0.5, xmin=0.1, xmax=0.9, 
                   color='#bdc3c7', alpha=0.7, linewidth=1)
        y_pos -= line_height * 0.5  # Space after divider
        
        # Draw entries for this section
        for title, page_num in pages:
            # Draw the title on the left
            ax.text(0.15, y_pos, title, fontsize=11, color='#34495e')
            
            # Draw page number on the right
            ax.text(0.85, y_pos, str(page_num), ha='right', fontsize=11, color='#34495e')
            
            y_pos -= line_height
        
        # Extra space after section
        y_pos -= line_height