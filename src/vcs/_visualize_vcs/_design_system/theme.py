"""
VCS Professional Design System
==============================

Defines the visual design language for all VCS visualizations and reports.
Provides consistent colors, typography, spacing, and styling across the library.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from typing import Dict, Any, Tuple, List
import numpy as np

# =============================================================================
# COLOR PALETTE - Professional Academic Style
# =============================================================================

class VCSColors:
    """Professional color palette for VCS visualizations."""
    
    # Primary brand colors
    PRIMARY = "#2E86AB"      # Deep blue - main brand color
    PRIMARY_LIGHT = "#A8DADC" # Light blue - accents
    PRIMARY_DARK = "#1B5E7F"  # Dark blue - emphasis
    
    # Secondary colors
    SECONDARY = "#F1FAEE"     # Cream white - backgrounds
    ACCENT = "#E63946"        # Red - alerts/important
    ACCENT_LIGHT = "#F4A261"  # Orange - warnings
    
    # Metric-specific colors
    VCS_COLOR = "#2E86AB"     # Main VCS score - primary blue
    GAS_COLOR = "#457B9D"     # GAS - medium blue
    LAS_COLOR = "#1D3557"     # LAS - dark blue
    NAS_COLOR = "#F1FAEE"     # NAS - light cream (with dark border)
    
    # Precision/Recall colors
    PRECISION_COLOR = "#2E86AB"  # Blue
    RECALL_COLOR = "#E63946"     # Red
    
    # Status colors
    SUCCESS = "#06D6A0"       # Green
    WARNING = "#F4A261"       # Orange
    ERROR = "#E63946"         # Red
    INFO = "#A8DADC"          # Light blue
    
    # Neutral colors
    GRAY_DARK = "#2D3748"     # Dark text
    GRAY_MEDIUM = "#4A5568"   # Medium text
    GRAY_LIGHT = "#A0AEC0"    # Light text/borders
    GRAY_BG = "#F7FAFC"       # Light background
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    
    # Gradient colors for heatmaps
    GRADIENT_LOW = "#F7FAFC"
    GRADIENT_HIGH = "#2E86AB"
    
    @classmethod
    def get_metric_color(cls, metric_name: str) -> str:
        """Get the designated color for a specific metric."""
        color_map = {
            'vcs': cls.VCS_COLOR,
            'gas': cls.GAS_COLOR,
            'las': cls.LAS_COLOR,
            'nas': cls.NAS_COLOR,
            'precision': cls.PRECISION_COLOR,
            'recall': cls.RECALL_COLOR
        }
        return color_map.get(metric_name.lower(), cls.PRIMARY)
    
    @classmethod
    def get_qualitative_palette(cls, n_colors: int) -> List[str]:
        """Get a qualitative color palette with n colors."""
        base_colors = [
            cls.PRIMARY, cls.ACCENT, cls.ACCENT_LIGHT,
            cls.GAS_COLOR, cls.SUCCESS, cls.WARNING
        ]
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        
        # Generate additional colors if needed
        import matplotlib.cm as cm
        additional = cm.Set3(np.linspace(0, 1, n_colors - len(base_colors)))
        return base_colors + [matplotlib.colors.to_hex(c) for c in additional]

# =============================================================================
# TYPOGRAPHY SYSTEM
# =============================================================================

class VCSTypography:
    """Typography system for consistent text styling."""
    
    # Font families
    TITLE_FONT = "DejaVu Sans"        # Clean, professional
    BODY_FONT = "DejaVu Sans"         # Readable body text
    CODE_FONT = "DejaVu Sans Mono"    # Monospace for data/code
    
    # Font sizes
    TITLE_SIZE = 18                   # Main report titles
    SECTION_SIZE = 14                 # Section headers
    SUBSECTION_SIZE = 12              # Subsection headers
    BODY_SIZE = 10                    # Regular text
    CAPTION_SIZE = 8                  # Captions, page numbers
    SMALL_SIZE = 7                    # Fine print
    
    # Font weights
    BOLD = "bold"
    NORMAL = "normal"
    
    # Line spacing
    LINE_SPACING = 1.4

# =============================================================================
# SPACING AND LAYOUT
# =============================================================================

class VCSSpacing:
    """Consistent spacing system."""
    
    # Base unit for consistent spacing
    BASE_UNIT = 8  # pixels
    
    # Spacing scale
    XS = BASE_UNIT * 0.5      # 4px
    SM = BASE_UNIT * 1        # 8px
    MD = BASE_UNIT * 2        # 16px
    LG = BASE_UNIT * 3        # 24px
    XL = BASE_UNIT * 4        # 32px
    XXL = BASE_UNIT * 6       # 48px
    
    # Page margins
    PAGE_MARGIN = XL          # 32px
    SECTION_MARGIN = LG       # 24px
    CONTENT_MARGIN = MD       # 16px
    
    # Figure dimensions (inches)
    FIGURE_WIDTH = 15
    FIGURE_HEIGHT = 10
    SMALL_FIGURE_WIDTH = 12
    SMALL_FIGURE_HEIGHT = 8

# =============================================================================
# VISUAL ELEMENTS
# =============================================================================

class VCSVisualElements:
    """Reusable visual elements and styling."""
    
    # Border styles
    BORDER_WIDTH = 1.5
    BORDER_RADIUS = 4
    
    # Shadow/elevation
    SHADOW_COLOR = VCSColors.GRAY_LIGHT
    SHADOW_ALPHA = 0.3
    
    # Grid styles
    GRID_COLOR = VCSColors.GRAY_LIGHT
    GRID_ALPHA = 0.3
    GRID_WIDTH = 0.5
    
    @staticmethod
    def create_section_divider() -> patches.Rectangle:
        """Create a professional section divider."""
        return patches.Rectangle(
            (0, 0), 1, 0.002,
            facecolor=VCSColors.PRIMARY,
            alpha=0.8,
            transform=plt.gca().transAxes
        )
    
    @staticmethod
    def create_highlight_box(color: str = None) -> Dict[str, Any]:
        """Create styling for highlight boxes."""
        if color is None:
            color = VCSColors.INFO
        
        return {
            'boxstyle': 'round,pad=0.5',
            'facecolor': color,
            'alpha': 0.1,
            'edgecolor': color,
            'linewidth': VCSVisualElements.BORDER_WIDTH
        }
    
    @staticmethod
    def create_metric_card_style() -> Dict[str, Any]:
        """Create styling for metric cards."""
        return {
            'boxstyle': 'round,pad=0.6',
            'facecolor': VCSColors.WHITE,
            'edgecolor': VCSColors.GRAY_LIGHT,
            'linewidth': VCSVisualElements.BORDER_WIDTH,
            'alpha': 0.9
        }

# =============================================================================
# MATPLOTLIB THEME APPLICATION
# =============================================================================

def apply_vcs_theme():
    """Apply the VCS professional theme to matplotlib."""
    
    # Set the style parameters
    plt.style.use('default')  # Start with clean slate
    
    # Figure settings
    rcParams['figure.facecolor'] = VCSColors.WHITE
    rcParams['figure.edgecolor'] = VCSColors.WHITE
    rcParams['figure.figsize'] = (VCSSpacing.FIGURE_WIDTH, VCSSpacing.FIGURE_HEIGHT)
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.facecolor'] = VCSColors.WHITE
    rcParams['savefig.edgecolor'] = VCSColors.WHITE
    rcParams['savefig.bbox'] = 'tight'
    
    # Font settings
    rcParams['font.family'] = [VCSTypography.BODY_FONT]
    rcParams['font.size'] = VCSTypography.BODY_SIZE
    rcParams['font.weight'] = VCSTypography.NORMAL
    
    # Axes settings
    rcParams['axes.facecolor'] = VCSColors.WHITE
    rcParams['axes.edgecolor'] = VCSColors.GRAY_MEDIUM
    rcParams['axes.linewidth'] = VCSVisualElements.BORDER_WIDTH
    rcParams['axes.labelcolor'] = VCSColors.GRAY_DARK
    rcParams['axes.labelweight'] = VCSTypography.NORMAL
    rcParams['axes.labelsize'] = VCSTypography.BODY_SIZE
    rcParams['axes.titlesize'] = VCSTypography.SECTION_SIZE
    rcParams['axes.titleweight'] = VCSTypography.BOLD
    rcParams['axes.titlecolor'] = VCSColors.GRAY_DARK
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    
    # Grid settings
    rcParams['axes.grid'] = True
    rcParams['axes.grid.axis'] = 'y'
    rcParams['grid.color'] = VCSVisualElements.GRID_COLOR
    rcParams['grid.alpha'] = VCSVisualElements.GRID_ALPHA
    rcParams['grid.linewidth'] = VCSVisualElements.GRID_WIDTH
    
    # Tick settings
    rcParams['xtick.color'] = VCSColors.GRAY_MEDIUM
    rcParams['xtick.labelsize'] = VCSTypography.CAPTION_SIZE
    rcParams['ytick.color'] = VCSColors.GRAY_MEDIUM
    rcParams['ytick.labelsize'] = VCSTypography.CAPTION_SIZE
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    
    # Legend settings
    rcParams['legend.frameon'] = True
    rcParams['legend.facecolor'] = VCSColors.WHITE
    rcParams['legend.edgecolor'] = VCSColors.GRAY_LIGHT
    rcParams['legend.fontsize'] = VCSTypography.CAPTION_SIZE
    rcParams['legend.title_fontsize'] = VCSTypography.BODY_SIZE
    
    # Line and marker settings
    rcParams['lines.linewidth'] = 2.0
    rcParams['lines.markersize'] = 6
    
    # Text settings
    rcParams['text.color'] = VCSColors.GRAY_DARK

def create_professional_figure(
    figsize: Tuple[float, float] = None,
    title: str = "",
    subtitle: str = "",
    show_branding: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a professionally styled figure with consistent branding."""
    
    if figsize is None:
        figsize = (VCSSpacing.FIGURE_WIDTH, VCSSpacing.FIGURE_HEIGHT)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply consistent styling
    ax.set_facecolor(VCSColors.WHITE)
    
    # Add title with consistent styling
    if title:
        if subtitle:
            full_title = f"{title}\n{subtitle}"
            fig.suptitle(full_title, 
                        fontsize=VCSTypography.TITLE_SIZE,
                        fontweight=VCSTypography.BOLD,
                        color=VCSColors.GRAY_DARK,
                        y=0.95)
        else:
            fig.suptitle(title,
                        fontsize=VCSTypography.TITLE_SIZE,
                        fontweight=VCSTypography.BOLD,
                        color=VCSColors.GRAY_DARK,
                        y=0.95)
    
    # Add subtle branding
    if show_branding:
        fig.text(0.99, 0.01, "VCS Analysis Report", 
                ha='right', va='bottom',
                fontsize=VCSTypography.SMALL_SIZE,
                color=VCSColors.GRAY_LIGHT,
                style='italic')
    
    return fig, ax

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def style_metric_visualization(ax: plt.Axes, metric_name: str, value: float, 
                             range_min: float = 0.0, range_max: float = 1.0):
    """Apply consistent styling to metric visualizations."""
    
    # Get metric color
    color = VCSColors.get_metric_color(metric_name)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(VCSColors.GRAY_LIGHT)
    ax.spines['bottom'].set_color(VCSColors.GRAY_LIGHT)
    
    # Add value annotation with styling
    if range_min <= value <= range_max:
        bbox_props = VCSVisualElements.create_metric_card_style()
        ax.annotate(f'{metric_name.upper()}: {value:.4f}',
                   xy=(0.5, 0.9), xycoords='axes fraction',
                   ha='center', va='center',
                   fontsize=VCSTypography.SECTION_SIZE,
                   fontweight=VCSTypography.BOLD,
                   color=color,
                   bbox=bbox_props)

def add_professional_legend(ax: plt.Axes, handles=None, labels=None, 
                          loc='best', **kwargs):
    """Add a professionally styled legend."""
    
    default_props = {
        'frameon': True,
        'facecolor': VCSColors.WHITE,
        'edgecolor': VCSColors.GRAY_LIGHT,
        'fontsize': VCSTypography.CAPTION_SIZE,
        'title_fontsize': VCSTypography.BODY_SIZE,
        'loc': loc
    }
    default_props.update(kwargs)
    
    if handles and labels:
        legend = ax.legend(handles, labels, **default_props)
    else:
        legend = ax.legend(**default_props)
    
    # Style the legend frame
    legend.get_frame().set_linewidth(VCSVisualElements.BORDER_WIDTH)
    legend.get_frame().set_alpha(0.95)
    
    return legend