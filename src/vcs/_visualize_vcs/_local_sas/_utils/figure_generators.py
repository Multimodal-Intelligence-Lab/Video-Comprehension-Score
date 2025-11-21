import matplotlib.pyplot as plt
from typing import Dict, Any

def create_precision_load_sharing_figure(internals: Dict[str, Any]) -> plt.Figure:
    """Create precision load sharing figure matching best match style."""
    from .precision_load_sharing import create_precision_load_sharing_section

    fig, ax = plt.subplots(figsize=(15, 10))

    sas_internals = internals['metrics']['sas']
    precision_internals = sas_internals.get('precision_internals', {})
    
    create_precision_load_sharing_section(ax, precision_internals)
    
    return fig

def create_recall_load_sharing_figure(internals: Dict[str, Any]) -> plt.Figure:
    """Create recall load sharing figure matching best match style."""
    from .recall_load_sharing import create_recall_load_sharing_section

    fig, ax = plt.subplots(figsize=(15, 10))

    sas_internals = internals['metrics']['sas']
    recall_internals = sas_internals.get('recall_internals', {})
    
    create_recall_load_sharing_section(ax, recall_internals)
    
    return fig