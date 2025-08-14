import matplotlib.pyplot as plt
from typing import Dict, Any

def visualize_metrics_summary(internals: Dict[str, Any]) -> plt.Figure:
    """Create a comprehensive overview of all VCS metrics and their components.
    
    Displays all computed metrics in a clear horizontal bar chart, organized by
    metric type. Essential for getting a quick overview of the analysis results
    and understanding the relative contributions of different components to the
    final VCS score.
    
    Parameters
    ----------
    internals : dict
        The internals dictionary returned by ``compute_vcs_score`` with 
        ``return_internals=True``. Must contain complete 'metrics' section.
    
    Returns
    -------
    matplotlib.figure.Figure
        A figure showing all metrics as a horizontal bar chart with color coding
        by metric type and visual separators between metric families.
    
    Examples
    --------
    **Basic Usage:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text",
            generated_text="Your generated text",
            segmenter_fn=your_segmenter,
            embedding_fn_las=your_embedder,
            return_internals=True,
            return_all_metrics=True
        )
        fig = visualize_metrics_summary(result['internals'])
        fig.show()
    
    See Also
    --------
    compute_vcs_score : Core function that generates the metrics displayed here
    visualize_config : See parameters that produced these results
    create_vcs_pdf_report : Generate comprehensive PDF report including this summary
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = {}
    
    metrics['VCS'] = internals['metrics']['vcs']['value']
    
    metrics['GAS'] = internals['metrics']['gas']['value']
    
    las_metrics = internals['metrics']['las']
    metrics['LAS'] = las_metrics['f1']
    metrics['LAS Precision'] = las_metrics['precision']
    metrics['LAS Recall'] = las_metrics['recall']
    
    metrics['SAS'] = internals['metrics']['vcs']['gas_las_scaled']
    
    metrics['NAS'] = internals['metrics']['nas']['regularized_nas']
    
    nas_d = internals['metrics']['nas']['nas_d']
    metrics['NAS-D'] = nas_d['f1']
    metrics['NAS-D Precision'] = nas_d['precision']['value']
    metrics['NAS-D Recall'] = nas_d['recall']['value']
    
    nas_l = internals['metrics']['nas']['nas_l']
    metrics['NAS-L'] = nas_l['f1']
    metrics['NAS-L Precision'] = nas_l['precision']['value']
    metrics['NAS-L Recall'] = nas_l['recall']['value']
    
    metrics['NAS F1'] = internals['metrics']['nas']['nas_f1']
    metrics['Length Regularizer'] = internals['metrics']['nas']['regularizer']['value']
    
    order = [
        'VCS',
        'GAS',
        'LAS',
        'LAS Precision',
        'LAS Recall',
        'SAS',
        'NAS',
        'Length Regularizer',
        'NAS F1',
        'NAS-D',
        'NAS-D Precision',
        'NAS-D Recall',
        'NAS-L',
        'NAS-L Precision',
        'NAS-L Recall'
    ]
    
    y_pos = 0
    y_ticks = []
    y_labels = []
    
    colors = {
        'VCS': 'gold',
        'GAS': 'skyblue',
        'LAS': 'lightgreen',
        'SAS': 'lightcyan',
        'NAS': 'salmon',
        'NAS-D': 'plum',
        'NAS-L': 'orchid',
        'Length Regularizer': 'peachpuff',
    }
    
    def get_color(metric_name):
        for key in colors:
            if metric_name.startswith(key):
                return colors[key]
        return 'lightgray'
    
    for i, metric_name in enumerate(order):
        if metric_name in metrics:
            value = metrics[metric_name]
            ax.barh(i, value, color=get_color(metric_name), alpha=0.7)
            ax.text(value + 0.01, i, f"{value:.4f}", va='center', fontsize=9)
            y_labels.append(metric_name)
            y_ticks.append(i)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    ax.axhline(y=5.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=8.5, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Metric Value')
    ax.set_title('VCS Metrics Summary')
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    return fig