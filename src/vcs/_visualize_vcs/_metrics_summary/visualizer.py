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
            embedding_fn_global_sas=your_embedder,
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

    metrics['Global_SAS'] = internals['metrics']['global_sas']['value']

    local_sas_metrics = internals['metrics']['local_sas']
    metrics['Local_SAS'] = local_sas_metrics['f1']
    metrics['Local_SAS Precision'] = local_sas_metrics['precision']
    metrics['Local_SAS Recall'] = local_sas_metrics['recall']

    metrics['SAS'] = internals['metrics']['sas']['value']

    metrics['NAS'] = internals['metrics']['nas']['value']

    global_nas = internals['metrics']['global_nas']
    metrics['Global NAS'] = global_nas['f1']
    metrics['Global NAS Precision'] = global_nas['precision']['value']
    metrics['Global NAS Recall'] = global_nas['recall']['value']

    local_nas = internals['metrics']['local_nas']
    metrics['Local NAS'] = local_nas['f1']
    metrics['Local NAS Precision'] = local_nas['precision']['value']
    metrics['Local NAS Recall'] = local_nas['recall']['value']


    order = [
        'VCS',
        'Global_SAS',
        'Local_SAS',
        'Local_SAS Precision',
        'Local_SAS Recall',
        'SAS',
        'NAS',
        'Global NAS',
        'Global NAS Precision',
        'Global NAS Recall',
        'Local NAS',
        'Local NAS Precision',
        'Local NAS Recall'
    ]

    y_pos = 0
    y_ticks = []
    y_labels = []

    colors = {
        'VCS': 'gold',
        'Global_SAS': 'skyblue',
        'Local_SAS': 'lightgreen',
        'SAS': 'lightcyan',
        'NAS': 'salmon',
        'Global NAS': 'plum',
        'Local NAS': 'orchid',
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