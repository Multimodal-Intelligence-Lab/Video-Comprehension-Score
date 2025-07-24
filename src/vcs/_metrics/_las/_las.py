import numpy as np
from typing import List, Dict
from ..._utils import _calculate_f1

def _compute_las_metrics(
    precision_sim_values: np.ndarray,
    recall_sim_values: np.ndarray
) -> Dict[str, float]:

    precision_las = float(np.mean(precision_sim_values)) if precision_sim_values.size else 0.0
    recall_las = float(np.mean(recall_sim_values)) if recall_sim_values.size else 0.0
    f1_las = _calculate_f1(precision_las, recall_las)
    
    return {
        "Precision LAS": precision_las,
        "Recall LAS": recall_las,
        "LAS": f1_las 
    }