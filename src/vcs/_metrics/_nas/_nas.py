import numpy as np
from typing import List, Tuple, Dict, Any
from ..._utils import _calculate_f1

from ._nas_components._global_nas._global_nas import _calculate_global_nas
from ._nas_components._local_nas._local_nas import _calculate_local_nas

def _compute_nas_metrics(
    sim_matrix: np.ndarray,
    ref_len: int,
    gen_len: int,
    precision_matches: List[Tuple],
    precision_indices: np.ndarray,
    precision_sim_values: np.ndarray,
    recall_matches: List[Tuple],
    recall_indices: np.ndarray,
    recall_sim_values: np.ndarray,
    prec_map_windows: List[Tuple[int, int]],
    rec_map_windows: List[Tuple[int, int]],
    ref_chunks: List[str],
    gen_chunks: List[str],
    Rn: int = 0
) -> Tuple[Dict[str, float], Dict[str, Any]]:

    prec_nas, prec_nas_internals = _calculate_global_nas(
        precision_indices, prec_map_windows, ref_len, "precision",
        ref_len=ref_len, gen_len=gen_len, Rn=Rn
    )
    
    rec_nas, rec_nas_internals = _calculate_global_nas(
        recall_indices, rec_map_windows, gen_len, "recall",
        ref_len=ref_len, gen_len=gen_len, Rn=Rn
    )
    
    global_nas = _calculate_f1(prec_nas, rec_nas)
    
    aligned_col = []
    for g_idx, r_idx in precision_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            aligned_col.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))
    
    aligned_row = []
    for g_idx, r_idx in recall_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            aligned_row.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))
    
    col_ratio, col_ratio_internals = _calculate_local_nas(aligned_col, prec_map_windows, ref_len, gen_len, Rn=Rn)
    row_ratio, row_ratio_internals = _calculate_local_nas(aligned_row, rec_map_windows, ref_len, gen_len, swap=True, Rn=Rn)
    
    local_nas = _calculate_f1(col_ratio, row_ratio)
    
    f1_nas = _calculate_f1(global_nas, local_nas)

    metrics = {
        "Precision Global NAS": prec_nas,
        "Recall Global NAS": rec_nas,
        "Global NAS": global_nas,
        "Precision Local NAS": col_ratio,
        "Recall Local NAS": row_ratio,
        "Local NAS": local_nas,
        "NAS": f1_nas
    }
    
    internals = {
        "precision_nas_internals": prec_nas_internals,
        "recall_nas_internals": rec_nas_internals,
        "precision_line_internals": col_ratio_internals,
        "recall_line_internals": row_ratio_internals,
        "aligned_precision": aligned_col,
        "aligned_recall": aligned_row,
    }
    
    return metrics, internals