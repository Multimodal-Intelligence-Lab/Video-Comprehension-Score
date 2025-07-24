import numpy as np
from typing import List, Tuple, Dict, Any
from ..._utils import _calculate_f1

from ._nas_components._regularize_nas._regularize_nas import _calculate_window_regularizer, _regularize_nas
from ._nas_components._distance_nas._distance_nas import _calculate_distance_based_nas
from ._nas_components._line_nas._line_nas import _calculate_line_based_nas

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
    lct: int = 0
) -> Tuple[Dict[str, float], Dict[str, Any]]:

    prec_nas, prec_nas_internals = _calculate_distance_based_nas(
        precision_indices, prec_map_windows, ref_len, "precision",
        ref_len=ref_len, gen_len=gen_len, lct=lct
    )
    
    rec_nas, rec_nas_internals = _calculate_distance_based_nas(
        recall_indices, rec_map_windows, gen_len, "recall",
        ref_len=ref_len, gen_len=gen_len, lct=lct
    )
    
    nas_d = _calculate_f1(prec_nas, rec_nas)
    
    aligned_col = []
    for g_idx, r_idx in precision_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            aligned_col.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))
    
    aligned_row = []
    for g_idx, r_idx in recall_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            aligned_row.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))
    
    col_ratio, col_ratio_internals = _calculate_line_based_nas(aligned_col, prec_map_windows, ref_len, gen_len, lct=lct)
    row_ratio, row_ratio_internals = _calculate_line_based_nas(aligned_row, rec_map_windows, ref_len, gen_len, swap=True, lct=lct)
    
    nas_l = _calculate_f1(col_ratio, row_ratio)
    
    f1_nas = _calculate_f1(nas_d, nas_l)
    
    window_regularizer, regularizer_internals = _calculate_window_regularizer(ref_len, gen_len, prec_map_windows, rec_map_windows)
    regularized_nas = _regularize_nas(f1_nas, window_regularizer)

    metrics = {
        "Precision NAS-D": prec_nas,
        "Recall NAS-D": rec_nas,
        "NAS-D": nas_d,
        "Precision NAS-L": col_ratio,
        "Recall NAS-L": row_ratio,
        "NAS-L": nas_l,
        "NAS-F1": f1_nas,
        "Window-Regularizer": window_regularizer,
        "NAS": regularized_nas
    }
    
    internals = {
        "precision_nas_internals": prec_nas_internals,
        "recall_nas_internals": rec_nas_internals,
        "precision_line_internals": col_ratio_internals,
        "recall_line_internals": row_ratio_internals,
        "regularizer_internals": regularizer_internals,
        "aligned_precision": aligned_col,
        "aligned_recall": aligned_row,
    }
    
    return metrics, internals