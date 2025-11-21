import numpy as np
from typing import List, Tuple, Dict, Any
from .._utils import _calculate_f1

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
    precision_alignment_windows: List[Tuple[int, int]],
    recall_alignment_windows: List[Tuple[int, int]],
    ref_chunks: List[str],
    gen_chunks: List[str],
    Rn: int = 0
) -> Tuple[Dict[str, float], Dict[str, Any]]:

    precision_global_nas, precision_global_nas_internals = _calculate_global_nas(
        precision_indices, precision_alignment_windows, ref_len, "precision",
        ref_len=ref_len, gen_len=gen_len, Rn=Rn
    )

    recall_global_nas, recall_global_nas_internals = _calculate_global_nas(
        recall_indices, recall_alignment_windows, gen_len, "recall",
        ref_len=ref_len, gen_len=gen_len, Rn=Rn
    )

    global_nas = _calculate_f1(precision_global_nas, recall_global_nas)

    precision_aligned_segments = []
    for g_idx, r_idx in precision_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            precision_aligned_segments.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))

    recall_aligned_segments = []
    for g_idx, r_idx in recall_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            recall_aligned_segments.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))
    
    precision_local_nas, precision_local_nas_internals = _calculate_local_nas(precision_aligned_segments, precision_alignment_windows, ref_len, gen_len, Rn=Rn)
    recall_local_nas, recall_local_nas_internals = _calculate_local_nas(recall_aligned_segments, recall_alignment_windows, ref_len, gen_len, swap=True, Rn=Rn)

    local_nas = _calculate_f1(precision_local_nas, recall_local_nas)

    nas = _calculate_f1(global_nas, local_nas)

    metrics = {
        "Precision Global NAS": precision_global_nas,
        "Recall Global NAS": recall_global_nas,
        "Global NAS": global_nas,
        "Precision Local NAS": precision_local_nas,
        "Recall Local NAS": recall_local_nas,
        "Local NAS": local_nas,
        "NAS": nas
    }

    internals = {
        "precision_global_nas_internals": precision_global_nas_internals,
        "recall_global_nas_internals": recall_global_nas_internals,
        "precision_local_nas_internals": precision_local_nas_internals,
        "recall_local_nas_internals": recall_local_nas_internals,
        "aligned_precision": precision_aligned_segments,
        "aligned_recall": recall_aligned_segments,
    }
    
    return metrics, internals