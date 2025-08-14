from typing import List, Tuple, Dict, Any

from ._calculate_penalties._calculate_penalties import _calculate_penalties

def _calculate_length_regularizer(
    ref_len: int,
    gen_len: int,
    prec_map_windows: List[Tuple[int, int]],
    rec_map_windows: List[Tuple[int, int]],
    nas_blend_factor: float = 0.0,  # k
    nas_coverage_cutoff: float = 0.5,           # τ
) -> Tuple[float, Dict[str, Any]]:
    """Calculate length regularizer for NAS using penalty calculation."""
    out = _calculate_penalties(
        ref_len, gen_len, prec_map_windows, rec_map_windows,
        nas_blend_factor=nas_blend_factor, nas_coverage_cutoff=nas_coverage_cutoff
    )
    return out["length_regularizer"], out["internals"]

def _regularize_nas(nas_f1: float, length_regularizer: float) -> float:
    """Apply regularization to NAS F1 score."""
    nas_regularized = nas_f1 - length_regularizer
    return (nas_regularized / (1 - length_regularizer)) if (nas_regularized > 0) else 0.0