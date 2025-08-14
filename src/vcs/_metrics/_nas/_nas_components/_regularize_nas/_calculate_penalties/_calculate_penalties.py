from typing import List, Tuple, Dict, Any

from ._helpers.math_utils import clamp01
from ._helpers.blend_functions import blend_noisy_or

def _calculate_penalties(
    ref_len: int,
    gen_len: int,
    prec_map_windows: List[Tuple[int, int]],
    rec_map_windows: List[Tuple[int, int]],
    nas_blend_factor: float = 0.0,  # k in [-1, 1]: -1 => follow i_p; +1 => follow wp; 0 => default blend
    nas_coverage_cutoff: float = 0.5,           # τ in [0,1]: MW coverage at/above which NAS is considered meaningless
) -> Dict[str, Any]:
    """
    Returns:
      {
        'wp': float,                 # window penalty
        'i_p': float,                # length-imbalance penalty
        'length_regularizer': float, # blended final penalty
        'internals': {               # trimmed diagnostics (no k/tau, no L0/exponents/M/m)
            'coverage_ratio': float,
            'total_mapping_window_area': int,
            'timeline_area': int,
            'min_area': float,
        }
      }
    """
    # Choose MW set (legacy logic)
    if ref_len < gen_len:
        mapping_windows = rec_map_windows
        max_len = gen_len
    else:
        mapping_windows = prec_map_windows
        max_len = ref_len

    # Calculate areas
    total_mapping_window_area = 0
    for start, end in mapping_windows:
        mapping_window_height = end - start
        total_mapping_window_area += mapping_window_height * 1  # width=1

    timeline_area = ref_len * gen_len
    min_area = (1 / max_len) if max_len > 0 else 0.0

    # Window penalty wp (uses nas_coverage_cutoff)
    nas_coverage_cutoff = clamp01(nas_coverage_cutoff)
    coverage_ratio = (total_mapping_window_area / timeline_area) if timeline_area > 0 else 0.0

    if timeline_area > 0 and min_area < nas_coverage_cutoff and (nas_coverage_cutoff - min_area) > 1e-9:
        wp_val = (coverage_ratio - min_area) / (nas_coverage_cutoff - min_area)
        wp = clamp01(wp_val)
    elif timeline_area > 0 and min_area >= nas_coverage_cutoff:
        wp = 0.0 if coverage_ratio <= (min_area + 1e-9) else 1.0
    else:
        wp = 0.0

    # Imbalance penalty i_p
    i_p = 0.0 if max(ref_len, gen_len) == 0 else (1.0 - min(ref_len, gen_len) / max(ref_len, gen_len))
    i_p = clamp01(i_p)

    # Final length regularizer using blending
    length_regularizer = blend_noisy_or(i_p, wp, nas_blend_factor)

    return {
        "wp": wp,
        "i_p": i_p,
        "length_regularizer": length_regularizer,
        "internals": {
            "coverage_ratio": coverage_ratio,
            "total_mapping_window_area": total_mapping_window_area,
            "timeline_area": timeline_area,
            "min_area": min_area,
        },
    }