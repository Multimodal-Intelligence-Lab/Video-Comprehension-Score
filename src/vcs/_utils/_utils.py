from typing import Callable


def _calculate_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom else 0.0

def _compute_sas(gas: float, las: float) -> float:
    if las <= 0:
        return 0.0
    val = gas - (1 - las)
    return (val / las) if (val > 0) else 0.0

def _compute_vcs_scaled(gas_scaled: float, nas: float) -> float:
    if gas_scaled < nas:
        numerator = gas_scaled - (1 - nas)
        denominator = nas
    else:
        numerator = nas - (1 - gas_scaled)
        denominator = gas_scaled
    
    return (numerator / denominator) if (numerator > 0 and denominator != 0) else 0.0

def _validate_seg_embed_functions(segmenter_fn: Callable, embedding_fn_global_sas: Callable, embedding_fn_local_sas: Callable | None = None) -> None:
    if not callable(segmenter_fn):
        raise ValueError("segmenter_fn must be a callable function!")
    if not callable(embedding_fn_global_sas):
        raise ValueError("embedding_fn_global_sas must be a callable function!")
    if embedding_fn_local_sas is not None and not callable(embedding_fn_local_sas):
        raise ValueError("embedding_fn_local_sas must be a callable function!")