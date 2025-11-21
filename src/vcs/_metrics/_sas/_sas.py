import numpy as np
from typing import Dict, Any, Tuple
from .._utils import _compute_sas


def _compute_sas_metrics(
    global_sas: float,
    local_sas_metrics: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute Semantic Alignment Score (SAS) by combining Global_SAS and Local_SAS.

    Args:
        global_sas: Global Semantic Alignment Score value
        local_sas_metrics: Dict containing Local_SAS metrics with keys:
            - "Local_SAS": float (Local component)
            - Other keys for internals

    Returns:
        Tuple containing:
        - Dict with "Global_SAS", "Local_SAS", "SAS" scores
        - Dict with internals/breakdown information
    """
    local_sas = local_sas_metrics["Local_SAS"]

    # Compute combined SAS from Global and Local components
    sas = _compute_sas(global_sas, local_sas)

    metrics = {
        "Global_SAS": global_sas,
        "Local_SAS": local_sas,
        "SAS": sas
    }

    internals = {
        "global_sas_internals": {
            "value": global_sas,
        },
        "local_sas_internals": local_sas_metrics.get("internals", {}) if isinstance(local_sas_metrics.get("internals"), dict) else {},
        "scaling_details": {
            "sas": sas,
            "computation": "SAS = _compute_sas(Global_SAS, Local_SAS)"
        }
    }

    return metrics, internals
