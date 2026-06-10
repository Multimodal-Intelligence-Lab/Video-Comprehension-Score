import numpy as np
from typing import Dict, Any, Tuple
from ..._utils import _compute_sas


def _compute_sas_metrics(
    global_sas: float,
    local_sas_metrics: Dict[str, Any],
    local_sas_internals: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute Semantic Alignment Score (SAS) by combining Global_SAS and Local_SAS.

    Args:
        global_sas: Global Semantic Alignment Score value
        local_sas_metrics: Dict with "Precision Local_SAS", "Recall Local_SAS",
            and "Local_SAS" scores
        local_sas_internals: Internal calculation details from the Local_SAS
            computation, surfaced under "local_sas_internals"

    Returns:
        Tuple containing:
        - Dict with "Global_SAS", "Precision Local_SAS", "Recall Local_SAS",
          "Local_SAS", "SAS" scores
        - Dict with internals/breakdown information
    """
    local_sas = local_sas_metrics["Local_SAS"]

    # Compute combined SAS from Global and Local components
    sas = _compute_sas(global_sas, local_sas)

    metrics = {
        "Global_SAS": global_sas,
        "Precision Local_SAS": local_sas_metrics["Precision Local_SAS"],
        "Recall Local_SAS": local_sas_metrics["Recall Local_SAS"],
        "Local_SAS": local_sas,
        "SAS": sas
    }

    internals = {
        "global_sas_internals": {
            "value": global_sas,
        },
        "local_sas_internals": local_sas_internals if local_sas_internals is not None else {},
        "scaling_details": {
            "sas": sas,
            "computation": "SAS = _compute_sas(Global_SAS, Local_SAS)"
        }
    }

    return metrics, internals
