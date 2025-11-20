from typing import Dict
import numpy as np
from ..._utils import _compute_sas, _compute_vcs_scaled

def _compute_vcs_metrics(
    gas: float,
    nas: Dict[str, float],
    las: float,
) -> Dict[str, float]:

    sas = _compute_sas(gas, las)
    vcs = _compute_vcs_scaled(sas, nas)

    return {
        "GAS": gas,
        "SAS": sas,
        "VCS": vcs,
    }