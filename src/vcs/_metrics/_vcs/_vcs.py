from typing import Dict
import numpy as np
from ..._utils import _compute_gas_las_scaled, _compute_vcs_scaled

def _compute_vcs_metrics(
    gas: float,
    nas: Dict[str, float],
    las: float,
) -> Dict[str, float]:

    gas_las_scaled = _compute_gas_las_scaled(gas, las)
    vcs = _compute_vcs_scaled(gas_las_scaled, nas)
    
    return {
        "GAS": gas,
        "GAS-LAS-Scaled": gas_las_scaled,
        "VCS": vcs,
    }