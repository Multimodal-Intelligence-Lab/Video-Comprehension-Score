from typing import Dict
from ._utils import _compute_vcs_scaled


def _compute_vcs_metrics(
    sas: float,
    nas: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute Video Comprehension Score (VCS) from SAS and NAS.

    Args:
        sas: Semantic Alignment Score (combined Global_SAS and Local_SAS)
        nas: Dict with NAS metrics, must contain "NAS" key

    Returns:
        Dict with "SAS" and "VCS" scores
    """
    vcs = _compute_vcs_scaled(sas, nas["NAS"])

    return {
        "SAS": sas,
        "VCS": vcs,
    }