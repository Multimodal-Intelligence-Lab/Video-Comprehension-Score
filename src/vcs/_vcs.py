from typing import Dict
from ._utils import _compute_vcs_scaled


def _compute_vcs_metrics(
    sas: float,
    nas: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute Video Comprehension Score (VCS) from SAS and NAS.

    "VCS Margin" = SAS + NAS - 1 in [-1, 1] is the shared Lukasiewicz
    numerator of both _compute_vcs_scaled branches, so VCS > 0 iff
    margin > 0 and VCS == margin / max(SAS, NAS) when positive. Unlike
    VCS it is not clamped, so it keeps ranking candidates that the VCS
    zero gate maps to a flat 0.

    Args:
        sas: Semantic Alignment Score (combined Global_SAS and Local_SAS)
        nas: Dict with NAS metrics, must contain "NAS" key

    Returns:
        Dict with "SAS", "VCS", and "VCS Margin" scores
    """
    vcs = _compute_vcs_scaled(sas, nas["NAS"])

    return {
        "SAS": sas,
        "VCS": vcs,
        "VCS Margin": sas + nas["NAS"] - 1.0,
    }