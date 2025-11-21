"""
Semantic Alignment Score (SAS) modules.

Contains SAS computation combining Global_SAS and Local_SAS components.
"""

from ._sas import _compute_sas_metrics
from ._sas_components._global_sas._global_sas import _compute_global_sas_metrics
from ._sas_components._local_sas._local_sas import _compute_local_sas_metrics

__all__ = [
    "_compute_sas_metrics",
    "_compute_global_sas_metrics",
    "_compute_local_sas_metrics",
]
