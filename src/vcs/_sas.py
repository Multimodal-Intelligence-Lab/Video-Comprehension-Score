import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Tuple

from ._utils import _calculate_f1, _compute_sas
from ._validation import _validate_embedding_output


def _apply_scaling_function(similarity: float, semantic_coverage: float) -> float:
    """
    Apply the affine scaling function based on semantic coverage.
    
    scaling_function(s) = {
        0,                                          if coverage=0 or s ≤ 1-coverage
        (s - (1-coverage)) / coverage,             if coverage∈(0,1] and s > 1-coverage
    }
    
    Key properties:
    - scaling_function(1) = 1 (perfect similarity maps to 1)
    - Threshold at 1-coverage
    - Monotone and bounded in [0,1]
    - When coverage=1: scaling_function(s) = s (no penalty when adequacy is perfect)
    
    Args:
        similarity: Original similarity value
        semantic_coverage: Semantic coverage ratio (0 ≤ coverage ≤ 1)
    
    Returns:
        Scaled similarity value based on semantic coverage
    """
    if semantic_coverage == 0 or similarity <= (1 - semantic_coverage):
        return 0.0
    else:
        return (similarity - (1 - semantic_coverage)) / semantic_coverage


def _apply_semantic_load_penalty(
    sim_values: np.ndarray,
    indices: np.ndarray,
    opposite_len: int,
    direction: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply the semantic load sharing penalty for one matching direction.

    Chunks are grouped by the opposite-side chunk they matched to; when two
    or more chunks share one target, their semantic coverage (total supply /
    total demand) scales every member through the affine scaling function.
    The computation is direction-symmetric; only the internals key names
    differ ("who shared whom"), so they are parameterized by direction:

    - precision: generated chunks grouped by chosen reference chunk
    - recall:    reference chunks grouped by chosen generated chunk

    Returns:
        Tuple containing adjusted similarities and internal calculation details
    """
    if direction == "precision":
        group_key, count_key, member_key = (
            "reference_chunk_idx", "num_gens_shared", "generated_chunk_idx",
        )
    elif direction == "recall":
        group_key, count_key, member_key = (
            "generated_chunk_idx", "num_refs_shared", "reference_chunk_idx",
        )
    else:
        raise ValueError(f"direction must be 'precision' or 'recall', got {direction!r}")

    if sim_values.size == 0:
        return sim_values, {"load_sharing_details": []}

    adjusted_sim = sim_values.copy()
    load_sharing_details = []

    # Group this side's chunks by the opposite-side chunk they chose.
    # np.unique discovers the shared targets directly (ascending, matching
    # v1's range() visit order) instead of probing every possible target;
    # the per-group member selection and np.sum reduction are unchanged,
    # so results stay bit-identical.
    unique_targets, target_counts = np.unique(indices, return_counts=True)
    for group_idx, target_count in zip(unique_targets, target_counts):
        if group_idx < 0 or group_idx >= opposite_len or target_count < 2:
            continue
        members_sharing_this_group = np.where(indices == group_idx)[0]
        num_sharing = len(members_sharing_this_group)

        if num_sharing >= 2:  # Only apply penalty when load sharing occurs
            # Calculate semantic coverage ratio: total_supply / total_demand
            total_supply = np.sum(sim_values[members_sharing_this_group])
            total_demand = num_sharing  # Each member chunk demands 1 unit
            semantic_coverage = total_supply / total_demand
            chunk_details = []

            # Apply scaling function to each member sharing this group
            for member_idx in members_sharing_this_group:
                original_similarity = sim_values[member_idx]
                adjusted_similarity = _apply_scaling_function(original_similarity, semantic_coverage)
                adjusted_sim[member_idx] = adjusted_similarity

                chunk_details.append({
                    member_key: int(member_idx),
                    "original_similarity": float(original_similarity),
                    "adjusted_similarity": float(adjusted_similarity)
                })

            load_sharing_details.append({
                group_key: int(group_idx),
                count_key: int(num_sharing),
                "semantic_coverage": float(semantic_coverage),
                "load_sharing_penalty": float(1 - semantic_coverage),
                "chunk_details": chunk_details
            })

    internals = {"load_sharing_details": load_sharing_details}
    return adjusted_sim, internals


def _compute_global_sas_metrics(
    reference_text: str,
    generated_text: str,
    embedding_fn: Callable[[List[str]], torch.Tensor]
) -> float:

    emb_all = _validate_embedding_output(
        embedding_fn([reference_text, generated_text]), 2, "embedding_fn_global_sas"
    )

    return _global_sas_from_embeddings(emb_all[0], emb_all[1])


def _global_sas_from_embeddings(
    reference_vec: torch.Tensor,
    generated_vec: torch.Tensor,
) -> float:
    """Cosine similarity of two 1-D document embeddings. The single shared
    site for this op so both entry points are bit-identical."""
    ref_vec = reference_vec.unsqueeze(0)
    gen_vec = generated_vec.unsqueeze(0)
    sim = F.cosine_similarity(ref_vec, gen_vec, dim=1)

    return sim.item()


def _compute_local_sas_metrics(
    precision_sim_values: np.ndarray,
    recall_sim_values: np.ndarray,
    precision_indices: np.ndarray,
    recall_indices: np.ndarray,
    ref_len: int,
    gen_len: int
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute Local SAS metrics with bidirectional semantic load sharing penalty.

    This approach measures semantic coverage by comparing demand vs supply
    for each shared chunk, regardless of overall length balance. Both directions
    are penalized when load sharing occurs.

    Args:
        precision_sim_values: Similarity values for precision direction (gen->ref)
        recall_sim_values: Similarity values for recall direction (ref->gen)
        precision_indices: Matched reference indices for each generated chunk
        recall_indices: Matched generated indices for each reference chunk
        ref_len: Number of reference chunks (m)
        gen_len: Number of generated chunks (n)

    Returns:
        Tuple containing:
        - Dict with Precision Local SAS, Recall Local SAS, and combined Local SAS scores
        - Dict with internal calculations for both directions
    """
    
    # Apply semantic load sharing penalty to both directions
    adjusted_precision_sim, precision_internals = _apply_semantic_load_penalty(
        precision_sim_values, precision_indices, ref_len, "precision"
    )
    adjusted_recall_sim, recall_internals = _apply_semantic_load_penalty(
        recall_sim_values, recall_indices, gen_len, "recall"
    )
    
    # Calculate final Local_SAS scores
    precision_local_sas = float(np.mean(adjusted_precision_sim)) if adjusted_precision_sim.size else 0.0
    recall_local_sas = float(np.mean(adjusted_recall_sim)) if adjusted_recall_sim.size else 0.0
    f1_local_sas = _calculate_f1(precision_local_sas, recall_local_sas)

    metrics = {
        "Precision Local_SAS": precision_local_sas,
        "Recall Local_SAS": recall_local_sas,
        "Local_SAS": f1_local_sas
    }
    
    internals = {
        "precision": {
            "original_similarities": precision_sim_values.tolist(),
            "adjusted_similarities": adjusted_precision_sim.tolist(),
            "original_las": float(np.mean(precision_sim_values)) if precision_sim_values.size else 0.0,
            "adjusted_las": float(np.mean(adjusted_precision_sim)) if adjusted_precision_sim.size else 0.0,
            **precision_internals
        },
        "recall": {
            "original_similarities": recall_sim_values.tolist(),
            "adjusted_similarities": adjusted_recall_sim.tolist(),
            "original_las": float(np.mean(recall_sim_values)) if recall_sim_values.size else 0.0,
            "adjusted_las": float(np.mean(adjusted_recall_sim)) if adjusted_recall_sim.size else 0.0,
            **recall_internals
        }
    }
    
    return metrics, internals


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
