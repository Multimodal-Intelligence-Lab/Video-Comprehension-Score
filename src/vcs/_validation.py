"""Input validation for the public scoring entry points.

Cheap argument checks run before any computation; embedding-output checks
run at the moment each embedding tensor is produced, before it is consumed.
Everything raises ValueError with a message that names the offending
argument — except the L2-normalization check, which only warns: VCS uses
raw dot products as similarities, so un-normalized embeddings are almost
certainly a mistake, but erroring would reject inputs v1 accepted.
"""
import warnings
from typing import Callable, List

import torch


def _validate_seg_embed_functions(
    segmenter_fn: Callable,
    embedding_fn_global_sas: Callable,
    embedding_fn_local_sas: Callable | None = None,
) -> None:
    if not callable(segmenter_fn):
        raise ValueError("segmenter_fn must be a callable function!")
    if not callable(embedding_fn_global_sas):
        raise ValueError("embedding_fn_global_sas must be a callable function!")
    if embedding_fn_local_sas is not None and not callable(embedding_fn_local_sas):
        raise ValueError("embedding_fn_local_sas must be a callable function!")


def _validate_texts(reference_text, generated_text) -> None:
    for label, value in (
        ("reference_text", reference_text),
        ("generated_text", generated_text),
    ):
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError(f"{label} must be a non-empty string")


def _validate_parameters(chunk_size, context_cutoff_value, context_window_control, Rn) -> None:
    # bool is a subclass of int, so reject it explicitly everywhere
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    if isinstance(context_cutoff_value, bool) or not isinstance(context_cutoff_value, (int, float)):
        raise ValueError(
            f"context_cutoff_value must be a number, got {type(context_cutoff_value).__name__}"
        )
    if not 0.0 <= context_cutoff_value <= 1.0:
        raise ValueError(f"context_cutoff_value must be in [0, 1], got {context_cutoff_value}")

    if isinstance(context_window_control, bool) or not isinstance(context_window_control, (int, float)):
        raise ValueError(
            f"context_window_control must be a number, got {type(context_window_control).__name__}"
        )
    if context_window_control <= 0:
        raise ValueError(f"context_window_control must be > 0, got {context_window_control}")

    if isinstance(Rn, bool) or not isinstance(Rn, int):
        raise ValueError(f"Rn must be an int, got {type(Rn).__name__}")
    if Rn < 0:
        raise ValueError(f"Rn must be >= 0, got {Rn}")


def _validate_segments(segments: List[str], label: str) -> None:
    """Check raw segmenter output BEFORE chunking joins it: a non-list (e.g.
    a plain string) or non-str items would otherwise crash or silently
    produce garbage chunks."""
    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError(
            f"segmenter_fn must return a non-empty list of str for the {label} "
            f"text, got {type(segments).__name__}"
            + ("" if isinstance(segments, list) else " (did it return the text itself?)")
        )
    if not all(isinstance(segment, str) for segment in segments):
        raise ValueError(f"segmenter_fn must return a list of str for the {label} text")


def _validate_embedding_output(embeddings, expected_rows: int, fn_label: str) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError(
            f"{fn_label} must return a torch.Tensor, got {type(embeddings).__name__}"
        )
    if embeddings.dim() != 2:
        raise ValueError(
            f"{fn_label} must return a 2-D tensor of shape (n_texts, embedding_dim), "
            f"got shape {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{fn_label} returned {embeddings.shape[0]} embedding rows "
            f"for {expected_rows} input texts"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{fn_label} returned NaN or infinite values")
    _warn_if_not_normalized(embeddings, fn_label)
    return embeddings


def _warn_if_not_normalized(embeddings: torch.Tensor, fn_label: str, atol: float = 1e-3) -> None:
    norms = torch.linalg.vector_norm(embeddings.detach().to(torch.float64), dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=atol):
        warnings.warn(
            f"{fn_label} returned embeddings whose rows are not L2-normalized "
            "(row norms deviate from 1). VCS uses raw dot products as "
            "similarities, so un-normalized embeddings produce unbounded or "
            "misleading scores. Normalize each row to unit length.",
            UserWarning,
            stacklevel=3,
        )
