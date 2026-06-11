"""Input validation for the public scoring entry points.

Cheap argument checks run before any computation; embedding-output checks
run at the moment each embedding tensor is produced, before it is consumed.
Everything raises ValueError with a message that names the offending
argument. Embedding rows are L2-normalized internally: VCS uses raw dot
products as similarities, so the unit-norm contract is enforced rather
than warned about — rows already within 1e-12 of unit norm pass through
bit-untouched, off rows are renormalized, and zero-norm rows (which have
no direction to normalize) raise.
"""
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
    return _normalize_embedding_rows(embeddings, fn_label)


def _validate_doc_embedding(embedding, label: str) -> torch.Tensor:
    """Document-level embedding for compute_vcs_from_embeddings: a 1-D
    tensor (embedding_dim,); a 2-D single-row tensor is accepted and
    squeezed."""
    if not isinstance(embedding, torch.Tensor):
        raise ValueError(f"{label} must be a torch.Tensor, got {type(embedding).__name__}")
    if embedding.dim() == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]
    if embedding.dim() != 1 or embedding.shape[0] == 0:
        raise ValueError(
            f"{label} must be a 1-D tensor of shape (embedding_dim,) "
            f"(or 2-D with a single row), got shape {tuple(embedding.shape)}"
        )
    if not torch.isfinite(embedding).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return _normalize_embedding_rows(embedding.unsqueeze(0), label)[0]


def _validate_chunk_embeddings(embeddings, label: str) -> torch.Tensor:
    """Chunk-level embedding matrix for compute_vcs_from_embeddings: a 2-D
    tensor (n_chunks, embedding_dim) with at least one row."""
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError(f"{label} must be a torch.Tensor, got {type(embeddings).__name__}")
    if embeddings.dim() != 2 or embeddings.shape[0] < 1:
        raise ValueError(
            f"{label} must be a 2-D tensor of shape (n_chunks, embedding_dim) "
            f"with at least one row, got shape {tuple(embeddings.shape)}"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return _normalize_embedding_rows(embeddings, label)


def _resolve_chunk_texts(chunks, expected_len: int, label: str) -> List[str]:
    """Chunk texts are optional in compute_vcs_from_embeddings (they only
    appear in internals, never in any score); placeholders are generated
    when omitted."""
    if chunks is None:
        return [f"<chunk {i}>" for i in range(expected_len)]
    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        raise ValueError(f"{label} must be a list of str (or None for placeholders)")
    if len(chunks) != expected_len:
        raise ValueError(
            f"{label} has {len(chunks)} texts but the corresponding chunk "
            f"embeddings have {expected_len} rows"
        )
    return chunks


def _normalize_embedding_rows(
    embeddings: torch.Tensor, fn_label: str, atol: float = 1e-12
) -> torch.Tensor:
    """Enforce the unit-norm contract on embedding rows.

    Rows whose L2 norm is already within ``atol`` of 1 are not touched —
    when every row complies, the INPUT TENSOR ITSELF is returned, so
    compliant embeddings stay bit-identical by construction. Otherwise a
    clone is returned with only the off rows divided by their norms.
    Zero-norm rows have no direction to normalize and raise instead of
    silently producing a garbage similarity row.
    """
    if embeddings.shape[0] == 0:
        return embeddings
    norms = torch.linalg.vector_norm(embeddings.detach().to(torch.float64), dim=1)
    if (norms == 0.0).any():
        row = int((norms == 0.0).nonzero()[0])
        raise ValueError(
            f"{fn_label} contains an all-zero embedding row (row {row} has "
            "L2 norm 0), which cannot be normalized to unit length"
        )
    off = (norms - 1.0).abs() > atol
    if not bool(off.any()):
        return embeddings
    if not embeddings.dtype.is_floating_point:
        raise ValueError(
            f"{fn_label} returned non-unit-norm rows with a non-floating "
            f"dtype ({embeddings.dtype}); provide floating-point embeddings "
            "so the rows can be L2-normalized"
        )
    normalized = embeddings.clone()
    normalized[off] = embeddings[off] / norms[off].to(embeddings.dtype).unsqueeze(1)
    return normalized
