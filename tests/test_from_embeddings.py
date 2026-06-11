"""compute_vcs_from_embeddings contract (added v2.0.0, plan C10).

Keystone: for every characterization case, feeding the embeddings that
compute_vcs_score would itself produce must yield EXACTLY equal results
(bit-identical floats, identical internals up to the documented
config.chunk_size = None difference).
"""
import numpy as np
import pytest
import torch

from cases import CASES, SEGMENTERS, EMBEDDERS, build_call_kwargs
from embedder import embed_dim64
from helpers import canonicalize
from vcs import compute_vcs_from_embeddings, compute_vcs_score
from vcs._segmenting import _group_segments


def embeddings_for(case):
    """Reproduce exactly the embedding inputs compute_vcs_score would build."""
    kwargs = build_call_kwargs(case)
    segmenter = kwargs["segmenter_fn"]
    global_embedder = kwargs["embedding_fn_global_sas"]
    local_embedder = kwargs.get("embedding_fn_local_sas", global_embedder)
    chunk_size = kwargs.get("chunk_size", 1)

    doc_embeds = global_embedder([kwargs["reference_text"], kwargs["generated_text"]])
    ref_chunks = _group_segments(segmenter(kwargs["reference_text"]), chunk_size)
    gen_chunks = _group_segments(segmenter(kwargs["generated_text"]), chunk_size)
    return {
        "reference_doc_embedding": doc_embeds[0],
        "generated_doc_embedding": doc_embeds[1],
        "reference_chunk_embeddings": local_embedder(ref_chunks),
        "generated_chunk_embeddings": local_embedder(gen_chunks),
        "reference_chunks": ref_chunks,
        "generated_chunks": gen_chunks,
        "context_cutoff_value": kwargs.get("context_cutoff_value", 0.6),
        "context_window_control": kwargs.get("context_window_control", 4.0),
        "Rn": kwargs.get("Rn", 0),
    }


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_exactly_equal_to_text_entry_point(case):
    from_text = compute_vcs_score(
        **build_call_kwargs(case), return_all_metrics=True, return_internals=True
    )
    from_emb = compute_vcs_from_embeddings(
        **embeddings_for(case), return_all_metrics=True, return_internals=True
    )

    a, b = canonicalize(from_text), canonicalize(from_emb)
    # the single documented difference: chunking happened upstream
    assert b["internals"]["config"]["chunk_size"] is None
    a["internals"]["config"]["chunk_size"] = None
    assert a == b  # exact equality, floats included


def test_placeholder_chunk_texts_do_not_affect_scores():
    case = next(c for c in CASES if c["name"] == "typical_8v3_defaults")
    inputs = embeddings_for(case)
    with_texts = compute_vcs_from_embeddings(**inputs, return_all_metrics=True)
    del inputs["reference_chunks"], inputs["generated_chunks"]
    with_placeholders = compute_vcs_from_embeddings(**inputs, return_all_metrics=True)
    assert with_texts == with_placeholders


def test_placeholder_texts_appear_in_internals():
    case = next(c for c in CASES if c["name"] == "square_4v4_defaults")
    inputs = embeddings_for(case)
    del inputs["reference_chunks"], inputs["generated_chunks"]
    out = compute_vcs_from_embeddings(**inputs, return_internals=True)
    assert out["internals"]["texts"]["reference_chunks"] == [f"<chunk {i}>" for i in range(4)]


def test_single_row_2d_doc_embeddings_accepted():
    case = next(c for c in CASES if c["name"] == "square_4v4_defaults")
    inputs = embeddings_for(case)
    flat = compute_vcs_from_embeddings(**inputs)
    inputs["reference_doc_embedding"] = inputs["reference_doc_embedding"].unsqueeze(0)
    inputs["generated_doc_embedding"] = inputs["generated_doc_embedding"].unsqueeze(0)
    assert compute_vcs_from_embeddings(**inputs) == flat


# --- validation of the new inputs -------------------------------------------

def _square_inputs():
    case = next(c for c in CASES if c["name"] == "square_4v4_defaults")
    return embeddings_for(case)


def test_doc_embedding_wrong_type_raises():
    inputs = _square_inputs()
    inputs["reference_doc_embedding"] = np.zeros(8)
    with pytest.raises(ValueError, match="reference_doc_embedding must be a torch.Tensor"):
        compute_vcs_from_embeddings(**inputs)


def test_doc_embedding_wrong_shape_raises():
    inputs = _square_inputs()
    inputs["generated_doc_embedding"] = torch.zeros((3, 8), dtype=torch.float64)
    with pytest.raises(ValueError, match="generated_doc_embedding must be a 1-D tensor"):
        compute_vcs_from_embeddings(**inputs)


def test_doc_embedding_dim_mismatch_raises():
    inputs = _square_inputs()
    inputs["generated_doc_embedding"] = torch.ones(16, dtype=torch.float64) / 4.0
    with pytest.raises(ValueError, match="document embeddings disagree"):
        compute_vcs_from_embeddings(**inputs)


def test_chunk_embeddings_1d_raises():
    inputs = _square_inputs()
    inputs["reference_chunk_embeddings"] = torch.ones(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="reference_chunk_embeddings must be a 2-D tensor"):
        compute_vcs_from_embeddings(**inputs)


def test_chunk_embedding_dim_mismatch_raises():
    inputs = _square_inputs()
    n = inputs["generated_chunk_embeddings"].shape[0]
    inputs["generated_chunk_embeddings"] = torch.eye(n, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="chunk embeddings disagree"):
        compute_vcs_from_embeddings(**inputs)


def test_chunk_text_length_mismatch_raises():
    inputs = _square_inputs()
    inputs["reference_chunks"] = ["only one text"]
    with pytest.raises(ValueError, match="reference_chunks has 1 texts"):
        compute_vcs_from_embeddings(**inputs)


def test_nan_chunk_embeddings_raise():
    inputs = _square_inputs()
    poisoned = inputs["reference_chunk_embeddings"].clone()
    poisoned[0, 0] = float("nan")
    inputs["reference_chunk_embeddings"] = poisoned
    with pytest.raises(ValueError, match="NaN or infinite"):
        compute_vcs_from_embeddings(**inputs)


def test_unnormalized_chunk_embeddings_are_scale_invariant():
    # Rows are renormalized internally; tolerance, never exact — the
    # renormalized row (c*v)/||c*v|| differs from v in the last ulps.
    from helpers import assert_structurally_equal

    inputs = _square_inputs()
    baseline = compute_vcs_from_embeddings(
        **inputs, return_all_metrics=True, return_internals=True
    )
    inputs["reference_chunk_embeddings"] = inputs["reference_chunk_embeddings"] * 2.5
    inputs["generated_doc_embedding"] = inputs["generated_doc_embedding"] * 0.4
    scaled = compute_vcs_from_embeddings(
        **inputs, return_all_metrics=True, return_internals=True
    )
    assert_structurally_equal(canonicalize(scaled), canonicalize(baseline), atol=1e-12)


def test_zero_norm_doc_embedding_raises():
    inputs = _square_inputs()
    inputs["reference_doc_embedding"] = torch.zeros_like(inputs["reference_doc_embedding"])
    with pytest.raises(ValueError, match="reference_doc_embedding.*L2 norm 0"):
        compute_vcs_from_embeddings(**inputs)


def test_zero_norm_chunk_embedding_row_raises():
    inputs = _square_inputs()
    zeroed = inputs["generated_chunk_embeddings"].clone()
    zeroed[1] = 0.0
    inputs["generated_chunk_embeddings"] = zeroed
    with pytest.raises(ValueError, match="generated_chunk_embeddings.*row 1.*L2 norm 0"):
        compute_vcs_from_embeddings(**inputs)


def test_exported_from_package_root():
    import vcs

    assert callable(vcs.compute_vcs_from_embeddings)
    assert "compute_vcs_from_embeddings" in vcs.__all__
