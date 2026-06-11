"""Input-validation contract added in v2.0.0 (plan C9).

Every invalid input must raise ValueError BEFORE any metric math runs,
with a message naming the offending argument. Since v3, embedding rows
are L2-normalized internally (raw-dot-product similarity contract):
compliant rows pass through bit-identical, off rows are renormalized,
zero-norm rows raise.
"""
import warnings

import numpy as np
import pytest
import torch

from cases import CASES, build_call_kwargs
from embedder import embed_dim64, split_sentences
from vcs import compute_vcs_score

REF = "The storm rolled in from the north. The fishermen tied their boats."
GEN = "A storm came from the north. The boats were tied down."


def call(**overrides):
    kwargs = dict(
        reference_text=REF,
        generated_text=GEN,
        segmenter_fn=split_sentences,
        embedding_fn_global_sas=embed_dim64,
    )
    kwargs.update(overrides)
    return compute_vcs_score(**kwargs)


# --- function arguments -----------------------------------------------------

def test_missing_global_embedder_raises():
    with pytest.raises(ValueError, match="embedding_fn_global_sas is required"):
        call(embedding_fn_global_sas=None)


@pytest.mark.parametrize("field", ["segmenter_fn", "embedding_fn_local_sas"])
def test_non_callable_functions_raise(field):
    with pytest.raises(ValueError, match=field):
        call(**{field: "not callable"})


# --- texts -------------------------------------------------------------------

@pytest.mark.parametrize("field", ["reference_text", "generated_text"])
@pytest.mark.parametrize("bad,pattern", [
    (None, "must be a str"),
    (42, "must be a str"),
    (["a list"], "must be a str"),
    ("", "non-empty"),
    ("   \n\t ", "non-empty"),
])
def test_bad_texts_raise(field, bad, pattern):
    with pytest.raises(ValueError, match=f"{field}.*{pattern}"):
        call(**{field: bad})


# --- numeric parameters --------------------------------------------------------

@pytest.mark.parametrize("field,bad,pattern", [
    ("chunk_size", 0, ">= 1"),
    ("chunk_size", -3, ">= 1"),
    ("chunk_size", 1.5, "must be an int"),
    ("chunk_size", True, "must be an int"),
    ("chunk_size", "2", "must be an int"),
    ("context_cutoff_value", -0.1, r"in \[0, 1\]"),
    ("context_cutoff_value", 1.0001, r"in \[0, 1\]"),
    ("context_cutoff_value", "0.6", "must be a number"),
    ("context_cutoff_value", True, "must be a number"),
    ("context_window_control", 0, "> 0"),
    ("context_window_control", -4.0, "> 0"),
    ("context_window_control", None, "must be a number"),
    ("context_window_control", True, "must be a number"),
    ("Rn", -1, ">= 0"),
    ("Rn", 0.5, "must be an int"),
    ("Rn", True, "must be an int"),
])
def test_bad_parameters_raise(field, bad, pattern):
    with pytest.raises(ValueError, match=pattern):
        call(**{field: bad})


def test_boundary_parameter_values_accepted():
    result = call(chunk_size=1, context_cutoff_value=0.0, context_window_control=0.5, Rn=0)
    assert 0.0 <= result["VCS"] <= 1.0
    result = call(context_cutoff_value=1.0)
    assert 0.0 <= result["VCS"] <= 1.0


# --- segmenter output ----------------------------------------------------------

def test_segmenter_returning_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty list of str for the reference"):
        call(segmenter_fn=lambda text: [])


def test_segmenter_returning_non_list_raises():
    # returning the text itself would otherwise be silently sliced into
    # character-run garbage chunks by _group_segments
    with pytest.raises(ValueError, match="non-empty list of str"):
        call(segmenter_fn=lambda text: text)


def test_segmenter_returning_non_str_items_raises():
    with pytest.raises(ValueError, match="list of str"):
        call(segmenter_fn=lambda text: [1, 2, 3])


# --- embedding output -----------------------------------------------------------

def test_embedder_returning_numpy_raises():
    def np_embedder(texts):
        return np.zeros((len(texts), 8))
    with pytest.raises(ValueError, match="must return a torch.Tensor"):
        call(embedding_fn_global_sas=np_embedder)


def test_embedder_returning_1d_tensor_raises():
    def flat_embedder(texts):
        return torch.ones(len(texts), dtype=torch.float64)
    with pytest.raises(ValueError, match="2-D tensor"):
        call(embedding_fn_global_sas=flat_embedder)


def test_embedder_row_count_mismatch_raises():
    def short_embedder(texts):
        return torch.eye(max(len(texts) - 1, 1), 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="embedding rows"):
        call(embedding_fn_global_sas=short_embedder)


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_embedder_nan_or_inf_raises(poison):
    def poisoned_embedder(texts):
        out = embed_dim64(texts).clone()
        out[0, 0] = poison
        return out
    with pytest.raises(ValueError, match="NaN or infinite"):
        call(embedding_fn_global_sas=poisoned_embedder)


def test_local_embedder_errors_name_the_local_function():
    def bad_local(texts):
        return np.zeros((len(texts), 8))
    with pytest.raises(ValueError, match="embedding_fn_local_sas"):
        call(embedding_fn_local_sas=bad_local)


# --- L2 normalization: enforced internally ---------------------------------------

def test_unnormalized_embeddings_are_scale_invariant():
    # Scaling every embedding row by a constant must not move any output:
    # rows are renormalized internally. Tolerance, never exact — the
    # renormalized row (c*v)/||c*v|| differs from v in the last ulps.
    from helpers import assert_structurally_equal, canonicalize

    def scaled(texts):
        return embed_dim64(texts) * 3.7

    baseline = call(return_all_metrics=True, return_internals=True)
    result = call(
        embedding_fn_global_sas=scaled,
        return_all_metrics=True, return_internals=True,
    )
    assert_structurally_equal(canonicalize(result), canonicalize(baseline), atol=1e-12)


def test_compliant_embeddings_pass_through_bit_identical():
    # The bit-no-op for compliant inputs is structural (the same tensor
    # object comes back), not numerical luck.
    from vcs._validation import _normalize_embedding_rows

    emb = embed_dim64(["a storm", "the boats"])
    assert _normalize_embedding_rows(emb, "embedding_fn_global_sas") is emb


def test_compliant_embeddings_do_not_warn():
    # Assert only on OUR warning: escalating every UserWarning to an error
    # couples the test to environment noise (e.g. torch first-use warnings
    # on CI runners that don't fire locally).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    ours = [w for w in caught if "normaliz" in str(w.message).lower()]
    assert not ours, f"unexpected normalization warning: {ours[0].message}"


def test_zero_norm_embedding_row_raises():
    def zero_row(texts):
        out = embed_dim64(texts).clone()
        out[0] = 0.0
        return out
    with pytest.raises(ValueError, match="embedding_fn_global_sas.*L2 norm 0"):
        call(embedding_fn_global_sas=zero_row)


# --- validation must not change valid-input behavior ----------------------------

def test_valid_golden_case_unaffected():
    # Golden comparisons must use the tolerant comparator: exact == against
    # the committed file fails on hardware whose BLAS rounds float64 ops
    # differently by ~1e-16 (this exact mistake broke CI once).
    import json
    from pathlib import Path

    from helpers import assert_structurally_equal, canonicalize
    golden = json.loads((Path(__file__).parent / "golden" / "golden_cases.json").read_text())
    case = next(c for c in CASES if c["name"] == "typical_8v3_defaults")
    out = compute_vcs_score(**build_call_kwargs(case), return_all_metrics=True, return_internals=True)
    assert_structurally_equal(
        canonicalize(out), golden["cases"]["typical_8v3_defaults"]["output"], atol=1e-12
    )
