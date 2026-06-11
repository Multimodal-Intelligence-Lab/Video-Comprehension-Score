"""Metric semantics pinned by the v3 math batch.

Direct one-hot constructions through compute_vcs_from_embeddings for
behaviors the golden text cases cannot reach: exact similarity control
(orthogonal / anti-parallel chunks) and degenerate grid shapes. Basis
vectors give bit-exact dot products, so assertions are exact unless a
genuine float chain is involved.
"""
import pytest
import torch

from cases import CASES, build_call_kwargs
from vcs import compute_vcs_from_embeddings, compute_vcs_score

DIM = 8


def e(i):
    v = torch.zeros(DIM, dtype=torch.float64)
    v[i] = 1.0
    return v


def run(ref_rows, gen_rows, ref_doc=None, gen_doc=None, **kwargs):
    return compute_vcs_from_embeddings(
        ref_doc if ref_doc is not None else e(0),
        gen_doc if gen_doc is not None else e(0),
        torch.stack(ref_rows),
        torch.stack(gen_rows),
        return_all_metrics=True,
        return_internals=True,
        **kwargs,
    )


# --- A2: similarities are clamped at 0 ---------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_similarity_matrix_nonnegative(case):
    out = compute_vcs_score(**build_call_kwargs(case), return_internals=True)
    matrix = out["internals"]["similarity"]["matrix"]
    assert all(value >= 0.0 for row in matrix for value in row)


def test_no_signal_query_matches_in_window():
    """A gen chunk anti-correlated with EVERY ref chunk carries no alignment
    signal: after the clamp its column is all zeros, so matching falls
    through to window-distance selection and the chunk maps INSIDE its
    alignment window at similarity 0.0 with no chronology penalty.
    (Unclamped, the lone argmax of a negative column would win outright,
    wherever it sits.)"""
    neg_all = -(e(0) + e(1) + e(2) + e(3)) / 2.0  # unit norm, all dots -0.5
    out = run([e(0), e(1), e(2), e(3)], [e(0), neg_all, e(3)])

    matrix = out["internals"]["similarity"]["matrix"]
    assert [row[1] for row in matrix] == [0.0, 0.0, 0.0, 0.0]

    segment = out["internals"]["best_match"]["precision"]["segments"][1]
    selection = segment["selection"]
    assert selection["context_window_applied"] is False
    assert selection["max_value"] == 0.0
    assert selection["selection_reason"] == "in_alignment_window"
    window = segment["alignment_window"]
    assert window["start"] <= selection["selected_index"] < window["end"]

    # no chronology penalty from the no-signal chunk in either direction
    assert out["Precision Global NAS"] == 1.0
    assert out["Recall Global NAS"] == 1.0


def test_negative_gas_unclamped_dies_at_gate():
    """Doc-level GAS deliberately stays unclamped: anti-parallel document
    embeddings report Global_SAS = -1.0 and the SAS gate (not a clamp on
    the cosine) zeroes the score."""
    out = run([e(0)], [e(0)], ref_doc=e(1), gen_doc=-e(1))
    assert out["Global_SAS"] == -1.0
    assert out["SAS"] == 0.0
    assert out["VCS"] == 0.0
