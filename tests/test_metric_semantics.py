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


def test_no_signal_distinct_negatives_dethrone_argmax():
    """Distinct negative similarities, with the LEAST negative one outside
    the alignment window: unclamped, that argmax was the single candidate
    and won wherever it sat; clamped to zeros, window-distance selection
    takes over and an in-window chunk wins instead."""
    raw = -(0.1 * e(0) + 0.5 * e(1) + 0.6 * e(2) + 0.7 * e(3))
    neg_distinct = raw / torch.linalg.vector_norm(raw)
    out = run([e(0), e(1), e(2), e(3)], [e(0), neg_distinct, e(3)])

    segment = out["internals"]["best_match"]["precision"]["segments"][1]
    window = segment["alignment_window"]
    # raw argmax (ref 0, the least negative) sits outside the window...
    assert not window["start"] <= 0 < window["end"]
    # ...but selection lands inside it, at clamped similarity 0.0
    selection = segment["selection"]
    assert selection["selection_reason"] == "in_alignment_window"
    assert window["start"] <= selection["selected_index"] < window["end"]


def test_negative_gas_unclamped_dies_at_gate():
    """Doc-level GAS deliberately stays unclamped: anti-parallel document
    embeddings report Global_SAS = -1.0 and the SAS gate (not a clamp on
    the cosine) zeroes the score."""
    out = run([e(0)], [e(0)], ref_doc=e(1), gen_doc=-e(1))
    assert out["Global_SAS"] == -1.0
    assert out["SAS"] == 0.0
    assert out["VCS"] == 0.0


# --- A1: min(m, n) == 1 makes Global NAS vacuously perfect --------------------

def test_single_chunk_identity_scores_one():
    out = run([e(0)], [e(0)])
    assert out["Precision Global NAS"] == 1.0
    assert out["Recall Global NAS"] == 1.0
    assert out["VCS"] == 1.0


def test_one_ref_many_gen_global_nas_vacuous():
    out = run([e(0)], [e(0), e(1), e(2), e(3), e(4)])
    assert out["Precision Global NAS"] == 1.0
    assert out["Recall Global NAS"] == 1.0
    assert out["Global NAS"] == 1.0


def test_many_ref_one_gen_missing_content_dies_via_sas():
    # The A1 fix must not hand out free points: a gen that dropped most of
    # the reference still dies, but at the SAS gate where it belongs (GAS
    # sees the missing content), not via a vacuous-chronology zero.
    out = run([e(0), e(1), e(2), e(3), e(4)], [e(0)], ref_doc=e(5), gen_doc=e(6))
    assert out["Global NAS"] == 1.0
    assert out["SAS"] == 0.0
    assert out["VCS"] == 0.0


def test_vacuous_branch_unreachable_at_2x2():
    # For min(m, n) >= 2 the max-penalty normalizer is positive: the
    # vacuous-1.0 branch is exclusive to the min(m, n) == 1 family.
    out = run([e(0), e(1)], [e(0), e(1)])
    global_nas = out["internals"]["metrics"]["global_nas"]
    assert global_nas["precision"]["max_penalty"] > 0
    assert global_nas["recall"]["max_penalty"] > 0


# --- C12: Local NAS per-step in-band bound ------------------------------------

def _assert_segments_match_window_geometry(out, direction):
    """Keystone geometric property: every segment's threshold must equal the
    in-band bound recomputed from the alignment windows and the segment's
    own recorded x positions — (end_window(x_to) - 1) - start_window(x_from)
    (x positions 1-based, windows 0-based) — and the CASE partition must
    follow from dy, the bound, and Rn."""
    internals = out["internals"]
    windows = internals["alignment_windows"][direction]
    rn = internals["config"]["Rn"]
    for seg in internals["metrics"]["local_nas"][direction]["segments"]:
        x_from, x_to = seg["start"][0], seg["end"][0]
        bound = (windows[x_to - 1][1] - 1) - windows[x_from - 1][0]
        assert seg["threshold"] == float(bound)
        assert seg["threshold_with_Rn"] == float(bound + rn)
        dy_value = abs(seg["dy"]) if rn > 0 else seg["dy"]
        if 0 <= dy_value <= bound:
            assert seg["is_calculable"] and seg["calculation_method"] == "standard"
        elif rn > 0 and bound < dy_value <= bound + rn:
            assert seg["is_calculable"] and seg["calculation_method"] == "Rn_capped"
        else:
            assert not seg["is_calculable"] and seg["calculation_method"] == "none"


@pytest.mark.parametrize("direction", ["precision", "recall"])
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_local_nas_thresholds_match_window_geometry(case, direction):
    out = compute_vcs_score(**build_call_kwargs(case), return_internals=True)
    _assert_segments_match_window_geometry(out, direction)


def test_per_step_bound_tightens_8v3_precision():
    """8v3 precision windows are [(0,3), (2,5), (5,8)]: the old global
    closed form allowed dy <= 5 on EVERY step, but between x=1 and x=2 no
    in-band path can jump more than (5-1) - 0 = 4. A dy=5 step there is now
    out of band (Rn=0) and exactly one Rn beyond the bound at Rn=1."""
    refs = [e(i) for i in range(8)]
    gens = [e(0), e(5), e(6)]  # matches land on refs 1, 6, 7 (1-based)

    seg = run(refs, gens)["internals"]["metrics"]["local_nas"]["precision"]["segments"][0]
    assert tuple(seg["start"]) == (1, 1) and tuple(seg["end"]) == (2, 6)
    assert seg["dy"] == 5
    assert seg["threshold"] == 4.0
    assert not seg["is_calculable"] and seg["calculation_method"] == "none"

    seg_rn = run(refs, gens, Rn=1)["internals"]["metrics"]["local_nas"]["precision"]["segments"][0]
    assert seg_rn["threshold"] == 4.0 and seg_rn["threshold_with_Rn"] == 5.0
    assert seg_rn["is_calculable"] and seg_rn["calculation_method"] == "Rn_capped"


def test_recall_bound_zero_step():
    """8v3 recall windows start [(0,1), (0,1), (0,2), ...]: between ref
    positions 1 and 2 an in-band path cannot move at all (bound 0), while
    the old global formula allowed dy <= 1 everywhere on the short axis."""
    refs = [e(i) for i in range(8)]
    gens = [e(0), e(1), e(5)]  # ref 2 matches gen 2 outside its (0,1) window

    seg = run(refs, gens)["internals"]["metrics"]["local_nas"]["recall"]["segments"][0]
    assert seg["dy"] == 1
    assert seg["threshold"] == 0.0
    assert not seg["is_calculable"] and seg["calculation_method"] == "none"
