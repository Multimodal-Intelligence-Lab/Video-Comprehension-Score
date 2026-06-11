"""Return-shape contract of compute_vcs_score.

Asserts the exact key sets for each return-flag combination, independent of
the golden values. As of v2.0.0, 'Precision Local_SAS' / 'Recall Local_SAS'
are returned as the scorer docstring always promised (v1 computed but
dropped them in _compute_sas_metrics).
"""
import pytest

from cases import CASES, build_call_kwargs
from vcs import compute_vcs_score

METRIC_KEYS = {
    "VCS",
    "Global_SAS",
    "Precision Local_SAS",
    "Recall Local_SAS",
    "Local_SAS",
    "SAS",
    "Precision Global NAS",
    "Recall Global NAS",
    "Global NAS",
    "Precision Local NAS",
    "Recall Local NAS",
    "Local NAS",
    "NAS",
    "VCS Margin",
    "Config",
}

INTERNALS_TOP_KEYS = {
    "texts",
    "similarity",
    "alignment_windows",
    "alignment",
    "metrics",
    "config",
    "best_match",
}

INTERNALS_METRICS_KEYS = {
    "global_sas",
    "local_sas",
    "sas",
    "global_nas",
    "local_nas",
    "nas",
    "vcs",
}


def _kwargs(name="typical_8v3_defaults"):
    return build_call_kwargs(next(c for c in CASES if c["name"] == name))


def test_minimal_return_is_vcs_only():
    output = compute_vcs_score(**_kwargs())
    assert set(output) == {"VCS"}
    assert isinstance(output["VCS"], float)


def test_all_metrics_key_set():
    output = compute_vcs_score(**_kwargs(), return_all_metrics=True)
    assert set(output) == METRIC_KEYS


def test_internals_only_key_set():
    output = compute_vcs_score(**_kwargs(), return_internals=True)
    assert set(output) == {"VCS", "internals"}
    assert set(output["internals"]) == INTERNALS_TOP_KEYS


def test_internals_subtree_key_sets():
    output = compute_vcs_score(**_kwargs(), return_all_metrics=True, return_internals=True)
    internals = output["internals"]
    assert set(internals["metrics"]) == INTERNALS_METRICS_KEYS
    assert set(internals["metrics"]["vcs"]) == {"value", "margin"}
    assert set(internals["alignment"]) == {"precision", "recall"}
    for direction in ("precision", "recall"):
        assert set(internals["alignment"][direction]) == {
            "matches", "indices", "similarity_values", "aligned_segments",
        }
    assert set(internals["alignment_windows"]) == {"precision", "recall"}
    assert set(internals["config"]) == {
        "chunk_size", "context_cutoff_value", "context_window_control", "Rn",
    }
    assert set(internals["best_match"]) == {"precision", "recall"}
    assert set(internals["texts"]) == {
        "reference_chunks", "generated_chunks", "reference_length", "generated_length",
    }


def test_internals_dimensions_match_chunk_counts():
    output = compute_vcs_score(**_kwargs(), return_internals=True)
    internals = output["internals"]
    ref_len = internals["texts"]["reference_length"]
    gen_len = internals["texts"]["generated_length"]
    assert len(internals["texts"]["reference_chunks"]) == ref_len
    assert len(internals["texts"]["generated_chunks"]) == gen_len
    assert len(internals["similarity"]["matrix"]) == ref_len
    assert all(len(row) == gen_len for row in internals["similarity"]["matrix"])
    # precision iterates generated chunks, recall iterates reference chunks
    assert len(internals["alignment"]["precision"]["indices"]) == gen_len
    assert len(internals["alignment"]["recall"]["indices"]) == ref_len
    assert len(internals["alignment_windows"]["precision"]) == gen_len
    assert len(internals["alignment_windows"]["recall"]) == ref_len


@pytest.mark.parametrize("flags", [
    {},
    {"return_all_metrics": True},
    {"return_internals": True},
    {"return_all_metrics": True, "return_internals": True},
])
def test_vcs_value_is_invariant_to_return_flags(flags):
    baseline = compute_vcs_score(**_kwargs())["VCS"]
    assert compute_vcs_score(**_kwargs(), **flags)["VCS"] == baseline


@pytest.mark.parametrize("name", [
    "typical_8v3_defaults", "duplication_load_sharing", "rn_2", "reordered_8v8",
])
def test_all_metrics_identical_with_and_without_internals(name):
    """Detail collection is skipped entirely when internals are not
    requested (v2 perf gating) - every score must be bit-identical either
    way."""
    kwargs = build_call_kwargs(next(c for c in CASES if c["name"] == name))
    lean = compute_vcs_score(**kwargs, return_all_metrics=True)
    full = compute_vcs_score(**kwargs, return_all_metrics=True, return_internals=True)
    assert lean == {k: v for k, v in full.items() if k != "internals"}
