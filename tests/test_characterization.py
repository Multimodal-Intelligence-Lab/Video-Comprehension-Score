"""Golden characterization tests.

Pins the FULL output of compute_vcs_score (all metrics + complete internals
tree) for 19 deterministic cases against tests/golden/golden_cases.json.
These tests define "behavior-identical" for the v2.0.0 software overhaul:
refactors must keep them green without regenerating the goldens; the only
commits allowed to regenerate are intentional output-shape changes, with
the JSON diff reviewed line by line.
"""
import json
from pathlib import Path

import pytest

from cases import CASES, build_call_kwargs
from helpers import assert_structurally_equal, canonicalize
from vcs import compute_vcs_score

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_cases.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text())


def test_golden_covers_exactly_the_case_matrix():
    assert set(GOLDEN["cases"]) == {case["name"] for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_full_output_matches_golden(case):
    kwargs = build_call_kwargs(case)
    output = compute_vcs_score(**kwargs, return_all_metrics=True, return_internals=True)
    expected = GOLDEN["cases"][case["name"]]["output"]
    assert_structurally_equal(canonicalize(output), expected, atol=1e-12)


def test_single_chunk_identical_scores_one():
    """m = n = 1: every alignment window spans the whole other side, so the
    Global-NAS max-penalty normalizer is vacuously 0 — no match can deviate
    from chronology, and v3 scores that as vacuously perfect (1.0).
    v1/v2 scored it 0.0, zeroing VCS for identical single-sentence texts
    (the old M1 pathology, fixed in the v3 math batch)."""
    case = next(c for c in CASES if c["name"] == "single_chunk_identical")
    output = compute_vcs_score(**build_call_kwargs(case), return_all_metrics=True)
    assert output["Global NAS"] == 1.0
    assert output["VCS"] == pytest.approx(1.0, abs=1e-12)
    assert output["Global_SAS"] == pytest.approx(1.0, abs=1e-12)
    assert output["Local_SAS"] == pytest.approx(1.0, abs=1e-12)


def test_identical_texts_score_one():
    case = next(c for c in CASES if c["name"] == "identical_4")
    output = compute_vcs_score(**build_call_kwargs(case))
    assert output["VCS"] == pytest.approx(1.0, abs=1e-12)
