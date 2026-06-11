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


def test_known_pathology_single_chunk_identical_scores_zero():
    """m = n = 1: the Global-NAS max-penalty normalizer is vacuously 0, so
    Global NAS = 0 and VCS = 0.0 even for IDENTICAL texts, while Global_SAS
    and Local_SAS are ~1. Pinned deliberately as v1 behavior — math-phase
    backlog item M1, not to be 'fixed' in the software overhaul."""
    case = next(c for c in CASES if c["name"] == "single_chunk_identical")
    output = compute_vcs_score(**build_call_kwargs(case), return_all_metrics=True)
    assert output["VCS"] == 0.0
    assert output["Global NAS"] == 0.0
    assert output["Global_SAS"] == pytest.approx(1.0, abs=1e-12)
    assert output["Local_SAS"] == pytest.approx(1.0, abs=1e-12)


def test_identical_texts_score_one():
    case = next(c for c in CASES if c["name"] == "identical_4")
    output = compute_vcs_score(**build_call_kwargs(case))
    assert output["VCS"] == pytest.approx(1.0, abs=1e-12)
