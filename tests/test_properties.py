"""Empirically verified invariants of v1 behavior.

Each property below was checked against the full 20-case matrix before
being asserted (plan C1): determinism is bit-exact, ref<->gen swap is
EXACTLY symmetric (precision and recall exchange roles, F1 and the
Lukasiewicz combiners are symmetric), and metrics stay within [0, 1] up
to one float ulp (cosine of identical unit vectors can round to
1.0000000000000002, which propagates into Local_SAS).
"""
import pytest

from cases import CASES, build_call_kwargs
from helpers import canonicalize
from vcs import compute_vcs_score

METRIC_BOUND_EPS = 1e-12

SWAP_KEY = {
    "VCS": "VCS",
    "VCS Margin": "VCS Margin",
    "Config": "Config",
    "Global_SAS": "Global_SAS",
    "Local_SAS": "Local_SAS",
    "SAS": "SAS",
    "NAS": "NAS",
    "Global NAS": "Global NAS",
    "Local NAS": "Local NAS",
    "Precision Local_SAS": "Recall Local_SAS",
    "Recall Local_SAS": "Precision Local_SAS",
    "Precision Global NAS": "Recall Global NAS",
    "Recall Global NAS": "Precision Global NAS",
    "Precision Local NAS": "Recall Local NAS",
    "Recall Local NAS": "Precision Local NAS",
}


@pytest.mark.parametrize("name", [
    "typical_8v3_defaults", "duplication_load_sharing", "rn_1", "context_tight",
])
def test_determinism_is_bit_exact(name):
    kwargs = build_call_kwargs(next(c for c in CASES if c["name"] == name))
    first = compute_vcs_score(**kwargs, return_all_metrics=True, return_internals=True)
    second = compute_vcs_score(**kwargs, return_all_metrics=True, return_internals=True)
    assert canonicalize(first) == canonicalize(second)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_metrics_bounded(case):
    output = compute_vcs_score(**build_call_kwargs(case), return_all_metrics=True)
    for key, value in output.items():
        if key == "Config":  # str, not a metric
            continue
        # the margin is the one deliberately signed output: SAS + NAS - 1
        lower = -1.0 if key == "VCS Margin" else 0.0
        assert lower - METRIC_BOUND_EPS <= value <= 1.0 + METRIC_BOUND_EPS, (
            f"{case['name']}: {key} = {value!r} outside [{lower}, 1]"
        )


@pytest.mark.parametrize("name", [
    "typical_8v3_defaults", "square_4v4_defaults", "disjoint_4",
    "half_omission_8v4", "reordered_8v8",
])
def test_swapping_texts_swaps_precision_and_recall_exactly(name):
    case = next(c for c in CASES if c["name"] == name)
    kwargs = build_call_kwargs(case)
    swapped = dict(kwargs)
    swapped["reference_text"] = kwargs["generated_text"]
    swapped["generated_text"] = kwargs["reference_text"]
    forward = compute_vcs_score(**kwargs, return_all_metrics=True)
    backward = compute_vcs_score(**swapped, return_all_metrics=True)
    for key, mirror in SWAP_KEY.items():
        assert forward[key] == backward[mirror], (
            f"{key}: {forward[key]!r} != swapped {mirror}: {backward[mirror]!r}"
        )
