"""Fuzz equivalence of rewritten hot paths against the frozen v1 originals
(tests/reference_impls.py). Exact equality required - these rewrites claim
identical OUTPUT, not approximate behavior (plan C12/C13).
"""
import numpy as np
import pytest

import reference_impls as v1
from vcs._windows import _get_alignment_windows as live_windows
from vcs._nas import _calculate_actual_penalty as live_actual_penalty


def _norm(windows):
    """Tuple values may be np.int64 (direct) or int (reverse); compare values."""
    return [(int(a), int(b)) for a, b in windows]


# (ref_len, gen_len) sweep: every adversarial shape class we could think of,
# plus a seeded random sweep.
ADVERSARIAL = (
    [(1, 1), (1, 2), (2, 1), (1, 50), (50, 1), (2, 3), (3, 2)]
    + [(k, k) for k in (2, 3, 7, 64)]
    + [(k, k + 1) for k in (1, 9, 99)] + [(k + 1, k) for k in (1, 9, 99)]
    + [(97, 13), (13, 97), (101, 7), (7, 101), (400, 399), (399, 400),
       (256, 3), (3, 256), (360, 240), (240, 360)]
)

_rng = np.random.default_rng(20260610)
RANDOM_PAIRS = [tuple(map(int, _rng.integers(1, 201, size=2))) for _ in range(800)]
RANDOM_PAIRS += [tuple(map(int, _rng.integers(1, 401, size=2))) for _ in range(25)]


@pytest.mark.parametrize("ref_len,gen_len", ADVERSARIAL)
def test_alignment_windows_equal_v1_adversarial(ref_len, gen_len):
    live_p, live_r = live_windows(ref_len, gen_len)
    v1_p, v1_r = v1._get_alignment_windows(ref_len, gen_len)
    assert _norm(live_p) == _norm(v1_p)
    assert _norm(live_r) == _norm(v1_r)


def test_alignment_windows_equal_v1_random_sweep():
    for ref_len, gen_len in RANDOM_PAIRS:
        live_p, live_r = live_windows(ref_len, gen_len)
        v1_p, v1_r = v1._get_alignment_windows(ref_len, gen_len)
        assert _norm(live_p) == _norm(v1_p), (ref_len, gen_len)
        assert _norm(live_r) == _norm(v1_r), (ref_len, gen_len)


def test_actual_penalty_equal_v1_fuzz():
    rng = np.random.default_rng(31415)
    for _ in range(600):
        n = int(rng.integers(1, 120))
        y_len = int(rng.integers(1, 120))
        # precision windows: one per generated element (n of them), each a
        # (start, end) range over the reference axis [0, y_len)
        windows, _ = v1._get_alignment_windows(y_len, n)
        best = rng.integers(-1, y_len, size=n)
        Rn = int(rng.integers(0, 4))
        direction = "precision" if rng.integers(0, 2) else "recall"

        live_pen, live_int = live_actual_penalty(
            best, windows, y_len, direction, Rn, y_len, n, collect_details=True
        )
        v1_pen, v1_int = v1.calculate_actual_penalty(
            best, windows, y_len, direction, Rn, y_len, n
        )
        assert live_pen.dtype == v1_pen.dtype
        assert np.array_equal(live_pen, v1_pen)
        # v1 also exported the dead alignment_window_height key (removed in C4)
        for key in ("penalties", "in_window", "in_rn_zone"):
            assert live_int[key] == v1_int[key]
