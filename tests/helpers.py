"""Canonicalization and structural comparison used by the golden tests.

The library's output tree currently leaks numpy scalar types (np.int64 in
match tuples, np.float64 from np.sum reductions) and tuples. Goldens are
stored as plain JSON, so both sides are canonicalized to pure-Python
JSON-compatible values before comparison. Changing the LIBRARY's output
types is out of scope for v2.0.0 (see plan risk #5) — this hook is where
that leak is absorbed.
"""
import numpy as np


def canonicalize(obj):
    """Recursively convert an output tree to JSON-compatible pure Python:
    numpy scalars -> int/float/bool, arrays -> lists, tuples -> lists."""
    if isinstance(obj, dict):
        return {key: canonicalize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return [canonicalize(item) for item in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def assert_structurally_equal(actual, expected, atol=1e-12, path="$"):
    """Walk two canonicalized trees: identical keys/lengths/types everywhere,
    floats compared at ``atol``, everything else exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        assert not missing and not extra, f"{path}: missing keys {sorted(missing)}, unexpected keys {sorted(extra)}"
        for key in expected:
            assert_structurally_equal(actual[key], expected[key], atol, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_structurally_equal(a, e, atol, f"{path}[{i}]")
    elif isinstance(expected, bool):
        assert isinstance(actual, bool) and actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
            f"{path}: expected number, got {type(actual).__name__}"
        )
        if isinstance(expected, int) and isinstance(actual, int):
            assert actual == expected, f"{path}: {actual} != {expected}"
        else:
            assert abs(actual - expected) <= atol, (
                f"{path}: |{actual!r} - {expected!r}| = {abs(actual - expected):.3e} > {atol}"
            )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
