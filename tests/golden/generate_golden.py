"""Regenerate tests/golden/golden_cases.json from the CURRENT library.

Run this ONLY when an intentional output change is made (e.g. plan C4),
then review the JSON diff line by line before committing:

    python tests/golden/generate_golden.py
    git diff tests/golden/golden_cases.json
"""
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

from cases import CASES, build_call_kwargs  # noqa: E402
from helpers import canonicalize  # noqa: E402

import vcs  # noqa: E402


def main():
    payload = {
        "generated_against_vcs_version": vcs.__version__,
        "comparison": "floats at atol=1e-12, everything else exact (see helpers.py)",
        "cases": {},
    }
    for case in CASES:
        kwargs = build_call_kwargs(case)
        output = vcs.compute_vcs_score(
            **kwargs, return_all_metrics=True, return_internals=True
        )
        config = {key: value for key, value in kwargs.items()
                  if key not in ("reference_text", "generated_text",
                                 "segmenter_fn", "embedding_fn_global_sas",
                                 "embedding_fn_local_sas")}
        payload["cases"][case["name"]] = {
            "config_overrides": config,
            "output": canonicalize(output),
        }

    target = Path(__file__).with_name("golden_cases.json")
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
