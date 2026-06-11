"""Classify the diff between two golden_cases.json files for review.

Usage (review an intentional golden regeneration before committing):

    git show HEAD:tests/golden/golden_cases.json > /tmp/golden_old.json
    python tests/golden/generate_golden.py
    python tests/golden/classify_diff.py /tmp/golden_old.json tests/golden/golden_cases.json

Walks both JSON trees, compares leaves EXACTLY (stricter than the test
suite's atol=1e-12 on purpose: every ulp must be explained), and buckets
every changed/added/removed leaf by (case, path with list indices
normalized to [*]). Review = check each printed bucket against the
commit's MUST / MUST-NOT lists.
"""
import json
import re
import sys


def _walk(node, path, leaves):
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}", leaves)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", leaves)
    else:
        leaves[path] = node


def _bucket_key(path):
    parts = path.split(".")
    case = parts[2] if len(parts) > 2 and parts[1] == "cases" else "<header>"
    return case, re.sub(r"\[\d+\]", "[*]", path)


def main(old_path, new_path):
    old, new = {}, {}
    _walk(json.loads(open(old_path).read()), "", old)
    _walk(json.loads(open(new_path).read()), "", new)

    buckets = {}
    for path in sorted(old.keys() | new.keys()):
        if path in old and path in new:
            if old[path] != new[path]:
                kind = "changed"
            else:
                continue
        else:
            kind = "removed" if path in old else "added"
        case, pattern = _bucket_key(path)
        buckets.setdefault((kind, case, pattern), []).append(
            (path, old.get(path), new.get(path))
        )

    if not buckets:
        print("no changed, added, or removed leaves")
        return
    for (kind, case, pattern), items in sorted(buckets.items()):
        print(f"{kind:<8} {case:<28} {pattern}  x{len(items)}")
        for path, before, after in items[:3]:
            print(f"         {path}: {before!r} -> {after!r}")
        if len(items) > 3:
            print(f"         ... {len(items) - 3} more")
    total = sum(len(items) for items in buckets.values())
    print(f"\n{total} leaves in {len(buckets)} buckets")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
