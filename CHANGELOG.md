# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-06-10

### Changed
- Package internals flattened: 23 nested packages (max depth 9, one
  function per folder) are now 10 flat modules. All moved modules are
  private; anything that imported `vcs._metrics.*` or other internal
  paths must update (the public API is unchanged).
- Performance: internals details (per-candidate selection dicts, segment
  dicts, list exports) are built only when `return_internals=True`
  (~8x lower peak allocations on the minimal path at 400x300 chunks);
  reverse alignment-window construction is O(n log n) instead of
  O(m*n); the Global-NAS penalty passes are vectorized. All rewrites
  are fuzz-verified bit-identical against the v1 implementations.
- `pyproject.toml` version is now the single source of truth for
  releases: publish.yml refuses to publish a mismatched version (the old
  flow sed-edited the version into the build) and tags only after a
  successful PyPI publish.
- Sphinx docs read their version from package metadata (was hardcoded
  1.0.2 while the package said 1.0.0).
- The package ships a `py.typed` marker: type checkers now use the
  inline annotations.

### Removed (BREAKING)
- The entire visualization suite: the 12 `visualize_*` functions and
  `create_vcs_pdf_report` are gone, along with the matplotlib and seaborn
  dependencies. The metric itself is untouched. If you need the v1 plots,
  pin `video-comprehension-score<2` or use the `legacy/v1-with-visualization`
  branch.

### Changed (BREAKING)
- `torch>=2.0` is now a declared dependency (v1 imported torch but never
  declared it, so `pip install video-comprehension-score` produced a
  broken install in clean environments). Note: pip resolves torch>=2.2 on
  Python 3.12 and >=2.6 on Python 3.13.
- Supported Python versions are now 3.10–3.13 (v1 metadata claimed 3.8+
  but the code uses `X | None` syntax, which crashes below 3.10).

### Added
- New public entry point `compute_vcs_from_embeddings(...)`: computes VCS
  directly from pre-computed document and chunk embeddings (for batch
  pipelines, cached embeddings, or sweeping VCS knobs without
  re-embedding). Given the embeddings the text entry point would produce,
  results are exactly equal. Chunk texts are optional (placeholders appear
  in internals); `internals["config"]["chunk_size"]` is `None` here.
- Input validation on all entry points: clear `ValueError`s before any
  computation for invalid texts, parameters, segmenter output, or
  embedding shapes, and a `UserWarning` when embedding rows are not
  L2-normalized (VCS similarities are raw dot products).
- `return_all_metrics=True` now includes `"Precision Local_SAS"` and
  `"Recall Local_SAS"`, as the `compute_vcs_score` docstring always
  promised (v1 computed but dropped them).
- `internals["metrics"]["sas"]["local_sas_internals"]` is now populated
  with the Local SAS load-sharing details (v1 always returned `{}` there).
- Offline characterization test suite (`tests/`) with golden outputs.

### Removed
- `internals["metrics"]["global_nas"][...]["alignment_window_height"]`:
  the value was computed by dead code and never used by any metric.

## [1.0.0] - 2024-12-19

### Added
- Initial production release of VCS Metrics library
- Core VCS (Video Comprehension Score) computation with all features
- Complete visualization suite and PDF report generation
- Production-ready API and documentation

### Changed
- Promoted to production status (1.0.0)
- Finalized API for stable release

### Removed
- Removed test suite (tests folder) as library is now production-ready