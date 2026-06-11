# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-06-10

The first release that changes metric values. Every change below was
golden-gated: each regeneration diff was classified leaf by leaf and
matched against the change's predicted footprint.

### Changed (BREAKING)
- **Single `embedding_fn` parameter.** `compute_vcs_score`'s 4th
  positional parameter is now `embedding_fn`; `embedding_fn_global_sas`
  and `embedding_fn_local_sas` are gone. One instrument measures both
  SAS levels — the SAS combination `max(0, GAS + LAS - 1) / LAS` reads
  GAS and LAS as the same quantity at two granularities, which two
  different embedders silently break. Migration: rename the keyword,
  drop the second embedder. Positional callers keep working.
  `compute_vcs_from_embeddings` documents the same one-instrument
  contract (doc + chunk embeddings from the same embedder).
- **Embedding rows are L2-normalized internally** (replaces the v2
  `UserWarning`). Rows already within `1e-12` of unit norm pass through
  bit-untouched (the same tensor object, so compliant float64 inputs
  cannot change results); other rows are renormalized in the embedding
  dtype; rows with L2 norm 0 now raise `ValueError`. Scores are
  scale-invariant per embedding row. Note: float32 rows normalized by
  the caller rarely measure within `1e-12` in float64 and are simply
  renormalized — results are identical either way.
- **Chunk similarities are clamped at 0.** Anti-correlated chunks carry
  no more alignment signal than unrelated ones, and a negative
  similarity no longer acts as less-than-zero evidence in LAS averages
  or load-sharing coverage. Consequence: a chunk with NO positive signal
  (all similarities <= 0) now matches inside its alignment window at
  similarity 0.0 with no chronology penalty, instead of crowning its
  least-negative cell wherever it sat. Document-level GAS is
  deliberately NOT clamped — negative document cosines feed the SAS
  gate. (Edge note: at extreme knob values, `context_cutoff_value`
  near 0 with a small `context_window_control`, zero-similarity cells
  can enter candidate sets that negative cells previously stayed out
  of; at default and typical knobs candidate sets are unchanged.)
- **`min(m, n) == 1` grids score Global NAS as vacuously perfect.**
  When every alignment window spans the whole other side (single-chunk
  texts, m-vs-1 grids), the max-penalty normalizer is 0 and no match
  can deviate from chronology; that now scores 1.0 instead of 0.0.
  Identical single-sentence texts score VCS = 1.0 (was 0.0). Content
  judgment is unaffected: an m-vs-1 generation that drops content still
  dies at the SAS gate.
- **Local NAS uses the exact per-step in-band bound.** The old global
  closed form (`2h-1` / `2h-2` from the axis ratio) applied one jump
  threshold to every step; the bound is now derived per step from the
  alignment windows: `(end_window(x_next) - 1) - start_window(x_curr)`.
  Values move in non-square geometries (e.g. a dy=1 recall step at a
  bound-0 position loses its credit). The `threshold` /
  `threshold_with_Rn` internals keys keep their shape but now vary per
  step. With `Rn > 0`, `|dy|` is compared against the forward bound (as
  before, backward jumps are graded against the forward allowance).

### Added
- **`"VCS Margin"`** (under `return_all_metrics`): `SAS + NAS - 1` in
  `[-1, 1]` — the shared numerator of both VCS scaling branches, so
  `VCS > 0` iff margin > 0 and `VCS == margin / max(SAS, NAS)` when
  positive. Unlike VCS it is not clamped, so it keeps ranking
  candidates that the VCS zero gate maps to a flat 0. Also exposed as
  `internals["metrics"]["vcs"]["margin"]`.
- **`"Config"`** (under `return_all_metrics`): a provenance string
  `vcs=<version>|chunk_size=...|rn=...|context_cutoff=...|context_window_control=...`
  covering the library version and every knob the library controls
  (the embedder's identity is the caller's to report). Reads
  `chunk_size=none` from `compute_vcs_from_embeddings`. Because the
  string embeds the package version, every release touches the golden
  file by exactly that substring — an accepted consequence.
- A metric-semantics test suite (`tests/test_metric_semantics.py`)
  pinning the new behaviors with exact one-hot constructions, plus a
  geometric property test that recomputes every Local-NAS threshold
  from the alignment windows.

### Removed
- The `dual_embedders` golden case and the dim48 test embedder (the
  dual-embedder API is gone).

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