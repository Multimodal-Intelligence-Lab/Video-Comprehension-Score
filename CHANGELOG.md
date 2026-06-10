# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

### Added
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