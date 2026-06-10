"""Package-level import and metadata contract.

Deliberately does NOT enumerate the visualization exports: they are removed
in v2.0.0 (plan C6), and pinning them here would only create churn. The
assertions below are the surface that must survive the overhaul.
"""
import vcs


def test_core_entry_point_exported():
    assert callable(vcs.compute_vcs_score)
    assert "compute_vcs_score" in vcs.__all__


def test_config_constants_exported_with_expected_values():
    assert vcs.DEFAULT_CONTEXT_CUTOFF_VALUE == 0.6
    assert vcs.DEFAULT_CONTEXT_WINDOW_CONTROL == 4.0
    assert vcs.DEFAULT_Rn == 0
    assert vcs.DEFAULT_CHUNK_SIZE == 1
    for name in (
        "DEFAULT_CONTEXT_CUTOFF_VALUE",
        "DEFAULT_CONTEXT_WINDOW_CONTROL",
        "DEFAULT_Rn",
        "DEFAULT_CHUNK_SIZE",
    ):
        assert name in vcs.__all__


def test_version_metadata_present():
    assert isinstance(vcs.__version__, str) and vcs.__version__
    assert isinstance(vcs.__author__, str) and vcs.__author__


def test_no_visualization_stack_imported():
    """v2.0.0 removed the visualization subtree; importing vcs must not pull
    in matplotlib/seaborn (they are no longer dependencies at all)."""
    import sys

    assert "matplotlib" not in sys.modules
    assert "seaborn" not in sys.modules
    assert not any(name.startswith("vcs._visualize") for name in sys.modules)


def test_no_visualization_exports():
    assert not [name for name in vcs.__all__ if "visualize" in name or "pdf" in name]
    assert not hasattr(vcs, "visualize_metrics_summary")
