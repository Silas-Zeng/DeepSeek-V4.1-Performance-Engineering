from pathlib import Path

import scripts.tutorial_cpu_profiler as tutorial


def test_resolve_output_path_uses_repo_root_for_relative_paths():
    result = tutorial.resolve_output_path("profiles/test/indexer_cpu_trace.json")
    assert result == tutorial.REPO_ROOT / "profiles/test/indexer_cpu_trace.json"
    assert result.is_absolute()


def test_default_output_path_points_to_repo_profiles_directory():
    result = tutorial.resolve_output_path()
    assert result == tutorial.REPO_ROOT / "profiles/cpu-tutorial/indexer_cpu_trace.json"
    assert result.parent == tutorial.REPO_ROOT / "profiles/cpu-tutorial"
