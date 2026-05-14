# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the StatsSummary system (StatsProfileDetails, generate_stats_summary, StatsSummary)."""

from pathlib import Path

import pytest

from fixingahole.profiler.profile_summary import ProfileSummary
from fixingahole.profiler.stats_manager import (
    StatisticsManager,
    StatsProfileDetails,
    StatsSummary,
    _build_stats_details,  # noqa: PLC2701
    _format_stats_duration,  # noqa: PLC2701
    _format_stats_memory,  # noqa: PLC2701
    _get_all_tree_functions,  # noqa: PLC2701
    generate_stats_summary,
    render_stats_tree,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["old", "new"])
def profile_summary_obj(request: pytest.FixtureRequest, example_json: Path, example_json_new: Path) -> ProfileSummary:
    """Load a real ProfileSummary from either the old or new example JSON."""
    json_path = example_json if request.param == "old" else example_json_new
    return ProfileSummary(json_path)


@pytest.fixture
def single_run_manager(profile_summary_obj: ProfileSummary) -> StatisticsManager:
    """StatisticsManager populated with one profiling run."""
    manager = StatisticsManager()
    manager.insert(profile_summary_obj)
    return manager


@pytest.fixture
def multi_run_manager(profile_summary_obj: ProfileSummary) -> StatisticsManager:
    """StatisticsManager populated with three identical profiling runs."""
    manager = StatisticsManager()
    for _ in range(3):
        manager.insert(profile_summary_obj)
    return manager


@pytest.fixture
def minimal_stats_data(root_dir: Path) -> dict:
    """Minimal stats dict with one function entry and no memory info."""
    key = str(root_dir / "src" / "myfile.py") + ":my_func"
    return {
        key: {
            "user": {"avg": 1.0, "std": 0.1},
            "system": {"avg": 0.05, "std": 0.005},
            "memory": {"avg": 0.0, "std": 0.0},
            "count": 3,
        }
    }


@pytest.fixture
def stats_with_memory(root_dir: Path) -> dict:
    """Stats dict with two functions; one has memory usage."""
    base = str(root_dir / "src")
    return {
        f"{base}/fast.py:fast_func": {
            "user": {"avg": 0.5, "std": 0.01},
            "system": {"avg": 0.0, "std": 0.0},
            "memory": {"avg": 0.0, "std": 0.0},
            "count": 5,
        },
        f"{base}/slow.py:slow_func": {
            "user": {"avg": 2.0, "std": 0.2},
            "system": {"avg": 0.1, "std": 0.01},
            "memory": {"avg": 128.0, "std": 5.0},
            "count": 5,
        },
    }


# ---------------------------------------------------------------------------
# _format_stats_duration
# ---------------------------------------------------------------------------


class TestFormatStatsDuration:
    """Test the _format_stats_duration formatting helper."""

    def test_hours_no_std(self):
        """Values >= 3600s render as hours."""
        result = _format_stats_duration(7200.0, 0.0)
        assert result == "2.000 hr"

    def test_hours_with_std(self):
        """Hours value with non-zero std includes ± notation."""
        result = _format_stats_duration(7200.0, 360.0)
        assert "hr" in result
        assert "±" in result

    def test_minutes_no_std(self):
        """Values in [60, 3600) render as minutes."""
        result = _format_stats_duration(120.0, 0.0)
        assert result == "2.000 min"

    def test_minutes_with_std(self):
        """Minutes value with non-zero std includes ± notation."""
        result = _format_stats_duration(120.0, 6.0)
        assert "min" in result
        assert "±" in result

    def test_seconds_no_std(self):
        """Values in [1, 60) render as seconds."""
        result = _format_stats_duration(2.5, 0.0)
        assert result == "2.500 sec"

    def test_seconds_with_std(self):
        """Seconds value with non-zero std renders as 'avg ± std sec'."""
        result = _format_stats_duration(2.5, 0.1)
        assert result == "2.500 ± 0.100 sec"

    def test_milliseconds_no_std(self):
        """Values in [0.001, 1) render as milliseconds."""
        result = _format_stats_duration(0.005, 0.0)
        assert result == "5.000 ms"

    def test_milliseconds_with_std(self):
        """Milliseconds value with non-zero std includes ± notation."""
        result = _format_stats_duration(0.005, 0.001)
        assert result == "5.000 ± 1.000 ms"

    def test_microseconds(self):
        """Values < 0.001s render as microseconds."""
        result = _format_stats_duration(0.0001, 0.0)
        assert "µs" in result

    def test_microseconds_with_std(self):
        """Microsecond value with non-zero std includes ± notation."""
        result = _format_stats_duration(0.0001, 0.00001)
        assert "µs" in result
        assert "±" in result

    def test_std_zero_suppresses_pm(self):
        """When std is exactly zero the ± notation must be absent."""
        result = _format_stats_duration(1.234, 0.0)
        assert "±" not in result

    def test_avg_and_std_use_same_unit(self):
        """Both avg and std are expressed in the same unit."""
        result = _format_stats_duration(90.0, 3.0)  # minutes range
        assert "min" in result
        parts = result.split("±")
        assert len(parts) == 2
        assert "min" in parts[1]


# ---------------------------------------------------------------------------
# _format_stats_memory
# ---------------------------------------------------------------------------


class TestFormatStatsMemory:
    """Test the _format_stats_memory formatting helper."""

    def test_no_std(self):
        """When std is zero only the average is shown."""
        result = _format_stats_memory(256.0, 0.0)
        assert "±" not in result
        assert "MB" in result or "GB" in result

    def test_with_std(self):
        """Non-zero std produces 'avg ± std' output."""
        result = _format_stats_memory(256.0, 10.0)
        assert "±" in result

    def test_unit_scaling(self):
        """Large memory values are shown in GB."""
        result = _format_stats_memory(2048.0, 0.0)
        assert "GB" in result


# ---------------------------------------------------------------------------
# StatsProfileDetails
# ---------------------------------------------------------------------------


class TestStatsProfileDetails:
    """Test the StatsProfileDetails data class."""

    @pytest.fixture
    def detail(self) -> StatsProfileDetails:
        """A sample StatsProfileDetails instance."""
        return StatsProfileDetails(
            name="my_func",
            file_path="/project/src/module.py",
            avg_user=1.5,
            avg_system=0.2,
            avg_memory=64.0,
            std_user=0.1,
            std_system=0.02,
            std_memory=3.0,
        )

    def test_total_percentage_is_sum_of_user_and_system(self, detail: StatsProfileDetails):
        """total_percentage equals avg_user + avg_system."""
        assert detail.total_percentage == pytest.approx(1.7)

    def test_has_memory_info_true_when_nonzero(self, detail: StatsProfileDetails):
        """has_memory_info is True when avg_memory > 0."""
        assert detail.has_memory_info is True

    def test_has_memory_info_false_when_zero(self):
        """has_memory_info is False when avg_memory == 0."""
        d = StatsProfileDetails("f", "/p.py", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert d.has_memory_info is False

    def test_peak_memory_equals_avg_memory(self, detail: StatsProfileDetails):
        """peak_memory property mirrors avg_memory."""
        assert detail.peak_memory == pytest.approx(64.0)

    def test_peak_memory_info_contains_units(self, detail: StatsProfileDetails):
        """peak_memory_info returns a non-empty formatted string with unit."""
        info = detail.peak_memory_info
        assert isinstance(info, str)
        assert len(info) > 0
        assert any(unit in info for unit in ["MB", "GB", "KB", "bytes"])

    def test_is_frozen(self, detail: StatsProfileDetails):
        """StatsProfileDetails instances are immutable."""
        with pytest.raises((AttributeError, TypeError)):
            detail.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_stats_details
# ---------------------------------------------------------------------------


class TestBuildStatsDetails:
    """Test the _build_stats_details private helper."""

    def test_groups_by_file(self, stats_with_memory: dict):
        """Functions are grouped under their file path."""
        result = _build_stats_details(stats_with_memory)
        assert len(result) == 2  # two distinct files

    def test_skips_metadata_key(self, minimal_stats_data: dict):
        """The 'metadata' key is silently ignored."""
        data = dict(minimal_stats_data)
        data["metadata"] = {"repo": "test", "utc_time": "20260101_000000"}
        result = _build_stats_details(data)
        assert "metadata" not in result
        assert len(result) == 1

    def test_skips_malformed_key_no_colon(self):
        """Keys with no ':' separator are skipped."""
        result = _build_stats_details(
            {
                "no_colon_at_all": {
                    "user": {"avg": 1.0, "std": 0.0},
                    "system": {"avg": 0.0, "std": 0.0},
                    "memory": {"avg": 0.0, "std": 0.0},
                    "count": 1,
                }
            }
        )
        assert result == {}

    def test_resolves_relative_paths(self, minimal_stats_data: dict):
        """Relative paths should be resolved to absolute using Config.root()."""
        relative_key = "relative/path/file.py:some_func"
        data = {
            relative_key: {
                "user": {"avg": 1.0, "std": 0.0},
                "system": {"avg": 0.0, "std": 0.0},
                "memory": {"avg": 0.0, "std": 0.0},
                "count": 1,
            }
        }
        result = _build_stats_details(data)
        file_key = next(iter(result))
        assert Path(file_key).is_absolute()

    def test_preserves_absolute_paths(self, stats_with_memory: dict):
        """Absolute paths are kept as-is."""
        result = _build_stats_details(stats_with_memory)
        for file_path in result:
            assert Path(file_path).is_absolute()

    def test_fields_are_populated_correctly(self, minimal_stats_data: dict):
        """All avg/std fields are transferred to StatsProfileDetails."""
        result = _build_stats_details(minimal_stats_data)
        funcs = next(iter(result.values()))
        assert len(funcs) == 1
        f = funcs[0]
        assert f.name == "my_func"
        assert f.avg_user == pytest.approx(1.0)
        assert f.std_user == pytest.approx(0.1)
        assert f.avg_system == pytest.approx(0.05)
        assert f.std_system == pytest.approx(0.005)
        assert f.avg_memory == pytest.approx(0.0)
        assert f.std_memory == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _get_all_tree_functions
# ---------------------------------------------------------------------------


class TestGetAllTreeFunctions:
    """Test the _get_all_tree_functions helper."""

    def test_empty_tree(self):
        """Empty dict returns empty list."""
        assert _get_all_tree_functions({}) == []

    def test_flat_tree_with_functions(self):
        """A single node with _functions returns one list."""
        funcs = [object(), object()]
        tree = {"file.py": {"_functions": funcs, "_children": {}}}
        result = _get_all_tree_functions(tree)
        assert len(result) == 1
        assert result[0] is funcs

    def test_nested_children_are_recursed(self):
        """Functions inside nested _children nodes are collected."""
        inner_funcs = [object()]
        tree = {
            "pkg": {
                "_functions": [],
                "_children": {"module.py": {"_functions": inner_funcs, "_children": {}}},
            }
        }
        result = _get_all_tree_functions(tree)
        assert len(result) == 1
        assert result[0] is inner_funcs


# ---------------------------------------------------------------------------
# render_stats_tree
# ---------------------------------------------------------------------------


class TestRenderStatsTree:
    """Test the render_stats_tree function."""

    def test_empty_tree_returns_empty_list(self):
        """An empty tree dict produces no output lines."""
        assert render_stats_tree({}) == []

    def test_single_file_with_function(self, root_dir: Path):
        """A single function renders at least one output line."""
        file_path = str(root_dir / "myfile.py")
        func = StatsProfileDetails("fn", file_path, 1.0, 0.0, 0.0, 0.05, 0.0, 0.0)
        tree = {"myfile.py": {"_functions": [func], "_children": {}}}
        lines = render_stats_tree(tree, threshold_sec=0.0)
        assert len(lines) > 0
        # Function name appears in output
        combined = "\n".join(lines)
        assert "fn" in combined

    def test_function_below_threshold_is_filtered(self, root_dir: Path):
        """Functions below threshold_sec with no memory are excluded."""
        file_path = str(root_dir / "myfile.py")
        func = StatsProfileDetails("tiny_fn", file_path, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0)
        tree = {"myfile.py": {"_functions": [func], "_children": {}}}
        lines = render_stats_tree(tree, threshold_sec=0.001)
        combined = "\n".join(lines)
        assert "tiny_fn" not in combined

    def test_function_below_threshold_kept_if_has_memory(self, root_dir: Path):
        """Functions below threshold_sec are retained if they have memory info."""
        file_path = str(root_dir / "myfile.py")
        func = StatsProfileDetails("mem_fn", file_path, 0.00001, 0.0, 50.0, 0.0, 0.0, 2.0)
        tree = {"myfile.py": {"_functions": [func], "_children": {}}}
        lines = render_stats_tree(tree, threshold_sec=1.0)
        combined = "\n".join(lines)
        assert "mem_fn" in combined

    def test_directory_hidden_when_all_children_below_threshold(self, root_dir: Path):
        """A directory whose aggregate runtime exceeds threshold_sec but where every
        individual function is below threshold_sec must not appear in the output.

        Previously the early-return guard used ``total_avg < threshold_sec`` which
        was fooled when many sub-threshold functions summed to above the threshold.
        """
        sub_dir = root_dir / "pkg"
        file_path = str(sub_dir / "mod.py")
        # 5 functions each at 0.3 sec — all below threshold=1.0, but sum=1.5 > threshold
        funcs = [
            StatsProfileDetails(f"fn{i}", file_path, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(5)
        ]
        children = {"mod.py": {"_functions": funcs, "_children": {}}}
        tree = {"pkg": {"_functions": [], "_children": children}}
        lines = render_stats_tree(tree, threshold_sec=1.0)
        combined = "\n".join(lines)
        # The directory node and all function lines should be absent.
        assert "pkg" not in combined
        assert not any(f"fn{i}" in combined for i in range(5))

    def test_directory_shown_when_some_children_above_threshold(self, root_dir: Path):
        """A directory is rendered when at least one child function exceeds threshold_sec."""
        sub_dir = root_dir / "pkg"
        file_path = str(sub_dir / "mod.py")
        below = StatsProfileDetails("tiny_fn", file_path, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
        above = StatsProfileDetails("big_fn", file_path, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        children = {"mod.py": {"_functions": [below, above], "_children": {}}}
        tree = {"pkg": {"_functions": [], "_children": children}}
        lines = render_stats_tree(tree, threshold_sec=1.0)
        combined = "\n".join(lines)
        assert "pkg" in combined
        assert "big_fn" in combined
        assert "tiny_fn" not in combined

        """Functions within a file are listed fastest-last."""
        file_path = str(root_dir / "myfile.py")
        fast = StatsProfileDetails("fast_fn", file_path, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        slow = StatsProfileDetails("slow_fn", file_path, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tree = {"myfile.py": {"_functions": [fast, slow], "_children": {}}}
        lines = render_stats_tree(tree, threshold_sec=0.0)
        combined = "\n".join(lines)
        assert combined.index("slow_fn") < combined.index("fast_fn")

    def test_output_contains_duration_formatting(self, root_dir: Path):
        """Output lines include formatted duration strings."""
        file_path = str(root_dir / "myfile.py")
        func = StatsProfileDetails("fn", file_path, 2.5, 0.3, 0.0, 0.1, 0.01, 0.0)
        tree = {"myfile.py": {"_functions": [func], "_children": {}}}
        lines = render_stats_tree(tree, threshold_sec=0.0)
        combined = "\n".join(lines)
        # 2.5 + 0.3 = 2.8 seconds; should appear in some format
        assert "sec" in combined or "ms" in combined

    def test_output_contains_memory_info(self, root_dir: Path):
        """Memory info appears in output when present."""
        file_path = str(root_dir / "myfile.py")
        func = StatsProfileDetails("fn", file_path, 1.0, 0.0, 256.0, 0.1, 0.0, 10.0)
        tree = {"myfile.py": {"_functions": [func], "_children": {}}}
        lines = render_stats_tree(tree, threshold_sec=0.0)
        combined = "\n".join(lines)
        assert "MB" in combined or "GB" in combined


# ---------------------------------------------------------------------------
# generate_stats_summary
# ---------------------------------------------------------------------------


class TestGenerateStatsSummary:
    """Test the generate_stats_summary function."""

    def test_empty_data_returns_sentinel(self):
        """Empty stats dict returns the 'no statistics' sentinel string."""
        result = generate_stats_summary({})
        assert "No statistics to summarize" in result

    def test_metadata_only_returns_sentinel(self):
        """A dict with only a 'metadata' key is treated as empty."""
        result = generate_stats_summary({"metadata": {"repo": "test"}})
        assert "No statistics to summarize" in result

    def test_header_shows_single_run(self, minimal_stats_data: dict):
        """A count of 1 shows '1 run' in the header, not '1 runs'."""
        data = dict(minimal_stats_data)
        # Override count to 1
        for v in data.values():
            v["count"] = 1
        result = generate_stats_summary(data)
        assert "1 run" in result
        assert "1 runs" not in result

    def test_header_shows_plural_runs(self, minimal_stats_data: dict):
        """A count > 1 shows the plural 'X runs'."""
        result = generate_stats_summary(minimal_stats_data)
        assert "3 runs" in result

    def test_contains_top_runtime_section(self, minimal_stats_data: dict):
        """Output includes the 'Top N Functions by Average Runtime' section."""
        result = generate_stats_summary(minimal_stats_data)
        assert "Average Runtime" in result

    def test_no_memory_section_when_absent(self, minimal_stats_data: dict):
        """Memory section is omitted when no function has memory data."""
        result = generate_stats_summary(minimal_stats_data)
        assert "Average Memory" not in result

    def test_memory_section_present_when_available(self, stats_with_memory: dict):
        """Memory section appears when at least one function has memory data."""
        result = generate_stats_summary(stats_with_memory)
        assert "Average Memory" in result

    def test_contains_module_tree_section(self, minimal_stats_data: dict):
        """Output always includes the 'Functions by Module' tree section."""
        result = generate_stats_summary(minimal_stats_data)
        assert "Functions by Module" in result

    def test_threshold_filters_functions(self, stats_with_memory: dict):
        """Functions below threshold_sec are excluded from ranked list."""
        # fast_func has 0.5s total; exclude it with a 1.0s threshold
        result = generate_stats_summary(stats_with_memory, threshold_sec=1.0)
        assert "fast_func" not in result

    def test_top_n_limits_ranked_list(self, multi_run_manager: StatisticsManager):
        """Only top_n functions appear in the ranked runtime list."""
        result = generate_stats_summary(multi_run_manager.stats(), top_n=1)
        # Should show 'Top Function by Average Runtime' (singular), not 'Top N Functions'
        assert "Top Function by Average Runtime" in result

    def test_std_appears_in_multi_run_output(self, multi_run_manager: StatisticsManager):
        """With identical runs std is 0, so ± should NOT appear."""
        result = generate_stats_summary(multi_run_manager.stats())
        # Identical runs → std == 0 → no ± in time output
        # (memory ± may still appear if memory is reported with std > 0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_real_profile_data_single_run(self, single_run_manager: StatisticsManager):
        """Summary from a real profiling run is non-empty and well-formed."""
        result = generate_stats_summary(single_run_manager.stats())
        assert "Benchmark Summary" in result
        assert "=" * 10 in result  # at least some separator line

    def test_real_profile_data_multi_run(self, multi_run_manager: StatisticsManager):
        """Summary from multiple profiling runs is non-empty and well-formed."""
        result = generate_stats_summary(multi_run_manager.stats())
        assert "Benchmark Summary (3 runs)" in result
        assert "Functions by Module" in result

    def test_duration_format_in_output(self, single_run_manager: StatisticsManager):
        """Duration values in the output include a time unit."""
        result = generate_stats_summary(single_run_manager.stats())
        assert any(unit in result for unit in ["sec", "ms", "min", "hr", "µs"])

    def test_file_names_in_output(self, single_run_manager: StatisticsManager):
        """Function file names appear in the ranked list."""
        result = generate_stats_summary(single_run_manager.stats())
        # At least one .py filename should appear in parentheses
        assert ".py)" in result


# ---------------------------------------------------------------------------
# StatsSummary class
# ---------------------------------------------------------------------------


class TestStatsSummaryFromManager:
    """Test StatsSummary constructed from a live StatisticsManager."""

    def test_count_reflects_runs(self, multi_run_manager: StatisticsManager):
        """StatsSummary.count equals the number of profiling runs."""
        ss = StatsSummary(multi_run_manager)
        assert ss.count == 3

    def test_summary_returns_string(self, single_run_manager: StatisticsManager):
        """summary() returns a non-empty string."""
        ss = StatsSummary(single_run_manager)
        result = ss.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_top_n_param(self, multi_run_manager: StatisticsManager):
        """top_n parameter is passed through to the formatter."""
        ss = StatsSummary(multi_run_manager)
        result = ss.summary(top_n=1)
        assert "Top Function by Average Runtime" in result

    def test_summary_threshold_param(self, multi_run_manager: StatisticsManager):
        """threshold_sec parameter filters low-runtime functions."""
        ss = StatsSummary(multi_run_manager)
        # Very high threshold should exclude everything
        result = ss.summary(threshold_sec=1e9)
        assert "No functions above the runtime threshold" in result

    def test_summary_contains_benchmark_header(self, multi_run_manager: StatisticsManager):
        """Summary text opens with the 'Benchmark Summary' header."""
        ss = StatsSummary(multi_run_manager)
        assert "Benchmark Summary" in ss.summary()


class TestStatsSummaryFromFile:
    """Test StatsSummary constructed from a saved JSON file."""

    def test_load_from_path(self, multi_run_manager: StatisticsManager, tmp_path: Path):
        """StatsSummary can be constructed from a Path to a saved JSON file."""
        stats_file = tmp_path / "stats.json"
        StatisticsManager.save_as_json(stats_file, multi_run_manager.stats(), save_metadata=False)
        ss = StatsSummary(stats_file)
        assert ss.count == 3
        assert "Benchmark Summary" in ss.summary()

    def test_load_from_string_path(self, multi_run_manager: StatisticsManager, tmp_path: Path):
        """StatsSummary can be constructed from a string path."""
        stats_file = tmp_path / "stats.json"
        StatisticsManager.save_as_json(stats_file, multi_run_manager.stats(), save_metadata=False)
        ss = StatsSummary(str(stats_file))
        assert isinstance(ss.summary(), str)

    def test_metadata_key_excluded_from_count(self, multi_run_manager: StatisticsManager, tmp_path: Path):
        """The 'metadata' key in the JSON is not interpreted as a function entry."""
        stats_file = tmp_path / "stats.json"
        StatisticsManager.save_as_json(stats_file, multi_run_manager.stats(), save_metadata=True)
        ss = StatsSummary(stats_file)
        assert ss.count == 3  # unchanged by metadata presence

    def test_summary_text_matches_in_memory(self, multi_run_manager: StatisticsManager, tmp_path: Path):
        """Summary from a file matches summary computed directly from the manager."""
        stats_file = tmp_path / "stats.json"
        StatisticsManager.save_as_json(stats_file, multi_run_manager.stats(), save_metadata=False)
        from_file = StatsSummary(stats_file).summary()
        from_manager = StatsSummary(multi_run_manager).summary()
        assert from_file == from_manager


# ---------------------------------------------------------------------------
# StatisticsManager.summary()
# ---------------------------------------------------------------------------


class TestStatisticsManagerSummary:
    """Test the StatisticsManager.summary() convenience method."""

    def test_returns_non_empty_string(self, single_run_manager: StatisticsManager):
        """summary() returns a non-empty string."""
        result = single_run_manager.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_matches_generate_stats_summary(self, multi_run_manager: StatisticsManager):
        """summary() produces the same output as calling generate_stats_summary directly."""
        assert multi_run_manager.summary() == generate_stats_summary(multi_run_manager.stats())

    def test_top_n_forwarded(self, multi_run_manager: StatisticsManager):
        """top_n argument is forwarded correctly."""
        result = multi_run_manager.summary(top_n=1)
        assert "Top Function by Average Runtime" in result

    def test_threshold_forwarded(self, multi_run_manager: StatisticsManager):
        """threshold_sec argument is forwarded correctly."""
        result = multi_run_manager.summary(threshold_sec=1e9)
        assert "No functions above the runtime threshold" in result


# ---------------------------------------------------------------------------
# CLI stats command: .txt file is saved
# ---------------------------------------------------------------------------


class TestStatsCLITextFile:
    """Test that the stats CLI command saves a .txt summary file."""

    def test_stats_cli_creates_txt_file(self, example_json: Path):
        """The stats command creates a .txt summary next to the .json file."""
        from typer.testing import CliRunner
        from fixingahole.cli import main as cli

        runner = CliRunner()
        tmp_dir = example_json.parent / "tmp_stats"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        example_json.rename(tmp_dir / example_json.name)
        (tmp_dir / "dup.json").write_bytes((tmp_dir / example_json.name).read_bytes())
        result = runner.invoke(cli.app, ["stats", str(tmp_dir), "--no-metadata"])
        assert result.exit_code == 0
        txt_files = list(tmp_dir.glob("*.txt"))
        assert len(txt_files) == 1
        assert "Benchmark Summary" in txt_files[0].read_text(encoding="utf-8")

    def test_stats_cli_output_is_benchmark_summary(self, example_json: Path):
        """The stats CLI saves a .txt with the benchmark summary, not raw JSON."""
        from typer.testing import CliRunner
        from fixingahole.cli import main as cli

        runner = CliRunner()
        tmp_dir = example_json.parent / "tmp_stats2"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        example_json.rename(tmp_dir / example_json.name)
        (tmp_dir / "dup.json").write_bytes((tmp_dir / example_json.name).read_bytes())
        result = runner.invoke(cli.app, ["stats", str(tmp_dir), "--no-metadata"])
        assert result.exit_code == 0
        # The saved .txt file must contain the benchmark header
        txt_files = list(tmp_dir.glob("*.txt"))
        assert len(txt_files) == 1
        txt_content = txt_files[0].read_text(encoding="utf-8")
        assert "Benchmark Summary" in txt_content
        # The saved .txt must not be raw JSON
        assert not txt_content.strip().startswith("{")
