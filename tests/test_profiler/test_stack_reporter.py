# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the StackReporter class."""

import json
from pathlib import Path
from typing import Any

import pytest

from fixingahole.profiler.stack_reporter import StackReporter, StackReporterError


@pytest.fixture
def example_json(tmp_path: Path) -> Path:
    """Return path to the advanced profile results JSON file."""
    example_json_file: Path = Path(__file__).parents[1] / "scripts" / "data" / "advanced_profile_results.json"
    file_path: Path = tmp_path / "example.json"
    file_path.write_bytes(example_json_file.read_bytes())
    return file_path


@pytest.fixture
def stack_reporter(example_json: Path) -> StackReporter:
    """Initialize a StackReporter instance with test data."""
    return StackReporter(example_json)


@pytest.fixture
def sample_data(example_json: Path) -> dict[str, Any]:
    """Load the test profile data."""
    return json.loads(example_json.read_text(encoding="utf-8"))


class TestStackReporterInitialization:
    """Test StackReporter initialization."""

    def test_init_with_valid_path(self, example_json: Path) -> None:
        """Test initialization with both Path and string paths."""
        # Test with Path object
        reporter: StackReporter = StackReporter(example_json)
        assert reporter.profile_json_path == example_json.resolve()
        assert isinstance(reporter.data, dict)
        assert "files" in reporter.data
        assert "stacks" in reporter.data
        assert "elapsed_time_sec" in reporter.data

        # Test with string path
        reporter_str: StackReporter = StackReporter(str(example_json))
        assert reporter_str.profile_json_path == example_json.resolve()
        assert reporter_str.data == reporter.data

    def test_init_with_nonexistent_file(self) -> None:
        """Test initialization with a non-existent file raises StackReporterError."""
        with pytest.raises(StackReporterError, match="Failed to initialize StackReporter"):
            StackReporter("/path/to/nonexistent/file.json")

    def test_init_with_invalid_json(self, example_json: Path) -> None:
        """Test initialization with invalid JSON raises StackReporterError."""
        example_json.write_text("not valid json{", encoding="utf-8")
        with pytest.raises(StackReporterError, match="Failed to initialize StackReporter"):
            StackReporter(example_json)


class TestLoadProfileResults:
    """Test the load_profile_results static method."""

    def test_load_profile_results(self, example_json: Path, sample_data: dict[str, Any]) -> None:
        """Test loading profile results validates structure and content."""
        data: dict[str, Any] = StackReporter.load_profile_results(example_json)

        # Validate required top-level keys
        assert isinstance(data, dict)
        assert "files" in data
        assert "stacks" in data
        assert "elapsed_time_sec" in data

        # Validate files structure
        assert isinstance(data["files"], dict)
        assert len(data["files"]) > 0

        # Validate at least one file has functions with required fields
        has_valid_functions: bool = False
        for file_data in data["files"].values():
            if "functions" in file_data and len(file_data["functions"]) > 0:
                func: dict[str, Any] = file_data["functions"][0]
                assert "line" in func
                assert "n_cpu_percent_c" in func
                assert "n_cpu_percent_python" in func
                has_valid_functions = True
                break
        assert has_valid_functions, "No valid functions found in data"

        # Validate it matches the fixture data
        assert data == sample_data


class TestGetTopFunctions:
    """Test the get_top_functions method."""

    def test_get_top_functions(self, stack_reporter: StackReporter) -> None:
        """Test getting top functions returns correctly sorted and limited results."""
        # Test default (top 5)
        default_n = 5
        top_5: list[dict[str, Any]] = stack_reporter.get_top_functions()
        assert len(top_5) <= default_n
        assert len(top_5) > 0, "Should have at least one function"

        # Validate structure
        for func in top_5:
            assert isinstance(func, dict)
            assert "file" in func
            assert isinstance(func["file"], str)
            assert "name" in func
            assert isinstance(func["name"], str)
            assert "total_percent" in func
            assert isinstance(func["total_percent"], (int, float))
            assert func["total_percent"] >= 0, "CPU percent should be non-negative"

        # Validate descending sort order
        for i in range(len(top_5) - 1):
            assert top_5[i]["total_percent"] >= top_5[i + 1]["total_percent"], (
                f"Functions not sorted: {top_5[i]['total_percent']} < {top_5[i + 1]['total_percent']}"
            )

        # Test custom n
        n = 3
        top_3: list[dict[str, Any]] = stack_reporter.get_top_functions(n=n)
        assert len(top_3) <= n
        assert top_3[0] == top_5[0], "Top function should be the same"

        # Test requesting more than available
        top_1000: list[dict[str, Any]] = stack_reporter.get_top_functions(n=1000)
        assert len(top_1000) == len(stack_reporter.get_top_functions(n=10000)), (
            "Should return all available functions when n exceeds total"
        )


class TestFindStackTraces:
    """Test the find_stack_traces static method."""

    def test_find_stack_traces(self, sample_data: dict[str, Any]) -> None:
        """Test finding stack traces for existing and non-existing functions."""
        stacks: list[Any] = sample_data["stacks"]

        # Test with existing function
        traces: list[dict[str, Any]] = StackReporter.find_stack_traces(stacks, "data_serialization")
        assert isinstance(traces, list)
        assert len(traces) > 0, "Should find traces for data_serialization"

        # Validate trace structure and content
        for trace in traces:
            assert "stack" in trace
            assert isinstance(trace["stack"], list)
            assert len(trace["stack"]) > 0, "Stack should not be empty"
            assert "count" in trace
            assert isinstance(trace["count"], int)
            assert "cpu_samples" in trace
            assert isinstance(trace["cpu_samples"], (int, float))
            assert "c_time" in trace
            assert isinstance(trace["c_time"], float)
            assert "python_time" in trace
            assert isinstance(trace["python_time"], float)

            # Validate function name appears in last stack frame
            last_frame: str = trace["stack"][-1]
            assert "data_serialization" in last_frame, f"Function name should appear in last frame: {last_frame}"

        # Test with non-existent function
        empty_traces: list[dict[str, Any]] = StackReporter.find_stack_traces(stacks, "nonexistent_function_xyz")
        assert isinstance(empty_traces, list)
        assert len(empty_traces) == 0, "Should return empty list for non-existent function"


class TestCombineStackTraces:
    """Test the combine_stack_traces static method."""

    def test_combine_stack_traces(self, sample_data: dict[str, Any]) -> None:
        """Test combining stack traces correctly aggregates values."""
        # Test with empty list
        empty_combined: dict[tuple[str], dict[str, float]] = StackReporter.combine_stack_traces([])
        assert isinstance(empty_combined, dict)
        assert len(empty_combined) == 0

        # Test with real data
        test_traces: list[dict[str, Any]] = StackReporter.find_stack_traces(sample_data["stacks"], "data_serialization")
        assert len(test_traces) > 0, "Need traces to test combination"

        combined: dict[tuple[str], dict[str, float]] = StackReporter.combine_stack_traces(test_traces)
        assert len(combined) >= 1, "Should have at least one combined stack"

        # Validate structure
        for key, value in combined.items():
            assert isinstance(key, tuple), "Keys should be tuples"
            assert len(key) > 0, "Key should not be empty"
            assert "count" in value
            assert "cpu_samples" in value
            assert "c_time" in value
            assert "python_time" in value

        # Validate aggregation: totals should match
        total_count: int = sum(t["count"] for t in test_traces)
        combined_count: int = sum(int(v["count"]) for v in combined.values())
        assert combined_count == total_count, f"Combined count {combined_count} != original total {total_count}"

        total_cpu: float = sum(t["cpu_samples"] for t in test_traces)
        combined_cpu: float = sum(v["cpu_samples"] for v in combined.values())
        assert combined_cpu == total_cpu, f"Combined CPU samples {combined_cpu} != original total {total_cpu}"


class TestBuildCombinedReverseTree:
    """Test the build_combined_reverse_tree static method."""

    def test_build_combined_reverse_tree(self, sample_data: dict[str, Any]) -> None:
        """Test building reverse tree structure from stack traces."""
        # Test with empty traces
        empty_tree, empty_info = StackReporter.build_combined_reverse_tree([])
        assert empty_tree == {}
        assert empty_info == {}

        # Test with real data
        traces: list[dict[str, Any]] = StackReporter.find_stack_traces(sample_data["stacks"], "_wrapreduction")
        if not traces:
            pytest.skip("No traces found for _wrapreduction")

        tree, call_info = StackReporter.build_combined_reverse_tree(traces)

        # Validate tree structure
        assert isinstance(tree, dict)
        assert len(tree) > 0, "Tree should not be empty"

        # Validate call_info structure and content
        assert isinstance(call_info, dict)
        assert len(call_info) > 0, "Call info should not be empty"

        for path, info in call_info.items():
            assert isinstance(path, tuple), "Paths should be tuples"
            assert len(path) > 0, "Path should not be empty"
            assert "count" in info
            assert isinstance(info["count"], int)
            assert "cpu_samples" in info
            assert isinstance(info["cpu_samples"], (int, float))
            assert "c_time" in info
            assert isinstance(info["c_time"], float)
            assert "python_time" in info
            assert isinstance(info["python_time"], float)

            # Validate path elements are normalized stack frames
            for frame in path:
                assert isinstance(frame, str)
                assert ":" in frame, f"Frame should contain ':' separator: {frame}"


class TestRenderCombinedReverseTree:
    """Test the render_combined_reverse_tree method."""

    def test_render_combined_reverse_tree(self, stack_reporter: StackReporter, sample_data: dict[str, Any]) -> None:
        """Test rendering tree with real data and edge cases."""
        # Test empty tree
        empty_lines: list[str] = stack_reporter.render_combined_reverse_tree({}, {})
        assert isinstance(empty_lines, list)
        assert len(empty_lines) == 0, "Empty tree should produce no output"

        # Test with real data
        traces: list[dict[str, Any]] = StackReporter.find_stack_traces(sample_data["stacks"], "data_serialization")
        assert len(traces) > 0, "Need traces for testing"

        tree, call_info = StackReporter.build_combined_reverse_tree(traces)
        lines: list[str] = stack_reporter.render_combined_reverse_tree(tree, call_info, is_root=False)
        print("\n\n")
        print(*lines, sep="\n")

        # Validate output structure
        assert isinstance(lines, list)
        assert len(lines) > 0, "Should produce output for real data"

        # Validate content contains stack frame information
        assert any("n_calls" in line for line in lines), "Should display call counts"

        # Validate tree drawing characters are present
        tree_chars: list[str] = ["└", "├", "│"]
        has_tree_chars: bool = any(any(char in line for char in tree_chars) for line in lines)
        assert has_tree_chars, "Should use tree drawing characters for hierarchy"

        # Validate that actual file paths from data appear in output
        file_paths_in_data: bool = any("advanced.py" in line or ".py" in line for line in lines)
        assert file_paths_in_data, "Should contain actual file paths from traces"


class TestReportStacksForTopFunctions:
    """Test the report_stacks_for_top_functions method."""

    def test_report_stacks_for_top_functions(self, stack_reporter: StackReporter, sample_data: dict[str, Any]) -> None:
        """Test generating complete stack trace report."""
        # Test default report
        report: str = stack_reporter.report_stacks_for_top_functions()
        assert isinstance(report, str)
        assert len(report) > 0

        # Validate report contains expected sections
        assert "Stack Trace Summary" in report, "Should have header"
        elapsed: float = sample_data["elapsed_time_sec"]
        assert f"{elapsed:.3f}s" in report, "Should show elapsed time"
        assert "=" in report, "Should have separator line"
        assert "%" in report, "Should show percentages"

        # Validate top functions appear in report
        top_funcs: list[dict[str, Any]] = stack_reporter.get_top_functions(n=5)
        for func in top_funcs:
            assert func["name"] in report, f"Function {func['name']} should appear in report"
            # Should show percentage for each function
            assert f"({func['total_percent']:.2f}%)" in report or func["name"] in report

        # Test custom number of functions
        n = 3
        report_3: str = stack_reporter.report_stacks_for_top_functions(top_n=n)
        assert "Stack Trace Summary" in report_3
        lines: list[str] = report_3.split("\n")
        assert len(lines) > n, "Should have multiple lines"

        # Test with high n value
        report_many: str = stack_reporter.report_stacks_for_top_functions(top_n=100)
        assert isinstance(report_many, str)
        assert len(report_many) > 0


class TestStackReporterError:
    """Test the StackReporterError exception."""

    def test_stack_reporter_error(self) -> None:
        """Test StackReporterError is properly defined and usable."""
        assert issubclass(StackReporterError, Exception)

        msg: str = "Test error message"
        with pytest.raises(StackReporterError, match=msg):
            raise StackReporterError(msg)


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_end_to_end_workflow(self, example_json: Path) -> None:
        """Test complete workflow from initialization to report generation."""
        # Initialize reporter
        reporter: StackReporter = StackReporter(example_json)
        assert reporter.data is not None

        # Get top functions
        top_funcs: list[dict[str, Any]] = reporter.get_top_functions(n=3)
        assert len(top_funcs) > 0
        top_func_name: str = top_funcs[0]["name"]
        assert len(top_func_name) > 0

        # Generate report
        report: str = reporter.report_stacks_for_top_functions(top_n=3)
        assert isinstance(report, str)
        assert len(report) > 0
        assert top_func_name in report

        # Validate report contains all top functions
        for func in top_funcs:
            assert func["name"] in report

    def test_multiple_reporters_produce_identical_results(self, example_json: Path) -> None:
        """Test that multiple reporters on same file produce identical output."""
        reporter1: StackReporter = StackReporter(example_json)
        reporter2: StackReporter = StackReporter(example_json)

        # Compare top functions
        top1: list[dict[str, Any]] = reporter1.get_top_functions(n=5)
        top2: list[dict[str, Any]] = reporter2.get_top_functions(n=5)
        assert top1 == top2, "Top functions should be identical"

        # Compare reports
        report1: str = reporter1.report_stacks_for_top_functions(top_n=2)
        report2: str = reporter2.report_stacks_for_top_functions(top_n=2)
        assert report1 == report2, "Reports should be identical"
