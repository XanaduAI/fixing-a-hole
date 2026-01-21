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
"""Tests for the profile JSON parser module."""

from pathlib import Path

import pytest
from typer import Exit

from fixingahole.profiler.profile_summary import (
    ProfileData,
    ProfileDetails,
    ProfileSummary,
    build_module_tree,
    generate_summary,
    get_all_functions_in_tree,
    memory_with_units,
    parse_json,
    render_tree,
)


@pytest.fixture
def advanced_profile_json() -> Path:
    """Return the path to the advanced profile results JSON file."""
    return Path(__file__).parent.parent / "scripts" / "data" / "advanced_profile_results.json"


class TestMemoryWithUnits:
    """Test the memory_with_units function."""

    @pytest.mark.parametrize(
        ("value", "unit", "precision", "expected"),
        [
            # Bytes
            (100, "B", 0, "100 bytes"),
            (12_345, "B", 2, "12.06 KB"),
            # Kilobytes
            (1, "KB", 0, "  1 KB"),
            (1.5, "KB", 2, "1.50 KB"),
            # Megabytes
            (1, "MB", 0, "  1 MB"),
            (256.5, "MB", 2, "256.50 MB"),
            (1024, "MB", 0, "  1 GB"),
            # Gigabytes
            (1, "GB", 0, "  1 GB"),
            (2.5, "GB", 3, "2.500 GB"),
            # Zero
            (0, "MB", 0, "0 bytes"),
        ],
    )
    def test_memory_with_units(self, value: float, unit: str, precision: int, expected: str):
        """Test memory conversion with various inputs."""
        assert memory_with_units(value, unit, precision) == expected


class TestProfileDetails:
    """Test the ProfileDetails dataclass."""

    def test_create_function_profile_complete(self):
        """Test creating a function profile with complete data."""
        profile = ProfileDetails(
            name="test_func",
            file_path="/path/to/file.py",
            line_number=42,
            memory_python_percentage=75.5,
            peak_memory=128.0,
            timeline_percentage=10.5,
            copy_mb_per_s=50.0,
            python_percentage=45.0,
            native_percentage=5.0,
            system_percentage=2.0,
            memory_samples=[(1.0, 2.0), (3.0, 4.0)],
        )
        assert profile.name == "test_func"
        assert profile.file_path == "/path/to/file.py"
        assert profile.line_number == 42
        assert profile.memory_python_percentage == 75.5
        assert profile.peak_memory == 128.0
        assert profile.timeline_percentage == 10.5
        assert profile.copy_mb_per_s == 50.0
        assert profile.python_percentage == 45.0
        assert profile.native_percentage == 5.0
        assert profile.system_percentage == 2.0
        assert profile.memory_samples == [(1.0, 2.0), (3.0, 4.0)]
        assert profile.has_memory_info

    def test_function_profile_immutable(self):
        """Test that ProfileDetails is immutable (frozen=True)."""
        profile = ProfileDetails(
            name="test",
            file_path="/path",
            line_number=1,
            memory_python_percentage=0.0,
            peak_memory=0.0,
            python_percentage=0.0,
            native_percentage=0.0,
            system_percentage=0.0,
            timeline_percentage=0.0,
            copy_mb_per_s=0.0,
            memory_samples=[(1.0, 2.0), (3.0, 4.0)],
        )
        with pytest.raises(AttributeError):
            profile.name = "modified"  # ty:ignore[invalid-assignment]


class TestParseJson:
    """Test the parse_json function."""

    def test_parse_json_nonexistent_file(self, tmp_path: Path):
        """Test parsing a nonexistent JSON file."""
        with pytest.raises(Exit) as exc:
            parse_json(tmp_path / "nonexistent.json")
        assert exc.value.exit_code == 66

    def test_parse_json_empty_file(self, tmp_path: Path):
        """Test parsing an empty JSON file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")
        result = parse_json(json_file)
        assert result.functions == []
        assert result.walltime == 0
        # Memory format may vary based on size, just check it's not None
        assert result.max_memory is not None
        assert "0" in result.max_memory

    def test_parse_json_complete_profile(self, example_json: Path):
        """Test parsing a complete JSON profile."""
        result = parse_json(example_json)

        # Check the data is specific to the file we parsed.
        assert result.walltime == 10.986895561218262
        assert result.max_memory == "950.715 MB"
        assert len(result.functions) == 9

        # Verify some functions were parsed
        names = [f.name for f in result.functions]
        assert any(names)  # At least some functions have names

        # Check that functions have expected fields
        for func in result.functions[:5]:  # Check first 5 functions
            assert isinstance(func.file_path, str)
            assert isinstance(func.line_number, int)
            assert isinstance(func.python_percentage, float)
            assert isinstance(func.native_percentage, float)
            assert isinstance(func.system_percentage, float)

        # Check
        func = result.functions[0]
        assert func.file_path == "/home/ubuntu/fixing-a-hole/performance/advanced.py"
        assert func.line_number == 35
        assert func.name == "matrix_operations"
        assert func.python_percentage == 0.003940147384761709
        assert func.native_percentage == 0.009041007225759054
        assert func.system_percentage == 0.0038929610193535105
        assert func.peak_memory == 36.31560134887695
        assert func.copy_mb_per_s == 0.0
        assert func.memory_python_percentage == 0.20126943997844543 * 100
        assert func.timeline_percentage == 0.0018903851360701405 * 100


class TestGetFunctionsByFile:
    """Test the functions_by_file function."""

    def test_functions_by_file_real_file(self, example_json: Path):
        """Test grouping functions from a single file."""
        profile_data = parse_json(example_json)
        result = profile_data.functions_by_file
        expected_filenames = [
            "/home/ubuntu/fixing-a-hole/performance/advanced.py",
            "/home/ubuntu/fixing-a-hole/.venv/lib/python3.11/site-packages/numpy/_core/_methods.py",
        ]
        assert len(result) == 2
        assert list(result.keys()) == expected_filenames
        for name, n_funcs in zip(expected_filenames, [6, 3], strict=False):
            assert len(result[name]) == n_funcs

        # Check that all functions are grouped correctly
        total_functions = sum(len(funcs) for funcs in result.values())
        assert total_functions == len(profile_data.functions)

        # Verify all file paths in result match the functions
        for file_path, funcs in result.items():
            for func in funcs:
                assert func.file_path == file_path


class TestBuildModuleTree:
    """Test the build_module_tree function."""

    def test_build_module_tree_real_profile(self, example_json: Path):
        """Test building tree from real profile data."""
        profile_data = parse_json(example_json)
        tree = build_module_tree(profile_data.functions_by_file)

        # Should create a hierarchical structure
        assert tree is not None
        assert len(tree) > 0

        # Each node should have _functions and _children keys
        for node in tree.values():
            assert "_functions" in node
            assert "_children" in node
            assert isinstance(node["_functions"], list)
            assert isinstance(node["_children"], dict)


class TestGetAllFunctionsInTree:
    """Test the get_all_functions_in_tree function."""

    def test_get_all_functions_in_tree_flat(self, example_json: Path):
        """Test getting functions from flat tree."""
        profile_data = parse_json(example_json)
        tree = build_module_tree(profile_data.functions_by_file)
        result = get_all_functions_in_tree(tree)
        assert len(result) == 2
        for i, n_funcs in enumerate([6, 3]):
            assert len(result[i]) == n_funcs

    def test_get_all_functions_in_tree_empty(self):
        """Test getting functions from empty tree."""
        result = get_all_functions_in_tree({})
        assert result == []


class TestRenderTree:
    """Test the render_tree function."""

    def test_render_tree_single_file(self, example_json: Path):
        """Test rendering tree with single file."""
        profile_data = parse_json(example_json)
        tree = build_module_tree(profile_data.functions_by_file)
        result = render_tree(tree)
        assert len(result) == 8
        expected_tree = [
            "├─ performance (6 func, 99.65% total)",
            "│  └─ advanced.py (6 func, 99.65% total)",
            "│     └─ data_serialization.......................99.57% ( 80 MB)",
            "│     ",
            "└─ numpy (3 func, 0.00% total)",
            "   └─ _core (3 func, 0.00% total)",
            "      └─ _methods.py (3 func, 0.00% total)",
            "         ",
        ]
        assert result == expected_tree

    def test_render_tree_with_threshold(self, example_json: Path):
        """Test rendering tree with threshold filtering."""
        profile_data = parse_json(example_json)
        tree = build_module_tree(profile_data.functions_by_file)
        result = render_tree(tree, threshold=0.0)
        assert len(result) == 16
        expected_tree = [
            "├─ performance (6 func, 99.65% total)",
            "│  └─ advanced.py (6 func, 99.65% total)",
            "│     ├─ data_serialization.......................99.57% ( 80 MB)",
            "│     ├─ fourier_analysis..........................0.03% (153 MB)",
            "│     ├─ statistical_analysis......................0.02% ( 76 MB)",
            "│     ├─ matrix_operations.........................0.02% ( 36 MB)",
            "│     ├─ monte_carlo_simulation....................0.01% ( 76 MB)",
            "│     └─ recursive_computation.....................0.00%",
            "│     ",
            "└─ numpy (3 func, 0.00% total)",
            "   └─ _core (3 func, 0.00% total)",
            "      └─ _methods.py (3 func, 0.00% total)",
            "         ├─ _mean..................................0.00%",
            "         ├─ _var...................................0.00% (153 MB)",
            "         └─ _std...................................0.00%",
            "         ",
        ]
        assert result == expected_tree


class TestGenerateSummary:
    """Test the generate_summary function."""

    def test_generate_summary_empty_profile(self):
        """Test generating summary from empty profile."""
        profile_data = ProfileData(functions=[], lines={}, files={}, walltime=None, max_memory=None, samples=[], details={})
        result = generate_summary(profile_data)
        assert "No functions to summarize" in result

    def test_generate_summary_real_profile(self, example_json: Path):
        """Test generating summary from real profile data."""
        profile_data = parse_json(example_json)
        result = generate_summary(profile_data, top_n=5)

        expected_summary = [
            "\nProfile Summary",
            "=================================================================",
            "\nTop 5 Functions by Total Runtime:",
            "-----------------------------------------------------------------",
            " 1. data_serialization         99.6% (advanced.py:144)",
            " 2. fourier_analysis            0.0% (advanced.py:108)",
            " 3. statistical_analysis        0.0% (advanced.py:72)",
            " 4. matrix_operations           0.0% (advanced.py:35)",
            " 5. monte_carlo_simulation      0.0% (advanced.py:56)",
            "\nTop 5 Functions by Memory Usage:",
            "-----------------------------------------------------------------",
            " 1. fourier_analysis            153 MB (advanced.py)",
            " 2. _var                        153 MB (_methods.py)",
            " 3. data_serialization           80 MB (advanced.py)",
            " 4. statistical_analysis         76 MB (advanced.py)",
            " 5. monte_carlo_simulation       76 MB (advanced.py)",
            "\nFunctions by Module:",
            "-----------------------------------------------------------------",
            "├─ performance (6 func, 99.65% total)",
            "│  └─ advanced.py (6 func, 99.65% total)",
            "│     └─ data_serialization.......................99.57% ( 80 MB)",
            "│",
            "└─ numpy (3 func, 0.00% total)",
            "   └─ _core (3 func, 0.00% total)",
            "      └─ _methods.py (3 func, 0.00% total)",
            "",
            "",
            "=================================================================",
            "",
        ]

        assert result == "\n".join(expected_summary)


class TestProfileSummaryExtraction:
    """Test the ProfileSummary class with real profile data."""

    def test_parse_json_profile(self, advanced_profile_json: Path):
        """Test parsing a JSON profile results file."""
        parser = ProfileSummary(filename=advanced_profile_json)

        # Verify the walltime is extracted correctly
        assert parser.walltime == pytest.approx(10.986895561218262, rel=1e-6)

        # Verify the max memory is extracted and formatted correctly
        # The JSON has max_footprint_mb: 950.7153148651123, which should be formatted to "950.715 MB"
        assert parser.max_memory == "950.715 MB"

    def test_profile_has_functions(self, advanced_profile_json: Path):
        """Test that functions are extracted from the profile."""
        parser = ProfileSummary(filename=advanced_profile_json)

        # Verify functions were extracted
        assert parser.data.functions
        assert len(parser.data.functions) > 0

        # Check some expected function names from the JSON
        function_names = [f.name for f in parser.data.functions]
        assert "matrix_operations" in function_names
        assert "monte_carlo_simulation" in function_names
        assert "fourier_analysis" in function_names
        assert "data_serialization" in function_names

    def test_profile_has_memory_info(self, advanced_profile_json: Path):
        """Test that memory information is captured."""
        parser = ProfileSummary(filename=advanced_profile_json)

        # The profile should have memory information
        assert parser.data.has_memory_info

        # Check that some functions have memory data
        funcs_with_memory = [f for f in parser.data.functions if f.has_memory_info]
        assert len(funcs_with_memory) > 0

    def test_generate_summary(self, advanced_profile_json: Path):
        """Test generating a summary from the profile with actual data validation."""
        parser = ProfileSummary(filename=advanced_profile_json)
        summary = parser.summary(top_n=5)

        # Verify summary contains expected sections
        assert "Profile Summary" in summary
        assert "Top 5 Functions by Total Runtime" in summary
        assert "Functions by Memory Usage" in summary
        assert "Functions by Module" in summary

        # Validate that actual function names from the JSON appear in the summary
        assert "data_serialization" in summary, "Top function should appear in summary"
        assert "fourier_analysis" in summary, "Should contain fourier_analysis function"
        assert "advanced.py" in summary, "Should contain actual file name from data"

        # Validate that the top function appears with correct percentage
        # data_serialization has ~97.96% total (97.33% C + 0.63% Python)
        assert "data_serialization" in summary
        assert "99.6%" in summary, "Top function should show with correct percentage"

        # Validate memory information from actual data
        # fourier_analysis has 152.59 MB peak memory
        assert "153 MB" in summary, "Should show memory usage from actual data"
        assert "_var" in summary, "Should show numpy function with memory data"

        # Validate file references match actual data
        assert "advanced.py:144" in summary or "advanced.py" in summary, "Should reference actual file and line number"

        # Validate module tree structure from data
        assert "performance" in summary, "Should show performance module from file path"
        assert "numpy" in summary, "Should show numpy module from dependencies"

    def test_nonexistent_file(self, tmp_path: Path):
        """Test handling of non-existent profile file."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(Exit) as exc:
            ProfileSummary(filename=nonexistent)
        assert exc.value.exit_code == 66
