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
"""Tests for the profile JSON parser module."""

from pathlib import Path

import pytest

from fixingahole.profiler.profile_json_parser import (
    FunctionProfile,
    ProfileData,
    build_module_tree,
    generate_summary,
    get_all_functions_in_tree,
    get_functions_by_file,
    get_top_functions,
    memory_with_units,
    parse_json,
    render_tree,
)


@pytest.fixture
def example_json(tmp_path: Path) -> Path:
    """Return path to the advanced profile results JSON file."""
    example_json = Path(__file__).parents[1] / "scripts" / "advanced_profile_results.json"
    file_path = tmp_path / "example.json"
    file_path.write_bytes(example_json.read_bytes())
    return file_path


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


class TestCreateFunctionProfile:
    """Test the create_function_profile function."""

    def test_create_function_profile_complete(self):
        """Test creating a function profile with complete data."""
        profile = FunctionProfile(
            function_name="test_func",
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
        assert profile.function_name == "test_func"
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


class TestFunctionProfile:
    """Test the FunctionProfile dataclass."""

    def test_function_profile_creation(self):
        """Test direct instantiation of FunctionProfile."""
        profile = FunctionProfile(
            function_name="test_function",
            file_path="/path/to/file.py",
            line_number=42,
            memory_python_percentage=75.0,
            peak_memory=256.0,
            python_percentage=45.0,
            native_percentage=5.0,
            system_percentage=2.0,
            timeline_percentage=10.0,
            copy_mb_per_s=50.0,
            memory_samples=[(1.0, 2.0), (3.0, 4.0)],
        )
        assert profile.function_name == "test_function"
        assert profile.file_path == "/path/to/file.py"
        assert profile.line_number == 42
        assert profile.memory_python_percentage == 75.0
        assert profile.peak_memory == 256.0
        assert profile.python_percentage == 45.0
        assert profile.native_percentage == 5.0
        assert profile.system_percentage == 2.0
        assert profile.timeline_percentage == 10.0
        assert profile.copy_mb_per_s == 50.0
        assert profile.has_memory_info

    def test_function_profile_immutable(self):
        """Test that FunctionProfile is immutable (frozen=True)."""
        profile = FunctionProfile(
            function_name="test",
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
            profile.function_name = "modified"


class TestParseJson:
    """Test the parse_json function."""

    def test_parse_json_nonexistent_file(self, tmp_path: Path):
        """Test parsing a nonexistent JSON file."""
        result = parse_json(tmp_path / "nonexistent.json")
        assert result.functions == []
        assert result.walltime is None
        assert result.max_memory is None

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

        # Check that we got data
        assert result.walltime is not None
        assert result.walltime > 0
        assert result.max_memory is not None
        assert len(result.functions) > 0

        # Check the data is specific to the file we parsed.
        assert result.walltime == 10.986895561218262
        assert result.max_memory == "950.715 MB"
        assert len(result.functions) == 9

        # Verify some functions were parsed
        function_names = [f.function_name for f in result.functions]
        assert any(function_names)  # At least some functions have names

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
        assert func.function_name == "matrix_operations"
        assert func.python_percentage == 0.003940147384761709
        assert func.native_percentage == 0.009041007225759054
        assert func.system_percentage == 0.0038929610193535105
        assert func.peak_memory == 36.31560134887695
        assert func.copy_mb_per_s == 0.0
        assert func.memory_python_percentage == 0.20126943997844543 * 100
        assert func.timeline_percentage == 0.0018903851360701405 * 100


class TestGetTopFunctions:
    """Test the get_top_functions function."""

    def test_get_top_functions_by_total_percentage(self, example_json: Path):
        """Test getting top functions by total percentage."""
        data = parse_json(example_json)
        result = get_top_functions(data.functions, n=2)
        assert len(result) == 2
        assert result[0].function_name == "data_serialization"
        assert result[1].function_name == "fourier_analysis"
        assert result[0].total_percentage == 99.56785634799832
        assert result[1].total_percentage == 0.03309718139989738

    def test_get_top_functions_by_memory(self, example_json: Path):
        """Test getting top functions by memory."""
        data = parse_json(example_json)
        result = get_top_functions(data.functions, n=2, key=lambda f: f.peak_memory)
        assert len(result) == 2
        assert result[0].function_name == "fourier_analysis"
        assert result[1].function_name == "_var"
        assert result[0].peak_memory_info == "153 MB"
        assert result[1].peak_memory_info == "153 MB"

    def test_get_top_functions_empty_list(self):
        """Test getting top functions from empty list."""
        result = get_top_functions([], n=5)
        assert result == []

    def test_get_top_functions_real_profile(self, example_json: Path):
        """Test getting top functions from real profile data."""
        profile_data = parse_json(example_json)
        result = get_top_functions(profile_data.functions, n=10)

        # Should get all 9 functions in the example.
        assert len(result) == 9
        # Verify they're sorted by total percentage (descending)
        for i in range(len(result) - 1):
            assert result[i].total_percentage >= result[i + 1].total_percentage


class TestGetFunctionsByFile:
    """Test the get_functions_by_file function."""

    def test_get_functions_by_file_single_file(self, example_json: Path):
        """Test grouping functions from a single file."""
        profile_data = parse_json(example_json)
        result = get_functions_by_file(profile_data.functions)
        expected_filenames = [
            "/home/ubuntu/fixing-a-hole/performance/advanced.py",
            "/home/ubuntu/fixing-a-hole/.venv/lib/python3.11/site-packages/numpy/_core/_methods.py",
        ]
        assert len(result) == 2
        assert list(result.keys()) == expected_filenames
        for name, n_funcs in zip(expected_filenames, [6, 3], strict=False):
            assert len(result[name]) == n_funcs

    def test_get_functions_by_file_empty_list(self):
        """Test grouping functions from empty list."""
        result = get_functions_by_file([])
        assert len(result) == 0

    def test_get_functions_by_file_real_profile(self, example_json: Path):
        """Test grouping functions from real profile data."""
        profile_data = parse_json(example_json)
        result = get_functions_by_file(profile_data.functions)

        # Should have multiple files
        assert len(result) > 0

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
        by_file = get_functions_by_file(profile_data.functions)
        tree = build_module_tree(by_file)

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
        by_file = get_functions_by_file(profile_data.functions)
        tree = build_module_tree(by_file)
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
        by_file = get_functions_by_file(profile_data.functions)
        tree = build_module_tree(by_file)
        result = render_tree(tree)
        assert len(result) == 13
        expected_tree = [
            "└─ fixing-a-hole (9 func, 99.65% total)",
            "   ├─ performance (6 func, 99.65% total)",
            "   │  └─ advanced.py (6 func, 99.65% total)",
            "   │     └─ data_serialization................................99.57% ( 80 MB)",
            "   │     ",
            "   └─ .venv (3 func, 0.00% total)",
            "      └─ lib (3 func, 0.00% total)",
            "         └─ python3.11 (3 func, 0.00% total)",
            "            └─ site-packages (3 func, 0.00% total)",
            "               └─ numpy (3 func, 0.00% total)",
            "                  └─ _core (3 func, 0.00% total)",
            "                     └─ _methods.py (3 func, 0.00% total)",
            "                        ",
        ]
        assert result == expected_tree

    def test_render_tree_with_threshold(self, example_json: Path):
        """Test rendering tree with threshold filtering."""
        profile_data = parse_json(example_json)
        by_file = get_functions_by_file(profile_data.functions)
        tree = build_module_tree(by_file)
        result = render_tree(tree, threshold=0.0)
        assert len(result) == 21
        expected_tree = [
            "└─ fixing-a-hole (9 func, 99.65% total)",
            "   ├─ performance (6 func, 99.65% total)",
            "   │  └─ advanced.py (6 func, 99.65% total)",
            "   │     ├─ data_serialization................................99.57% ( 80 MB)",
            "   │     ├─ fourier_analysis...................................0.03% (153 MB)",
            "   │     ├─ statistical_analysis...............................0.02% ( 76 MB)",
            "   │     ├─ matrix_operations..................................0.02% ( 36 MB)",
            "   │     ├─ monte_carlo_simulation.............................0.01% ( 76 MB)",
            "   │     └─ recursive_computation..............................0.00%",
            "   │     ",
            "   └─ .venv (3 func, 0.00% total)",
            "      └─ lib (3 func, 0.00% total)",
            "         └─ python3.11 (3 func, 0.00% total)",
            "            └─ site-packages (3 func, 0.00% total)",
            "               └─ numpy (3 func, 0.00% total)",
            "                  └─ _core (3 func, 0.00% total)",
            "                     └─ _methods.py (3 func, 0.00% total)",
            "                        ├─ _mean..............................................0.00%",
            "                        ├─ _var...............................................0.00% (153 MB)",
            "                        └─ _std...............................................0.00%",
            "                        ",
        ]
        assert result == expected_tree


class TestGenerateSummary:
    """Test the generate_summary function."""

    def test_generate_summary_empty_profile(self):
        """Test generating summary from empty profile."""
        profile_data = ProfileData(functions=[], lines={}, walltime=None, max_memory=None)
        result = generate_summary(profile_data)
        assert "No functions to summarize" in result

    def test_generate_summary_real_profile(self, example_json: Path):
        """Test generating summary from real profile data."""
        profile_data = parse_json(example_json)
        result = generate_summary(profile_data, top_n=5)

        expected_summary = [
            "\nProfile Summary (10.987s total)",
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
            "└─ fixing-a-hole (9 func, 99.65% total)",
            "   ├─ performance (6 func, 99.65% total)",
            "   │  └─ advanced.py (6 func, 99.65% total)",
            "   │     └─ data_serialization................................99.57% ( 80 MB)",
            "   │",
            "   └─ .venv (3 func, 0.00% total)",
            "      └─ lib (3 func, 0.00% total)",
            "         └─ python3.11 (3 func, 0.00% total)",
            "            └─ site-packages (3 func, 0.00% total)",
            "               └─ numpy (3 func, 0.00% total)",
            "                  └─ _core (3 func, 0.00% total)",
            "                     └─ _methods.py (3 func, 0.00% total)",
            "",
            "",
            "=================================================================",
            "",
        ]

        assert result == "\n".join(expected_summary)
