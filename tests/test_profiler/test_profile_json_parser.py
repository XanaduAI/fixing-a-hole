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

import json
from pathlib import Path

import pytest

from fixingahole.profiler.profile_json_parser import (
    FunctionProfile,
    ProfileData,
    build_module_tree,
    create_function_profile,
    generate_summary,
    get_all_functions_in_tree,
    get_functions_by_file,
    get_top_functions,
    memory_with_units,
    parse_json,
    render_tree,
)


@pytest.fixture
def advanced_profile_json() -> Path:
    """Return path to the advanced profile results JSON file."""
    return Path(__file__).parents[1] / "scripts" / "advanced_profile_results.json"


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

    def test_create_function_profile_minimal(self):
        """Test creating a function profile with minimal data."""
        profile = create_function_profile()
        assert not profile.function_name
        assert not profile.file_path
        assert profile.line_number == 0
        assert profile.memory_python_percentage == 0.0
        assert profile.peak_memory == 0.0
        assert profile.python_percentage == 0.0
        assert profile.native_percentage == 0.0
        assert profile.system_percentage == 0.0
        assert profile.timeline_percentage == 0.0
        assert profile.copy_mb_per_s == 0.0
        assert not profile.has_memory_info

    def test_create_function_profile_complete(self):
        """Test creating a function profile with complete data."""
        profile = create_function_profile(
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
        assert profile.has_memory_info

    def test_create_function_profile_has_memory_info_conditions(self):
        """Test has_memory_info with different combinations."""
        # Only peak_memory
        profile1 = create_function_profile(peak_memory=10.0)
        assert profile1.has_memory_info

        # Only memory_python_percentage
        profile2 = create_function_profile(memory_python_percentage=5.0)
        assert profile2.has_memory_info

        # Only timeline_percentage
        profile3 = create_function_profile(timeline_percentage=3.0)
        assert profile3.has_memory_info

        # None of them
        profile4 = create_function_profile()
        assert not profile4.has_memory_info


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
            has_memory_info=True,
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
            has_memory_info=False,
        )
        with pytest.raises(AttributeError):
            profile.function_name = "modified"

    def test_peak_memory_info(self):
        """Test peak_memory_info property."""
        profile = create_function_profile(peak_memory=256.0)
        assert profile.peak_memory_info == "256 MB"

    def test_total_percentage(self):
        """Test total_percentage property."""
        profile = create_function_profile(
            python_percentage=45.0,
            native_percentage=5.0,
            system_percentage=2.0,
        )
        assert profile.total_percentage == 52.0

    def test_total_percentage_all_zeros(self):
        """Test total_percentage when all percentages are zero."""
        profile = create_function_profile()
        assert profile.total_percentage == 0.0


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
        assert result.walltime == -1
        # Memory format may vary based on size, just check it's not None
        assert result.max_memory is not None
        assert "-1" in result.max_memory

    def test_parse_json_complete_profile(self, tmp_path: Path):
        """Test parsing a complete JSON profile."""
        json_file = tmp_path / "profile.json"
        profile_data = {
            "elapsed_time_sec": 10.5,
            "max_footprint_mb": 512.25,
            "files": {
                "/path/to/file.py": {
                    "functions": [
                        {
                            "lineno": 42,
                            "line": "def test_function():",
                            "n_cpu_percent_python": 45.5,
                            "n_cpu_percent_c": 5.2,
                            "n_sys_percent": 1.3,
                            "n_peak_mb": 128.0,
                            "n_copy_mb_s": 50.0,
                            "n_python_fraction": 0.75,
                            "n_usage_fraction": 0.10,
                        }
                    ]
                }
            },
        }
        json_file.write_text(json.dumps(profile_data))
        result = parse_json(json_file)

        assert result.walltime == 10.5
        assert result.max_memory == "512.250 MB"
        assert len(result.functions) == 1

        func = result.functions[0]
        assert func.file_path == "/path/to/file.py"
        assert func.line_number == 42  # No offset since file doesn't exist
        assert func.function_name == "def test_function():"
        assert func.python_percentage == 45.5
        assert func.native_percentage == 5.2
        assert func.system_percentage == 1.3
        assert func.peak_memory == 128.0
        assert func.copy_mb_per_s == 50.0
        assert func.memory_python_percentage == 75.0
        assert func.timeline_percentage == 10.0

    def test_parse_json_multiple_files_and_functions(self, tmp_path: Path):
        """Test parsing JSON with multiple files and functions."""
        json_file = tmp_path / "profile.json"
        profile_data = {
            "elapsed_time_sec": 15.0,
            "max_footprint_mb": 256.0,
            "files": {
                "/path/to/file1.py": {
                    "functions": [
                        {"lineno": 30, "line": "func1", "n_cpu_percent_python": 20.0},
                        {"lineno": 50, "line": "func2", "n_cpu_percent_python": 15.0},
                    ]
                },
                "/path/to/file2.py": {
                    "functions": [
                        {"lineno": 10, "line": "func3", "n_cpu_percent_python": 10.0},
                    ]
                },
            },
        }
        json_file.write_text(json.dumps(profile_data))
        result = parse_json(json_file)

        assert len(result.functions) == 3
        assert result.walltime == 15.0

    def test_parse_json_missing_optional_fields(self, tmp_path: Path):
        """Test parsing JSON with missing optional fields."""
        json_file = tmp_path / "profile.json"
        profile_data = {
            "elapsed_time_sec": 5.0,
            "files": {
                "/path/to/file.py": {
                    "functions": [
                        {
                            "line": "minimal_func",
                        }
                    ]
                }
            },
        }
        json_file.write_text(json.dumps(profile_data))
        result = parse_json(json_file)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.function_name == "minimal_func"
        assert func.line_number == 0  # No offset since file doesn't exist
        assert func.python_percentage == 0.0

    def test_parse_json_real_profile(self, advanced_profile_json: Path):
        """Test parsing a real advanced profile JSON file."""
        result = parse_json(advanced_profile_json)

        # Check that we got data
        assert result.walltime is not None
        assert result.walltime > 0
        assert result.max_memory is not None
        assert len(result.functions) > 0

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


class TestGetTopFunctions:
    """Test the get_top_functions function."""

    def test_get_top_functions_by_total_percentage(self):
        """Test getting top functions by total percentage."""
        functions = [
            create_function_profile(function_name="func1", python_percentage=50.0),
            create_function_profile(function_name="func2", python_percentage=30.0),
            create_function_profile(function_name="func3", python_percentage=10.0),
        ]
        result = get_top_functions(functions, n=2)
        assert len(result) == 2
        assert result[0].function_name == "func1"
        assert result[1].function_name == "func2"

    def test_get_top_functions_by_memory(self):
        """Test getting top functions by memory."""
        functions = [
            create_function_profile(function_name="func1", peak_memory=100.0),
            create_function_profile(function_name="func2", peak_memory=200.0),
            create_function_profile(function_name="func3", peak_memory=50.0),
        ]
        result = get_top_functions(functions, n=2, key=lambda f: f.peak_memory)
        assert len(result) == 2
        assert result[0].function_name == "func2"
        assert result[1].function_name == "func1"

    def test_get_top_functions_empty_list(self):
        """Test getting top functions from empty list."""
        result = get_top_functions([], n=5)
        assert result == []

    def test_get_top_functions_real_profile(self, advanced_profile_json: Path):
        """Test getting top functions from real profile data."""
        profile_data = parse_json(advanced_profile_json)
        result = get_top_functions(profile_data.functions, n=10)

        # Should get up to 10 functions
        assert len(result) <= 10
        if len(result) > 1:
            # Verify they're sorted by total percentage (descending)
            for i in range(len(result) - 1):
                assert result[i].total_percentage >= result[i + 1].total_percentage


class TestGetFunctionsByFile:
    """Test the get_functions_by_file function."""

    def test_get_functions_by_file_single_file(self):
        """Test grouping functions from a single file."""
        functions = [
            create_function_profile(function_name="func1", file_path="/path/file.py"),
            create_function_profile(function_name="func2", file_path="/path/file.py"),
        ]
        result = get_functions_by_file(functions)
        assert len(result) == 1
        assert "/path/file.py" in result
        assert len(result["/path/file.py"]) == 2

    def test_get_functions_by_file_multiple_files(self):
        """Test grouping functions from multiple files."""
        functions = [
            create_function_profile(function_name="func1", file_path="/path/file1.py"),
            create_function_profile(function_name="func2", file_path="/path/file2.py"),
            create_function_profile(function_name="func3", file_path="/path/file1.py"),
        ]
        result = get_functions_by_file(functions)
        assert len(result) == 2
        assert len(result["/path/file1.py"]) == 2
        assert len(result["/path/file2.py"]) == 1

    def test_get_functions_by_file_empty_list(self):
        """Test grouping functions from empty list."""
        result = get_functions_by_file([])
        assert len(result) == 0

    def test_get_functions_by_file_real_profile(self, advanced_profile_json: Path):
        """Test grouping functions from real profile data."""
        profile_data = parse_json(advanced_profile_json)
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

    def test_build_module_tree_single_file(self):
        """Test building tree from single file."""
        by_file = {"test.py": [create_function_profile(function_name="func1")]}
        tree = build_module_tree(by_file)
        assert "test.py" in tree
        assert len(tree["test.py"]["_functions"]) == 1

    def test_build_module_tree_nested_path(self):
        """Test building tree from nested path."""
        by_file = {"/path/to/module/file.py": [create_function_profile(function_name="func1")]}
        tree = build_module_tree(by_file)
        assert tree is not None

    def test_build_module_tree_multiple_files(self):
        """Test building tree from multiple files."""
        by_file = {
            "/path/file1.py": [create_function_profile(function_name="func1")],
            "/path/file2.py": [create_function_profile(function_name="func2")],
        }
        tree = build_module_tree(by_file)
        assert len(tree) > 0

    def test_build_module_tree_real_profile(self, advanced_profile_json: Path):
        """Test building tree from real profile data."""
        profile_data = parse_json(advanced_profile_json)
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

    def test_get_all_functions_in_tree_flat(self):
        """Test getting functions from flat tree."""
        tree = {
            "file1.py": {
                "_functions": [create_function_profile(function_name="func1")],
                "_children": {},
            },
            "file2.py": {
                "_functions": [create_function_profile(function_name="func2")],
                "_children": {},
            },
        }
        result = get_all_functions_in_tree(tree)
        assert len(result) == 2

    def test_get_all_functions_in_tree_nested(self):
        """Test getting functions from nested tree."""
        tree = {
            "dir": {
                "_functions": [],
                "_children": {
                    "file.py": {
                        "_functions": [create_function_profile(function_name="func1")],
                        "_children": {},
                    }
                },
            }
        }
        result = get_all_functions_in_tree(tree)
        assert len(result) == 1

    def test_get_all_functions_in_tree_empty(self):
        """Test getting functions from empty tree."""
        result = get_all_functions_in_tree({})
        assert result == []


class TestRenderTree:
    """Test the render_tree function."""

    def test_render_tree_single_file(self):
        """Test rendering tree with single file."""
        tree = {
            "file.py": {
                "_functions": [create_function_profile(function_name="func1", python_percentage=50.0)],
                "_children": {},
            }
        }
        result = render_tree(tree)
        assert len(result) > 0
        assert any("file.py" in line for line in result)
        assert any("func1" in line for line in result)

    def test_render_tree_with_threshold(self):
        """Test rendering tree with threshold filtering."""
        tree = {
            "file.py": {
                "_functions": [
                    create_function_profile(function_name="high_func", python_percentage=50.0),
                    create_function_profile(function_name="low_func", python_percentage=0.05),
                ],
                "_children": {},
            }
        }
        result = render_tree(tree, threshold=0.1)
        result_str = "\n".join(result)
        assert "high_func" in result_str
        assert "low_func" not in result_str

    def test_render_tree_nested_structure(self):
        """Test rendering nested tree structure."""
        tree = {
            "dir": {
                "_functions": [],
                "_children": {
                    "file.py": {
                        "_functions": [create_function_profile(function_name="func1", python_percentage=30.0)],
                        "_children": {},
                    }
                },
            }
        }
        result = render_tree(tree)
        assert len(result) > 0
        assert any("dir" in line for line in result)


class TestGenerateSummary:
    """Test the generate_summary function."""

    def test_generate_summary_empty_profile(self):
        """Test generating summary from empty profile."""
        profile_data = ProfileData(functions=[], walltime=None, max_memory=None)
        result = generate_summary(profile_data)
        assert "No functions to summarize" in result

    def test_generate_summary_basic(self):
        """Test generating basic summary."""
        functions = [
            create_function_profile(
                function_name="func1",
                file_path="/path/to/file.py",
                line_number=10,
                python_percentage=50.0,
                native_percentage=5.0,
                system_percentage=2.0,
            ),
            create_function_profile(
                function_name="func2",
                file_path="/path/to/file.py",
                line_number=20,
                python_percentage=30.0,
            ),
        ]
        profile_data = ProfileData(functions=functions, walltime=10.5, max_memory="256 MB")
        result = generate_summary(profile_data, top_n=5)

        assert "Profile Summary" in result
        assert "10.500s total" in result
        assert "Top" in result
        assert "Functions by Module" in result

    def test_generate_summary_with_memory(self):
        """Test generating summary with memory information."""
        functions = [
            create_function_profile(
                function_name="memory_func",
                file_path="/path/to/file.py",
                line_number=10,
                python_percentage=40.0,
                peak_memory=512.0,
            ),
        ]
        profile_data = ProfileData(functions=functions, walltime=5.0, max_memory="1 GB")
        result = generate_summary(profile_data, top_n=5)

        assert "Memory Usage" in result
        assert "memory_func" in result

    def test_generate_summary_top_n_limit(self):
        """Test that top_n parameter limits output correctly."""
        functions = [
            create_function_profile(
                function_name=f"func{i}",
                file_path="/path/to/file.py",
                python_percentage=float(50 - i * 5),
            )
            for i in range(20)
        ]
        profile_data = ProfileData(functions=functions, walltime=10.0, max_memory="256 MB")
        result = generate_summary(profile_data, top_n=3)

        # Check that it mentions "Top 3" not "Top 20"
        assert "Top 3" in result

    def test_generate_summary_threshold_filtering(self):
        """Test that threshold parameter filters functions correctly."""
        functions = [
            create_function_profile(
                function_name="high_func",
                file_path="/path/to/file.py",
                python_percentage=50.0,
            ),
            create_function_profile(
                function_name="low_func",
                file_path="/path/to/file.py",
                python_percentage=0.05,
            ),
        ]
        profile_data = ProfileData(functions=functions, walltime=10.0, max_memory="256 MB")
        result = generate_summary(profile_data, top_n=10, threshold=0.1)

        # low_func should be filtered out by threshold in the tree view
        assert "high_func" in result

    def test_generate_summary_real_profile(self, advanced_profile_json: Path):
        """Test generating summary from real profile data."""
        profile_data = parse_json(advanced_profile_json)
        result = generate_summary(profile_data, top_n=10, threshold=0.1)

        # Should contain expected sections
        assert "Profile Summary" in result
        assert "Top" in result
        assert "Functions by Module" in result

        # Should contain walltime information
        if profile_data.walltime:
            assert f"{profile_data.walltime:,.3f}s total" in result

        # Should be a multi-line string
        lines = result.split("\n")
        assert len(lines) > 10


class TestProfileData:
    """Test the ProfileData dataclass."""

    def test_profile_data_creation(self):
        """Test creating ProfileData instance."""
        functions = [create_function_profile(function_name="test")]
        data = ProfileData(functions=functions, walltime=10.0, max_memory="256 MB")

        assert len(data.functions) == 1
        assert data.walltime == 10.0
        assert data.max_memory == "256 MB"

    def test_profile_data_empty(self):
        """Test creating empty ProfileData."""
        data = ProfileData(functions=[], walltime=None, max_memory=None)

        assert data.functions == []
        assert data.walltime is None
        assert data.max_memory is None
