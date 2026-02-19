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
"""Tests for the profile summarizer module."""

import json
from pathlib import Path

import pytest

from fixingahole.profiler.profile_summary import (
    ProfileSummary,
    build_module_tree,
    generate_summary,
    get_all_functions_in_tree,
    render_tree,
)
from fixingahole.profiler.scalene_json_parser import ProfileData


@pytest.fixture
def advanced_profile_json() -> Path:
    """Return the path to the advanced profile results JSON file."""
    return Path(__file__).parent.parent / "scripts" / "data" / "advanced_profile_results.json"


class TestBuildModuleTree:
    """Test the build_module_tree function."""

    def test_build_module_tree_real_profile(self, example_json: Path):
        """Test building tree from real profile data."""
        profile_data = ProfileData.from_file(example_json)
        tree, depth = build_module_tree(profile_data.functions_by_file)

        # Should create a hierarchical structure
        assert tree is not None
        assert len(tree) > 0
        assert depth == 4

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
        profile_data = ProfileData.from_file(example_json)
        tree, depth = build_module_tree(profile_data.functions_by_file)
        result = get_all_functions_in_tree(tree)
        assert len(result) == 2
        assert depth == 4
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
        profile_data = ProfileData.from_file(example_json)
        tree, depth = build_module_tree(profile_data.functions_by_file)
        result = render_tree(tree, profile_data.walltime, threshold=0)
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
        assert depth == 4
        assert len(result) == len(expected_tree)
        assert result == expected_tree

    def test_render_tree_with_threshold(self, example_json: Path):
        """Test rendering tree with threshold filtering."""
        profile_data = ProfileData.from_file(example_json)
        tree, depth = build_module_tree(profile_data.functions_by_file)
        result = render_tree(tree, profile_data.walltime, threshold=0.0)
        assert depth == 4
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

    def test_generate_summary_empty_profile(self, example_json: Path):
        """Test generating summary from a profile with no functions."""
        # Manipulate the file to remove all the functions data.
        json_data = json.loads(example_json.read_text(encoding="utf-8"))
        for file in json_data["files"]:
            json_data["files"][file]["functions"] = []

        # Load the manipulated data.
        profile_data = ProfileData(**json_data)
        result = generate_summary(profile_data)
        assert "No functions to summarize" in result

    def test_generate_summary_real_profile(self, example_json: Path):
        """Test generating summary from real profile data."""
        profile_data = ProfileData.from_file(example_json)
        result = generate_summary(profile_data, top_n=10, threshold=0)

        expected_summary = [
            "\nProfile Summary",
            "=========================================================",
            "\nTop 9 Functions by Total Runtime:",
            "---------------------------------------------------------",
            " 1. data_serialization        99.57% (advanced.py:144)",
            " 2. fourier_analysis           0.03% (advanced.py:108)",
            " 3. statistical_analysis       0.02% (advanced.py:72)",
            " 4. matrix_operations          0.02% (advanced.py:35)",
            " 5. monte_carlo_simulation     0.01% (advanced.py:56)",
            " 6. _mean                      0.00% (_methods.py:117)",
            " 7. _var                       0.00% (_methods.py:150)",
            " 8. recursive_computation      0.00% (advanced.py:135)",
            " 9. _std                       0.00% (_methods.py:220)",
            "\nTop 6 Functions by Memory Usage:",
            "---------------------------------------------------------",
            " 1. fourier_analysis            153 MB (advanced.py:108)",
            " 2. _var                        153 MB (_methods.py:150)",
            " 3. data_serialization           80 MB (advanced.py:144)",
            " 4. statistical_analysis         76 MB (advanced.py:72)",
            " 5. monte_carlo_simulation       76 MB (advanced.py:56)",
            " 6. matrix_operations            36 MB (advanced.py:35)",
            "\nFunctions by Module:",
            "---------------------------------------------------------",
            "├─ performance (6 func, 99.65% total)",
            "│  └─ advanced.py (6 func, 99.65% total)",
            "│     ├─ data_serialization..........99.57% ( 80 MB)",
            "│     ├─ fourier_analysis.............0.03% (153 MB)",
            "│     ├─ statistical_analysis.........0.02% ( 76 MB)",
            "│     ├─ matrix_operations............0.02% ( 36 MB)",
            "│     ├─ monte_carlo_simulation.......0.01% ( 76 MB)",
            "│     └─ recursive_computation........0.00%",
            "│",
            "└─ numpy (3 func, 0.00% total)",
            "   └─ _core (3 func, 0.00% total)",
            "      └─ _methods.py (3 func, 0.00% total)",
            "         ├─ _mean.....................0.00%",
            "         ├─ _var......................0.00% (153 MB)",
            "         └─ _std......................0.00%",
            "\n\n=========================================================\n",
        ]
        expected_summary = "\n".join(expected_summary)
        assert result == expected_summary


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

    def test_nonexistent_file(self, tmp_path: Path):
        """Test handling of non-existent profile file."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError) as exc:
            ProfileSummary(filename=nonexistent)
        assert exc.value.errno == 2
