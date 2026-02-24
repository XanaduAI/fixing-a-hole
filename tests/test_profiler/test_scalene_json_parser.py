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
"""Tests for the Scalene JSON parser module."""

from pathlib import Path

import pytest

from fixingahole.profiler.scalene_json_parser import ProfileData
from fixingahole.profiler.utils import memory_with_units


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


class TestProfileData:
    """Test initializing the ProfileData class from file."""

    def test_parse_json_nonexistent_file(self, tmp_path: Path):
        """Test parsing a nonexistent JSON file."""
        with pytest.raises(FileNotFoundError) as exc:
            ProfileData.from_file(tmp_path / "nonexistent.json")
        assert exc.value.errno == 2

    def test_parse_json_empty_file(self, tmp_path: Path):
        """Test parsing an empty JSON file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")
        with pytest.raises(TypeError, match="missing 19 required positional arguments"):
            ProfileData.from_file(json_file)

    def test_parse_json_complete_profile(self, example_json: Path):
        """Test parsing a complete JSON profile."""
        result = ProfileData.from_file(example_json)

        # Parser contract: basic shape and aggregate metrics
        assert result.walltime == pytest.approx(10.986895561218262)
        assert result.max_memory == "950.715 MB"
        assert len(result.functions) == 9

        # Spot-check first function: identity and computed fields
        first = result.functions[0]
        assert first.file_path == "/home/ubuntu/fixing-a-hole/performance/advanced.py"
        assert first.line_number == 35
        assert first.name == "matrix_operations"
        assert first.python_percentage == pytest.approx(0.003940147384761709)
        assert first.native_percentage == pytest.approx(0.009041007225759054)
        assert first.system_percentage == pytest.approx(0.0038929610193535105)
        assert first.peak_memory == pytest.approx(36.31560134887695)
        assert first.copy_mb_per_s == pytest.approx(0.0)
        assert first.memory_python_percentage == pytest.approx(0.20126943997844543 * 100)
        assert first.timeline_percentage == pytest.approx(0.0018903851360701405 * 100)
        assert first.total_time == pytest.approx(first.total_percentage * first.walltime / 100)

    def test_get_all_line_data(self, example_json: Path):
        """Test getting all the line data."""
        data = ProfileData.from_file(example_json)
        assert set(data.lines) == set(data.files)
        assert len(set(data.lines)) == 2
        # Assert that the correct number of line details are found (based on the test data).
        lines_in_each_file: list[int] = [53, 21]
        for i, lines in enumerate(data.lines.values()):
            assert len(lines) == lines_in_each_file[i]


class TestGetFunctionsByFile:
    """Test the functions_by_file function."""

    def test_functions_by_file_real_file(self, example_json: Path):
        """Test grouping functions from a single file."""
        profile_data = ProfileData.from_file(example_json)
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
