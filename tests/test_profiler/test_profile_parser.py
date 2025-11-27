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
"""Tests for the MrKite Profiler."""

from pathlib import Path

import pytest

from fixingahole.profiler import ProfileParser
from fixingahole.profiler.profile_parser import FunctionProfile


class TestProfilerDetailsExtraction:
    """Test the get_details_from_profile method."""

    def test_get_details_from_profile_seconds(self, mock_file: Path):
        """Test extracting details from profile with seconds format."""
        # Write the content to the file instead of mocking
        value = 15.234
        mem_val = "256.5 MB"
        profile_content = f"""
        Some profile output...
        out of {value}s total time.
        max: {mem_val}
        """
        output_file = mock_file.with_suffix(".txt")
        output_file.write_text(profile_content)
        parser = ProfileParser(filename=output_file)

        assert parser.walltime == value
        assert parser.max_memory == mem_val

    def test_get_details_from_profile_milliseconds(self, mock_file: Path):
        """Test extracting details from profile with milliseconds format."""
        value = 1234.567
        mem_val = "128.0 GB"
        profile_content = f"""
        Some profile output...
        out of {value}ms total time.
        max: {mem_val}
        """
        mock_file.write_text(profile_content)
        parser = ProfileParser(filename=mock_file)

        # converted from milliseconds
        assert parser.walltime == value / 1000
        assert parser.max_memory == mem_val

    def test_get_details_from_profile_hours_minutes_seconds(self, mock_file: Path):
        """Test extracting details from profile with h:m:s format."""
        mem_val = "1.5 TB"
        profile_content = f"""
        Some profile output...
        out of 1h:30m:45.123s total time.
        max: {mem_val}
        """
        mock_file.write_text(profile_content)
        parser = ProfileParser(filename=mock_file)

        # 1 hour + 30 minutes + 45.123 seconds = 5445.123 seconds
        expected_time = 1 * 3600 + 30 * 60 + 45.123
        assert parser.walltime == expected_time
        assert parser.max_memory == mem_val

    def test_get_details_from_profile_no_memory_info(self, mock_file: Path):
        """Test extracting details when no memory info is found."""
        value = 15.234
        profile_content = f"""
        Some profile output...
        out of {value}s total time.
        No memory info here.
        """
        mock_file.write_text(profile_content)
        parser = ProfileParser(filename=mock_file)

        assert parser.walltime == value
        assert parser.max_memory == "an unknown amount"

    def test_get_details_from_profile_minutes_seconds(self, mock_file: Path):
        """Test extracting details from profile with m:s format."""
        profile_content = """
        Some profile output...
        out of 5m:30.5s total time.
        max: 512.25 MB
        """
        mock_file.write_text(profile_content)
        parser = ProfileParser(filename=mock_file)

        # 5 minutes + 30.5 seconds = 330.5 seconds
        expected_time = 5 * 60 + 30.5
        assert parser.walltime == expected_time
        assert parser.max_memory == "512.25 MB"


@pytest.fixture
def advanced_profile_txt() -> Path:
    """Return path to the advanced profile results text file."""
    return Path(__file__).parents[1] / "scripts" / "advanced_profile_results.txt"


class TestFunctionProfile:
    """Test the FunctionProfile class."""

    def test_function_profile_initialization_minimal(self):
        """Test FunctionProfile with minimal kwargs."""
        profile = FunctionProfile()
        assert profile.line_number == 0
        assert not profile.function_name
        assert not profile.file_path
        assert profile.memory_python_percentage == 0.0
        assert not profile.peak_memory
        assert profile.memory_size == 0
        assert profile.timeline_percentage == 0.0
        assert profile.copy_mb_per_s == 0.0
        assert profile.total_percentage == 0.0
        assert not profile.has_memory_info

    def test_function_profile_initialization_complete(self):
        """Test FunctionProfile with complete kwargs."""
        profile = FunctionProfile(
            line_number=42,
            function_name="test_func",
            file_path="/path/to/file.py",
            memory_python_percentage="75.5%",
            peak_memory="128M",
            timeline_percentage="10.5%",
            copy_mb_per_s="50.0",
            cpu_percentages=["45.0%", "5.0%", "2.0%"],
        )
        assert profile.line_number == 42
        assert profile.function_name == "test_func"
        assert profile.file_path == "/path/to/file.py"
        assert profile.memory_python_percentage == 75.5
        assert profile.peak_memory == "128M"
        assert profile.memory_size == 128 * 1024 * 1024  # 128 MB in bytes
        assert profile.timeline_percentage == 10.5
        assert profile.copy_mb_per_s == 50.0
        assert profile.total_percentage == 52.0  # 45 + 5 + 2
        assert profile.has_memory_info

    def test_peak_memory_info_formatting(self):
        """Test peak_memory_info property formatting."""
        profile1 = FunctionProfile(peak_memory="128M")
        assert profile1.peak_memory_info == "128 MB"

        profile2 = FunctionProfile(peak_memory="1.5G")
        assert profile2.peak_memory_info == "1.5 GB"

        profile3 = FunctionProfile(peak_memory="")
        assert not profile3.peak_memory_info

    @pytest.mark.parametrize(
        ("memory_str", "expected_bytes"),
        [
            ("100", 100),
            ("1K", 1024),
            ("1KB", 1024),
            ("10M", 10 * 1024 * 1024),
            ("10MB", 10 * 1024 * 1024),
            ("2G", 2 * 1024 * 1024 * 1024),
            ("2GB", 2 * 1024 * 1024 * 1024),
            ("153M", 153 * 1024 * 1024),
        ],
    )
    def test_parse_memory_size(self, memory_str: str, expected_bytes: float):
        """Test memory size parsing to bytes."""
        profile = FunctionProfile(peak_memory=memory_str)
        assert profile.memory_size == expected_bytes

    def test_get_as_float_various_formats(self):
        """Test _get_as_float with various input formats."""
        assert FunctionProfile._get_as_float("45.5%") == 45.5  # noqa: SLF001
        assert FunctionProfile._get_as_float("100") == 100.0  # noqa: SLF001
        assert FunctionProfile._get_as_float("1.234") == 1.234  # noqa: SLF001
        assert FunctionProfile._get_as_float("-5.5") == -5.5  # noqa: SLF001
        assert FunctionProfile._get_as_float("invalid") == 0.0  # noqa: SLF001
        assert FunctionProfile._get_as_float(None) == 0.0  # noqa: SLF001

    def test_has_memory_info_conditions(self):
        """Test has_memory_info with different conditions."""
        # With peak_memory
        profile1 = FunctionProfile(peak_memory="128M")
        assert profile1.has_memory_info

        # With memory_python_percentage
        profile2 = FunctionProfile(memory_python_percentage="75%")
        assert profile2.has_memory_info

        # With timeline_percentage
        profile3 = FunctionProfile(timeline_percentage="10%")
        assert profile3.has_memory_info

        # With none of them
        profile4 = FunctionProfile()
        assert not profile4.has_memory_info


class TestProfileParserParsing:
    """Test the ProfileParser parsing functionality."""

    def test_parse_file_with_advanced_profile(self, advanced_profile_txt: Path):
        """Test parsing the real advanced profile text file."""
        parser = ProfileParser(filename=advanced_profile_txt)

        # Verify walltime was extracted
        assert parser.walltime is not None
        assert parser.walltime == pytest.approx(10.987, rel=1e-3)

        # Verify max memory was extracted
        assert parser.max_memory == "950.715 MB"

        # Verify functions were parsed
        assert parser.functions is not None
        assert len(parser.functions) > 0

    def test_parse_content_function_extraction(self, advanced_profile_txt: Path):
        """Test that functions are properly extracted from content."""
        content = advanced_profile_txt.read_text()
        parser = ProfileParser()
        functions = parser.parse_content(content)

        # Check we found multiple files
        file_paths = {str(f.file_path) for f in functions}
        assert len(file_paths) > 1

        # Check we found the main script
        assert any("advanced.py" in fp for fp in file_paths)

        # Check we found numpy functions
        assert any("numpy" in fp for fp in file_paths)

    def test_parse_content_function_details(self, advanced_profile_txt: Path):
        """Test that function details are correctly parsed."""
        content = advanced_profile_txt.read_text()
        parser = ProfileParser()
        functions = parser.parse_content(content)

        # Find a specific function we know exists
        data_serialization = [f for f in functions if f.function_name == "data_serialization"]
        assert len(data_serialization) > 0

        func = data_serialization[0]
        assert func.line_number == 144
        assert "advanced" in func.file_path
        assert func.has_memory_info
        assert func.peak_memory == "80M"

    def test_parse_content_handles_special_characters(self):
        """Test parsing content with special Unicode box-drawing characters."""
        content = """
/path/to/file.py: % of time = 100%
        │Time   │Memory  │
  Line │Python │peak    │function
╺━━━━━━┿━━━━━━━┿━━━━━━━━┿━━━━━━━━━━━╸
       │       │        │
╶──────┼───────┼────────┼───────────╴
       │       │        │function summary for /path/to/file.py
    42 │  50%  │   10M  │test_function
       ╵       ╵        ╵
"""
        parser = ProfileParser()
        functions = parser.parse_content(content)

        assert len(functions) == 1
        assert functions[0].function_name == "test_function"
        assert functions[0].line_number == 42

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        parser = ProfileParser()
        functions = parser.parse_content("")

        assert functions == []
        assert parser.walltime is None
        assert parser.max_memory == "an unknown amount"

    def test_parse_content_without_function_summary(self):
        """Test parsing content without function summary sections."""
        content = """
        Some profile output...
        out of 10.5s total time.
        max: 256 MB
        No function summaries here.
        """
        parser = ProfileParser()
        functions = parser.parse_content(content)

        assert functions == []
        assert parser.walltime == 10.5
        assert parser.max_memory == "256 MB"


class TestProfileParserSummary:
    """Test the ProfileParser summary generation."""

    def test_summary_with_advanced_profile(self, advanced_profile_txt: Path):
        """Test generating summary from advanced profile."""
        parser = ProfileParser(filename=advanced_profile_txt)
        summary = parser.summary(top_n=5)

        # Check for expected sections
        assert "Profile Summary" in summary
        assert "Top 5 Functions by Total Runtime:" in summary
        assert "Top 5 Functions by Memory Usage:" in summary
        assert "Functions by Module:" in summary

        # Check that walltime is included
        assert "10.987s total" in summary

    def test_summary_top_n_parameter(self, advanced_profile_txt: Path):
        """Test that top_n parameter limits the output."""
        parser = ProfileParser(filename=advanced_profile_txt)
        summary = parser.summary(top_n=3)

        assert "Top 3 Functions" in summary

    def test_summary_without_memory_info(self, mock_file: Path):
        """Test summary generation when functions have no memory info."""
        content = """
        out of 5.0s total time.
        /path/to/file.py: % of time = 100%
╶──────┼───────┼───────────╴
       │       │function summary for /path/to/file.py
    10 │  50%  │func1
    20 │  30%  │func2
       ╵       ╵
"""
        mock_file.write_text(content)
        parser = ProfileParser(filename=mock_file)
        summary = parser.summary()

        # Should not have memory section
        assert "Top" in summary
        assert "Functions by Module:" in summary

    def test_summary_raises_on_no_functions(self):
        """Test that summary raises ValueError when no functions available."""
        parser = ProfileParser()
        with pytest.raises(ValueError, match="Missing functions to summarize"):
            parser.summary()


class TestProfileParserStaticMethods:
    """Test ProfileParser static methods."""

    def test_get_top_functions_by_runtime(self):
        """Test getting top functions by runtime."""
        functions = [
            FunctionProfile(function_name="func1", cpu_percentages=["50%", "5%", "2%"]),
            FunctionProfile(function_name="func2", cpu_percentages=["30%", "3%", "1%"]),
            FunctionProfile(function_name="func3", cpu_percentages=["20%", "2%", "1%"]),
        ]
        top = ProfileParser.get_top_functions(functions, n=2)

        assert len(top) == 2
        assert top[0].function_name == "func1"
        assert top[1].function_name == "func2"

    def test_get_top_functions_by_memory(self):
        """Test getting top functions by memory size."""
        functions = [
            FunctionProfile(function_name="func1", peak_memory="100M"),
            FunctionProfile(function_name="func2", peak_memory="500M"),
            FunctionProfile(function_name="func3", peak_memory="50M"),
        ]
        top = ProfileParser.get_top_functions(functions, n=2, key=lambda f: f.memory_size)

        assert len(top) == 2
        assert top[0].function_name == "func2"  # 500M
        assert top[1].function_name == "func1"  # 100M

    def test_get_functions_by_file(self):
        """Test grouping functions by file path."""
        functions = [
            FunctionProfile(function_name="func1", file_path="/path/file1.py"),
            FunctionProfile(function_name="func2", file_path="/path/file2.py"),
            FunctionProfile(function_name="func3", file_path="/path/file1.py"),
        ]
        by_file = ProfileParser.get_functions_by_file(functions)

        assert len(by_file) == 2
        assert len(by_file["/path/file1.py"]) == 2
        assert len(by_file["/path/file2.py"]) == 1

    def test_build_module_tree(self):
        """Test building hierarchical module tree."""
        by_file = {
            "/home/user/project/module/file1.py": [FunctionProfile(function_name="func1")],
            "/home/user/project/module/file2.py": [FunctionProfile(function_name="func2")],
        }
        tree = ProfileParser.build_module_tree(by_file)

        assert tree is not None
        assert len(tree) > 0

    def test_get_all_functions_in_tree(self):
        """Test extracting all functions from a tree structure."""
        tree = {
            "file1.py": {
                "_functions": [FunctionProfile(function_name="func1")],
                "_children": {},
            },
            "dir": {
                "_functions": [],
                "_children": {
                    "file2.py": {
                        "_functions": [FunctionProfile(function_name="func2")],
                        "_children": {},
                    }
                },
            },
        }
        all_funcs = ProfileParser.get_all_functions_in_tree(tree)

        assert len(all_funcs) == 2
        assert len(all_funcs[0]) == 1
        assert len(all_funcs[1]) == 1

    def test_render_tree_basic(self):
        """Test rendering a basic tree structure."""
        tree = {
            "file.py": {
                "_functions": [FunctionProfile(function_name="func1", cpu_percentages=["50%", "5%", "2%"])],
                "_children": {},
            }
        }
        lines = ProfileParser.render_tree(tree)

        assert len(lines) > 0
        assert any("file.py" in line for line in lines)
        assert any("func1" in line for line in lines)

    def test_render_tree_nested(self):
        """Test rendering nested tree structure."""
        tree = {
            "module": {
                "_functions": [],
                "_children": {
                    "submodule": {
                        "_functions": [],
                        "_children": {
                            "file.py": {
                                "_functions": [
                                    FunctionProfile(
                                        function_name="nested_func",
                                        cpu_percentages=["30%", "2%", "1%"],
                                    )
                                ],
                                "_children": {},
                            }
                        },
                    }
                },
            }
        }
        lines = ProfileParser.render_tree(tree)

        result = "\n".join(lines)
        assert "module" in result
        assert "submodule" in result
        assert "file.py" in result
        assert "nested_func" in result


class TestProfileParserRealWorldScenarios:
    """Test ProfileParser with real-world scenarios."""

    def test_full_parsing_workflow(self, advanced_profile_txt: Path):
        """Test complete parsing workflow with advanced profile."""
        # Parse the file
        parser = ProfileParser(filename=advanced_profile_txt)

        # Verify parsing succeeded
        assert parser.functions is not None
        assert len(parser.functions) > 10

        # Generate summary
        summary = parser.summary(top_n=10)
        assert len(summary) > 100  # Should be substantial

        # Check for specific known functions
        func_names = [f.function_name for f in parser.functions]
        assert "data_serialization" in func_names
        assert "_raw_fft" in func_names

    def test_multiple_files_parsing(self, advanced_profile_txt: Path):
        """Test that multiple files are correctly parsed."""
        parser = ProfileParser(filename=advanced_profile_txt)
        by_file = ProfileParser.get_functions_by_file(parser.functions)

        # Should have multiple files
        assert len(by_file) > 5

        # Check for expected file patterns
        file_paths = list(by_file.keys())
        assert any("advanced" in fp for fp in file_paths)
        assert any("numpy" in fp for fp in file_paths)
        # Verify we have standard library modules
        assert any("python" in fp.lower() for fp in file_paths)

    def test_memory_statistics(self, advanced_profile_txt: Path):
        """Test memory-related statistics from parsed data."""
        parser = ProfileParser(filename=advanced_profile_txt)

        # Find functions with memory info
        memory_funcs = [f for f in parser.functions if f.has_memory_info]
        assert len(memory_funcs) > 0

        # Check memory sizes are properly parsed
        memory_sizes = [f.memory_size for f in memory_funcs if f.peak_memory]
        assert all(size > 0 for size in memory_sizes)

        # Find the function with highest memory usage
        top_memory = max(parser.functions, key=lambda f: f.memory_size)
        assert top_memory.peak_memory
