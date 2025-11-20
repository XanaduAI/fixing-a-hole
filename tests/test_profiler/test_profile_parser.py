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

from fixingahole.profiler import ProfileParser


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
