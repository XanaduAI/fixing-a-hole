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
"""Tests for the Profiler."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fixingahole import LogLevel, ProfilerConfig
from fixingahole.profiler import Profiler
from fixingahole.profiler.profiler import ProfilerException, SuccessfulExit


class TestProfilerRunProfiler:
    """Test the run_profiler method."""

    def test_run_profiler_success(self, mock_file: Path):
        """Test successful profiler run."""
        profiler = Profiler(
            mock_file,
            precision=5,
            log_level=LogLevel.INFO,
            live_update=1,
        )
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)
        assert exc_info.value.exit_code == 0

    def test_run_profiler_cpu_only(self, mock_file: Path):
        """Test successful profiler run using only CPU."""
        profiler = Profiler(mock_file, cpu_only=True)
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)
        assert exc_info.value.exit_code == 0

    def test_run_profiler_detailed_mode(self, mock_file: Path):
        """Test profiler run with detailed profiling enabled."""
        profiler = Profiler(mock_file, precision=5, detailed=True)
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 0
        assert "numpy" in profiler.profile_file.read_text()

    @patch("subprocess.run")
    def test_run_profiler_subprocess_error(self, mock_run: MagicMock, mock_file: Path):
        """Test profiler run handling subprocess errors."""
        # Setup subprocess to raise CalledProcessError
        error = subprocess.CalledProcessError(1, ["cmd"])
        error.stdout = b"stdout error message"
        error.stderr = b"stderr error message"
        mock_run.side_effect = error

        profiler = Profiler(mock_file, precision=5)

        with pytest.raises(ProfilerException) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 1
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_profiler_keyboard_interrupt(self, mock_run: MagicMock, mock_file: Path):
        """Test profiler run handling keyboard interrupt."""
        # Setup subprocess to raise KeyboardInterrupt.
        mock_run.side_effect = KeyboardInterrupt()

        profiler = Profiler(mock_file, precision=5)

        with pytest.raises(ProfilerException) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 1
        mock_run.assert_called()
        assert "Profiling interrupted by user." in profiler.output_file.read_text()

    def test_run_profiler_with_script_args_and_logs(self, mock_file: Path):
        """Test profiler run with script arguments."""
        args = ["arg1=value1", "arg2=value2"]
        profiler = Profiler(
            mock_file,
            python_script_args=args,
            precision=5,
            log_level=LogLevel.INFO,
        )
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 0
        logs = profiler.log_file.read_text()
        for arg in args:
            assert arg in logs
        assert "This is a warning." in logs
        # Check that warning count is correctly calculated and included
        final_content = profiler.output_summary.read_text()
        assert "Check logs" in final_content
        assert "(1 warning)" in final_content

    def test_run_profiler_with_script_argparse(self, mock_file_with_argparse: Path):
        """Test profiler run with script arguments."""
        args = ["--base", "10", "--power", "5"]
        profiler = Profiler(
            mock_file_with_argparse,
            python_script_args=args,
            precision=5,
            log_level=LogLevel.INFO,
        )
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 0
        logs = profiler.log_file.read_text()
        for arg in args:
            assert arg in logs
        assert "This is a warning." in logs
        # Check that warning count is correctly calculated and included
        final_content = profiler.output_summary.read_text()
        assert "Check logs" in final_content
        assert "(1 warning)" in final_content

    def test_run_profiler_with_config_object(self, mock_file: Path, root_dir: Path):
        """Test profiler run using ProfilerConfig object."""

        class CustomConfig(ProfilerConfig):
            """A custom config for profiling."""

            def setup(self, profiler: Profiler) -> None:
                """Set up profiler with custom configuration."""
                output = root_dir / "config_test_output"
                output.mkdir(parents=True, exist_ok=True)

                profiler.python_file = mock_file
                profiler.filestem = "config_test"
                profiler.profile_root = output
                profiler.output_file = output / "config_results.txt"

        config = CustomConfig()
        profiler = Profiler(config, precision=3, log_level=LogLevel.INFO)

        # Verify config was applied
        assert profiler.filestem == "config_test"
        assert profiler.output_file.name == "config_results.txt"

        # Run the profiler
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 0

        # Verify outputs were created
        assert profiler.output_file.exists()
        assert profiler.output_json.exists()
        assert profiler.output_summary.exists()

        # Verify the command ran successfully
        summary_content = profiler.output_summary.read_text()
        assert "Finished in" in summary_content
