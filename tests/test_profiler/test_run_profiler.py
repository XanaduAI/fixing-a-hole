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

import sys
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

    def test_scalene_run_cmd_preserves_argument_boundaries(self, tmp_path: Path):
        script = tmp_path / "script with spaces.py"
        script.write_text("print('hello')\n", encoding="utf-8")
        output_dir = tmp_path / "output with spaces"

        profiler = Profiler(
            script,
            output_dir=output_dir,
            python_script_args=["argument with spaces"],
        )

        command = profiler._scalene_run_cmd  # noqa: SLF001

        assert command[:4] == [sys.executable, "-m", "scalene", "run"]
        assert str(script) in command
        assert str(profiler.output_json) in command
        assert "argument with spaces" in command

    @patch("os.wait4")
    @patch("subprocess.Popen")
    def test_run_profiler_subprocess_error(self, mock_popen: MagicMock, mock_wait4: MagicMock, mock_file: Path):
        """Test that a non-zero subprocess exit code is surfaced as a ProfilerException."""
        mock_rusage = MagicMock()
        mock_rusage.ru_maxrss = 1024 * 1024  # 1 MiB in bytes
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.stderr = iter(["expected stderr error message\n"])
        mock_popen.return_value = mock_proc
        mock_wait4.return_value = (12345, 256, mock_rusage)  # raw status 256 → exit code 1

        profiler = Profiler(mock_file, cpu_only=False, precision=5)

        with pytest.raises(ProfilerException) as exc_info, patch("os.waitstatus_to_exitcode", return_value=1):
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 1
        assert "stderr error message" in exc_info.value.message
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
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
        assert all(phrase in logs for phrase in ["This is a warning.", "This is an error.", "This is critical."])
        # Check that warning count is correctly calculated and included
        final_content = profiler.output_summary.read_text()
        assert "Check logs" in final_content
        assert "(3 INFO, 1 WARNING, 1 ERROR, 1 CRITICAL)" in final_content

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
        assert "(2 INFO, 1 WARNING)" in final_content

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
