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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fixingahole import LogLevel, ProfilerConfig
from fixingahole.profiler import Profiler
from fixingahole.profiler.profiler import ProfilerException, SuccessfulExit


class TestProfilerRunProfiler:
    """Test the run_profiler method."""

    def test_run_profiler_success(self, mock_file: Path, capfd: pytest.CaptureFixture[str]):
        """Periodic profiling preserves diagnostics and suppresses save/view chatter."""
        profiler = Profiler(
            mock_file,
            log_level=LogLevel.INFO,
            profiler_args=["--profile-interval", "1"],
        )
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)
        assert exc_info.value.exit_code == 0
        assert profiler.output_json.exists()
        assert "heap RAM" in profiler.output_summary.read_text()
        stderr = capfd.readouterr().err
        assert "Scalene: profile saved" not in stderr
        assert "To view in" not in stderr
        assert "This is a warning." in stderr

    def test_run_profiler_cpu_only(self, mock_file: Path):
        """Test CPU profiling of all modules with the default system Python exclusion."""
        profiler = Profiler(mock_file, profiler_args=["--cpu-only", "--profile-all"])
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)
        assert exc_info.value.exit_code == 0
        profile = json.loads(profiler.output_json.read_text())
        assert profile["memory"] is False
        assert str(mock_file) in profile["files"]
        assert not any(Path(filename).name == "encoder.py" for filename in profile["files"])

    def test_run_profiler_subprocess_error(self, mock_file: Path, capfd: pytest.CaptureFixture[str]):
        """Invalid arguments are rejected by Scalene when profiling starts."""
        profiler = Profiler(mock_file, profiler_args=["--allocation-sampling-window", "invalid"])
        with pytest.raises(ProfilerException) as exc_info:
            profiler.run_profiler()
        assert "--allocation-sampling-window" in exc_info.value.message
        assert exc_info.value.exit_code == 2
        assert "--allocation-sampling-window" in capfd.readouterr().err

    def test_run_profiler_does_not_summarize_stale_output(self, mock_file: Path, example_json: Path):
        """An overridden outfile cannot turn an old profile into a successful result."""
        alternate = mock_file.parent / "alternate.json"
        profiler = Profiler(mock_file, profiler_args=["--cpu-only", "--no-stacks", "--outfile", str(alternate)])
        profiler.output_json.write_bytes(example_json.read_bytes())

        with pytest.raises(ProfilerException) as exc_info:
            profiler.run_profiler()

        assert "empty or unavailable" in exc_info.value.message
        assert alternate.exists()
        assert not profiler.output_json.exists()

    @patch("subprocess.Popen")
    def test_run_profiler_keyboard_interrupt(self, mock_run: MagicMock, mock_file: Path):
        """Test profiler run handling keyboard interrupt."""
        # Setup subprocess to raise KeyboardInterrupt.
        mock_run.side_effect = KeyboardInterrupt()

        profiler = Profiler(mock_file)

        with pytest.raises(ProfilerException) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 1
        mock_run.assert_called()
        assert "Profiling interrupted by user." in profiler.output_file.read_text()

    def test_run_profiler_with_script_args_and_logs(self, mock_file: Path):
        """Test profiler run with script arguments."""
        mock_file = mock_file.rename(mock_file.with_name("script with spaces.py"))
        args = ["--repeat", "7", "--log-level", "DEBUG", "--precision", "5", "--help", "value with spaces"]
        profiler = Profiler(
            mock_file,
            script_args=args,
            log_level=LogLevel.INFO,
            output_dir=mock_file.parent / "output with spaces",
        )
        with pytest.raises(SuccessfulExit) as exc_info:
            profiler.run_profiler(raise_exit=True)

        assert exc_info.value.exit_code == 0
        logs = profiler.log_file.read_text()
        assert " ".join(args) in logs
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
            script_args=args,
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
        profiler = Profiler(config, log_level=LogLevel.INFO)

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
