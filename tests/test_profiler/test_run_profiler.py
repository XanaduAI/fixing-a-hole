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

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer import Exit

from fixingahole import LogLevel
from fixingahole.profiler import Profiler


class TestProfilerRunProfiler:
    """Test the run_profiler method."""

    def _mocked_file(self, mock_file: Path) -> Path:
        content = [
            "import numpy as np",
            "from sys import argv",
            "def main():",
            "  try:",
            "    logging.info(' '.join(argv[1:]))",
            "    logging.warning('This is a warning.')",
            "  except: pass",
            "  a = np.random.uniform(size=10**7)",
            "main()",
        ]
        mock_file.write_text("\n".join(content))
        return mock_file

    def test_run_profiler_success(self, mock_file: Path):
        """Test successful profiler run."""
        profiler = Profiler(path=self._mocked_file(mock_file), precision=5, loglevel=LogLevel.INFO)
        with pytest.raises(Exit) as exc_info:
            profiler.run_profiler()
        assert exc_info.value.exit_code == 0

    def test_run_profiler_cpu_only(self, mock_file: Path):
        """Test successful profiler run using only CPU."""
        profiler = Profiler(path=self._mocked_file(mock_file), cpu_only=True)
        with pytest.raises(Exit) as exc_info:
            profiler.run_profiler()
        assert exc_info.value.exit_code == 0

    def test_run_profiler_detailed_mode(self, mock_file: Path):
        """Test profiler run with detailed profiling enabled."""
        profiler = Profiler(path=self._mocked_file(mock_file), precision=5, detailed=True)
        with pytest.raises(Exit) as exc_info:
            profiler.run_profiler()

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

        profiler = Profiler(path=self._mocked_file(mock_file), precision=5)

        with pytest.raises(Exit) as exc_info:
            profiler.run_profiler()

        assert exc_info.value.exit_code == 1
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_profiler_keyboard_interrupt(self, mock_run: MagicMock, mock_file: Path):
        """Test profiler run handling keyboard interrupt."""
        # Setup subprocess to raise KeyboardInterrupt.
        mock_run.side_effect = KeyboardInterrupt()

        profiler = Profiler(path=self._mocked_file(mock_file), precision=5)

        with pytest.raises(Exit) as exc_info:
            profiler.run_profiler()

        assert exc_info.value.exit_code == 1
        mock_run.assert_called_once()
        assert "Profiling interrupted by user." in profiler.output_file.read_text()

    def test_run_profiler_with_script_args_and_logs(self, mock_file: Path):
        """Test profiler run with script arguments."""
        args = ["arg1=value1", "arg2=value2"]
        profiler = Profiler(
            path=self._mocked_file(mock_file),
            python_script_args=args,
            precision=5,
            loglevel=LogLevel.INFO,
        )
        with pytest.raises(Exit) as exc_info:
            profiler.run_profiler()

        assert exc_info.value.exit_code == 0
        logs = profiler.log_file.read_text()
        for arg in args:
            assert arg in logs
        assert "This is a warning." in logs
        # Check that warning count is correctly calculated and included
        final_content = profiler.output_file.read_text()
        assert "Check logs" in final_content
        assert "(1 warning)" in final_content
