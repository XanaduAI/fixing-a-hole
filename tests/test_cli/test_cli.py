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
"""Tests for the Fixing-A-Hole CLI tools."""

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from fixingahole import ROOT_DIR
from fixingahole.cli import main as cli
from tests.conftest import print_error

runner = CliRunner()


class TestProfilerRunProfiler:
    """Test the run_profiler method."""

    def test_profiler_cli_call(self, mock_file: Path):
        """Test how the CLI invokes the profiler."""
        result = runner.invoke(cli.app, ["profile", str(mock_file)])
        assert result.exit_code == 0, print_error(result)

    def test_profile_directory(self, mock_file: Path):
        """Test that the CLI fails to profile a directory."""
        result = runner.invoke(cli.app, ["profile", str(mock_file.parent)])
        assert result.exit_code == 1, print_error(result)
        assert "Error: cannot profile a directory." in result.stdout

    def test_profiler_cli_call_relative_path(self, mock_file: Path, root_dir: Path):
        """Test how the CLI invokes the profiler."""
        nested_dir = Path(root_dir / "nested" / "deeply")
        nested_dir.mkdir(parents=True, exist_ok=True)
        path = mock_file.rename(nested_dir / mock_file.name)
        result = runner.invoke(cli.app, ["profile", path.name])
        assert result.exit_code == 0, print_error(result)

    def test_version_call(self):
        """Test how the CLI invokes the --version flag."""
        cmd: list[str] = ["python", str(ROOT_DIR / "fixingahole" / "cli" / "main.py"), "--version"]
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout
