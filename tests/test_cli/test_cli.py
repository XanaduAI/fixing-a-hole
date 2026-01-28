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

from colours import Colour
from typer.testing import CliRunner

from fixingahole import ROOT_DIR
from fixingahole.cli import main as cli
from tests.conftest import print_error

runner = CliRunner()


class TestProfilerSummarize:
    """Test the summarize CLI command."""

    def test_summarize_cli(self, example_json: Path):
        """Test summarize CLI on a valid JSON profile file."""
        result = runner.invoke(cli.app, ["summarize", str(example_json)])
        assert result.exit_code == 0, print_error(result)

    def test_summarize_cli_missing_file(self, tmp_path: Path):
        """Test summarize CLI with a missing file."""
        missing_path = "tests/scripts/data/does_not_exist.json"
        result = runner.invoke(cli.app, ["summarize", str(missing_path)])
        # Should exit with nonzero and print error
        assert result.exit_code == 1, "Should be error code 1."
        output = Colour.remove_ansi(result.stdout).replace("\n", "")
        print(output)
        assert f"No files in {tmp_path.name} with name: {missing_path} were found." == output

    def test_summarize_cli_options(self, example_json: Path):
        """Test summarize CLI with -n and -t options."""
        # Use -n 1 to show only top function
        result = runner.invoke(cli.app, ["summarize", str(example_json), "-n", "1"])
        assert result.exit_code == 0, print_error(result)
        output = Colour.remove_ansi(result.stdout)
        # Should only show one function in the top list
        assert "Top Function by Total Runtime:" in output

        # Use -t 100 to filter out all functions (threshold too high)
        result = runner.invoke(cli.app, ["summarize", str(example_json), "-t", "100"])
        assert result.exit_code == 0, print_error(result)
        output = Colour.remove_ansi(result.stdout)
        assert "No functions to summarize by Total Runtime" in output


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
        assert "Error: cannot profile a directory." in Colour.remove_ansi(result.stdout)

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
