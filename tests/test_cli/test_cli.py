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
from unittest.mock import Mock, patch

import pytest
from colours import Colour
from typer.testing import CliRunner

from fixingahole.cli import main as cli
from tests.conftest import print_error

runner = CliRunner()


class TestProfilerSummarize:
    """Test the summarize CLI command."""

    def test_summarize_cli(self, example_json: Path):
        """Test summarize CLI on a valid JSON profile file."""
        result = runner.invoke(cli.app, ["summarize", str(example_json)])
        assert result.exit_code == 0, print_error(result)

    @patch("fixingahole.profiler.utils.Colour.error")
    def test_summarize_cli_missing_file(self, mock_colour_error: Mock, tmp_path: Path):
        """Test summarize CLI with a missing file."""
        missing_path = "tests/scripts/data/does_not_exist.json"
        result = runner.invoke(cli.app, ["summarize", str(missing_path)])
        # Should exit with nonzero and print error
        assert result.exit_code == 1, "Should be error code 1."
        expected_output: list[str] = [
            "No %s in %s with name: %s were found.",
            "files",
            f"[magenta]{tmp_path.name}[/magenta]",
            f"[green]{missing_path}[/green]",
        ]
        mock_colour_error.assert_called_once_with(*expected_output)

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


class TestStats:
    """Test the stats CLI command."""

    def test_stats_cli(self, example_json: Path):
        """Test stats CLI on a valid JSON profile file."""
        tmp_dir = example_json.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        example_json = example_json.rename(tmp_dir / example_json.name)
        (tmp_dir / "dup_data.json").write_bytes(example_json.read_bytes())
        result = runner.invoke(cli.app, ["stats", str(tmp_dir), "--no-metadata"])
        assert result.exit_code == 0, print_error(result)

    @patch("fixingahole.profiler.utils.Colour.error")
    def test_stats_cli_missing_file(self, mock_colour_error: Mock, tmp_path: Path):
        """Test stats CLI with a missing file."""
        missing_dir = "tests/scripts/data/does_not_exist"
        result = runner.invoke(cli.app, ["stats", str(missing_dir)])
        # Should exit with nonzero and print error
        assert result.exit_code == 1, "Should be error code 1."
        expected_output: list[str] = [
            "No %s in %s with name: %s were found.",
            "folders",
            f"[magenta]{tmp_path.name}[/magenta]",
            f"[green]{missing_dir}[/green]",
        ]
        mock_colour_error.assert_called_once_with(*expected_output)


class TestProfilerRunProfiler:
    """Test the run_profiler method."""

    def test_profiler_cli_call(self, mock_file: Path):
        """Test how the CLI invokes the profiler."""
        result = runner.invoke(cli.app, ["profile", str(mock_file)])
        assert result.exit_code == 0, print_error(result)

    @pytest.mark.parametrize("n_runs", [2, 3])
    def test_profiler_repeat(self, n_runs: int, mock_file: Path, root_dir: Path):
        """Test how the profiler handles repeated profilings."""
        result = runner.invoke(cli.app, ["profile", str(mock_file), "--repeat", str(n_runs)])
        assert result.exit_code == 0, print_error(result)
        output_files: list[Path] = sorted(file for file in (root_dir / "performance").rglob("*") if file.is_file())
        assert len(output_files) == (3 + 3 * n_runs)
        assert len([f for f in output_files if f.suffix == ".py"]) == 1
        assert len(logfile := [f for f in output_files if f.suffix == ".log"]) == 1  # one shared log file.
        assert len(logfile.pop().read_text().splitlines()) == n_runs  # one warning log per run.
        assert len([f for f in output_files if f.suffix == ".json"]) == n_runs + 1  # one JSON per run, and one stats file.
        assert len([f for f in output_files if f.suffix == ".txt"]) == n_runs * 2  # one results and one summary per run.

    def test_profile_directory(self, mock_file: Path):
        """Test that the CLI fails to profile a directory."""
        tmp_dir = mock_file.parent / "tmp" / "nested" / "dir"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli.app, ["profile", str(tmp_dir.relative_to(mock_file.parent))])
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
        cmd: list[str] = ["python", str(Path(__file__).parents[2] / "fixingahole" / "cli" / "main.py"), "--version"]
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout

    def test_profiler_cli_call_bad_flags_noplots_inplace(self, mock_file: Path):
        """Test that the CLI invocation fails with bad flag combinations."""
        result = runner.invoke(cli.app, ["profile", str(mock_file), "--in-place", "--no-plots"])
        assert result.exit_code == 1, print_error(result)

    def test_profiler_cli_call_bad_flags_filename_inplace(self, mock_file: Path):
        """Test that the CLI invocation fails with bad flag combinations."""
        result = runner.invoke(cli.app, ["profile", str(mock_file.with_suffix(".ipynb")), "--in-place"])
        assert result.exit_code == 1, print_error(result)
