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

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer import Exit

from fixingahole import ROOT_DIR, LogLevel
from fixingahole.profiler import ProfileParser, Profiler
from fixingahole.profiler.utils import find_path
from tests.conftest import basic_name


class TestFindPath:
    """Test that find_path() works correctly."""

    def test_find_file(self, root_dir: Path):
        """Test that the correct file is found and successfully return."""
        filename = "find_me.py"
        path = Path(root_dir) / "deeply" / "nested" / filename
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert find_path(path.name) == path

    def test_many_files(self, root_dir: Path):
        """Test that the correct files are found and return with an error."""
        filename = "find_me.py"
        path = Path(root_dir) / "deeply" / "nested" / filename
        path.mkdir(parents=True, exist_ok=True)
        for p in path.parents[:3]:
            (p / filename).touch()
        with pytest.raises(Exit) as exc_info:
            find_path(filename)
        assert exc_info.value.exit_code == 1

    def test_no_files(self):
        """Test that no files are found and return with an error."""
        filename = "find_me.py"
        with pytest.raises(Exit) as exc_info:
            find_path(filename)
        assert exc_info.value.exit_code == 1

    def test_find_dir(self, root_dir: Path):
        """Test that the correct dir is found and successfully return."""
        dirname = "find_me"
        path = Path(root_dir) / "deeply" / "nested" / dirname
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert find_path(path.name) == path

    def test_many_dirs(self, root_dir: Path):
        """Test that the correct dirs are found and return with an error."""
        dirname = "find_me"
        path = Path(root_dir) / "deeply" / "nested" / dirname
        path.mkdir(parents=True, exist_ok=True)
        for p in path.parents[:3]:
            (p / dirname).touch()
        with pytest.raises(Exit) as exc_info:
            find_path(dirname)
        assert exc_info.value.exit_code == 1

    def test_no_dirs(self):
        """Test that no dirs are found and return with an error."""
        dirname = "find_me"
        with pytest.raises(Exit) as exc_info:
            find_path(dirname)
        assert exc_info.value.exit_code == 1

    def test_find_files_in_dir(self, root_dir: Path):
        """Test that the correct files are found inside the directory."""
        filenames: list[str] = ["find_me.py", "find_me_too.py"]
        path = Path(root_dir) / "deeply" / "nested"
        path.mkdir(parents=True, exist_ok=True)
        for i, filename in enumerate(filenames):
            filenames[i] = (path / filename).resolve()
            filenames[i].touch()
        output_path, output_files = find_path(path.name, return_suffix=".py")
        assert output_path == path
        assert sorted(output_files) == sorted(filenames)

    def test_absolute_path(self, root_dir: Path):
        """Test that the correct dir is found."""
        filename = "find_me.py"
        path = Path(root_dir) / "deeply" / "nested" / filename
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        path.resolve()
        assert find_path(path) == path

    def test_additional_excluded_paths(self, root_dir: Path):
        """Test that the correct dir is found."""
        (Path(root_dir) / "not" / "looking" / "_here").mkdir(parents=True, exist_ok=True)
        (Path(root_dir) / ".hidden").mkdir(parents=True, exist_ok=True)
        path = Path(root_dir) / "deeply" / "nested" / "find_me.py"
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert find_path(path.name, exclude=["not/**/_*", ".hidden"]) == path


class TestProfilerInit:
    """Test the Profiler initialization and basic properties."""

    def test_init(self, mock_file: Path):
        """Import the profiler."""
        import fixingahole.profiler as p  # noqa: PLC0415

        p.Profiler(path=mock_file)

    def test_init_with_file_path(self, mock_file: Path):
        """Test profiler initialization with a file path."""
        precision_limit = 10
        profiler = Profiler(path=mock_file)

        assert profiler.cpu_only is False
        assert profiler.script_args == []
        assert profiler.precision == 0
        assert profiler.precision_limit == precision_limit
        assert profiler.detailed is False
        assert profiler.loglevel == LogLevel.CRITICAL
        assert profiler.noplots is False
        assert profiler.filestem == basic_name()
        assert profiler.python_file == mock_file
        assert not profiler.cli_inputs

    def test_init_with_bad_path(self, tmp_path: Path):
        """Test profiler initialization with path that doesn't exist."""
        for stem in ["file_does_not_exist.py", "dir_does_not_exist"]:
            with pytest.raises(Exit) as exc_info:
                Profiler(path=tmp_path / stem)
            err_code = 127
            assert exc_info.value.exit_code == err_code

    def test_init_with_all_options(self, mock_file: Path):
        """Test profiler initialization with all options set."""
        precition_value = 3
        precision_limit = 10
        profiler = Profiler(
            path=mock_file,
            python_script_args=["arg1", "arg2"],
            cpu_only=True,
            precision=precition_value,
            detailed=True,
            loglevel=LogLevel.DEBUG,
            noplots=True,
            profilename="custom.txt",
            trace=False,
        )

        assert profiler.cpu_only is True
        assert profiler.script_args == ["arg1", "arg2"]
        assert profiler.precision == precition_value
        assert profiler.precision_limit == precision_limit
        assert profiler.detailed is True
        assert profiler.loglevel == LogLevel.DEBUG
        assert profiler.noplots is True
        assert profiler.trace is False

    def test_init_handles_path_with_spaces(self, tmp_path: Path):
        """Test profiler initialization handles file names with spaces."""
        test_file = tmp_path / "test script with spaces.py"
        test_file.write_text("print('hello world')")
        profiler = Profiler(path=test_file)

        assert profiler.filestem == "test_script_with_spaces"

    def test_init_precision_conversion(self, mock_file: Path):
        """Test that precision parameter is properly converted to int."""
        value = 5
        profiler = Profiler(path=mock_file, precision=str(value))
        assert profiler.precision == value
        assert isinstance(profiler.precision, int)

    def test_init_precision_none_handling(self, mock_file: Path):
        """Test that None precision is handled correctly."""
        profiler = Profiler(path=mock_file, precision=None)
        assert profiler.precision == 0


class TestProfilerProperties:
    """Test the Profiler properties."""

    def test_excluded_folders_property(self, mock_file: Path):
        """Test the excluded_folders property."""
        profiler = Profiler(path=mock_file, detailed=True)
        excluded = profiler.excluded_folders
        # Should contain the system python directory exclude flag
        if sys.executable:
            exclude_dir = Path(sys.executable).resolve().parents[1]
            if not exclude_dir.is_relative_to(ROOT_DIR):
                assert "--profile-exclude" in excluded
                assert str(exclude_dir) in excluded
            else:
                assert not excluded

    def test_output_file_property_getter(self, mock_file: Path):
        """Test the output_file property getter."""
        profiler = Profiler(path=mock_file)
        assert profiler.output_file is not None
        assert isinstance(profiler.output_file, Path)

    def test_output_file_property_setter(self, mock_file: Path):
        """Test the output_file property setter."""
        profiler = Profiler(path=mock_file)

        new_output = mock_file.parent / "new_output.txt"
        profiler.output_file = new_output

        assert profiler.output_file == new_output
        # Should be created by touch()
        assert new_output.exists()

        # Also test that it works with a string path.
        profiler = Profiler(path=mock_file)

        new_output_str = str(mock_file.parent / "new_output.txt")
        profiler.output_file = new_output_str

        assert profiler.output_file == Path(new_output_str)
        # Should be created by touch()
        assert Path(new_output_str).exists()

    def test_output_path_property_within_root(self, mock_file: Path):
        """Test output_path property when output_file is within ROOT_DIR."""
        profiler = Profiler(path=mock_file)
        # output_path should be relative to ROOT_DIR
        output_path = profiler.output_path
        assert not output_path.is_absolute()

    def test_output_path_property_outside_root(self, mock_file: Path, root_dir: Path):
        """Test output_path property when output_file is outside ROOT_DIR."""
        external_dir = Path(root_dir).parent / "external"
        external_dir.mkdir(exist_ok=True, parents=True)

        profiler = Profiler(path=mock_file)
        profiler.output_file = external_dir / "output.txt"

        # output_path should return absolute path when outside ROOT_DIR
        output_path = profiler.output_path
        assert output_path == profiler.output_file

    def test_log_file_property(self, mock_file: Path):
        """Test the log_file property."""
        profiler = Profiler(path=mock_file)

        log_file = profiler.log_file
        assert log_file.name == "logs.log"
        assert profiler.profile_root in log_file.parents

    def test_log_path_property(self, mock_file: Path):
        """Test the log_path property."""
        profiler = Profiler(path=mock_file)

        log_path = profiler.log_path
        # Should be relative to ROOT_DIR
        assert not log_path.is_absolute()


class TestProfilerIPythonNotebookConversion:
    """Test the convert_ipynb_to_py method."""

    def test_convert_ipynb_to_py_basic(self):
        """Test basic Jupyter notebook to Python conversion."""
        notebook_content = {
            "cells": [
                {"cell_type": "code", "source": ["print('hello')", "x = 1", "y = 2"]},
                {"cell_type": "markdown", "source": ["# This is markdown"]},
                {"cell_type": "code", "source": ["print('world')", "z = x + y"]},
            ]
        }
        result = Profiler.convert_ipynb_to_py(json.dumps(notebook_content))

        # Should only include code cells, not markdown
        assert "print('hello')" in result
        assert "x = 1" in result
        assert "y = 2" in result
        assert "print('world')" in result
        assert "z = x + y" in result
        assert "# This is markdown" not in result

    def test_convert_ipynb_to_py_empty_notebook(self):
        """Test conversion of empty notebook and return with an error."""
        notebook_content = {"cells": []}
        with pytest.raises(Exit) as exc_info:
            Profiler.convert_ipynb_to_py(json.dumps(notebook_content))
        assert exc_info.value.exit_code == 1

    def test_convert_ipynb_to_py_only_markdown(self):
        """Test conversion of notebook with only markdown cells and return with an error."""
        notebook_content = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title"]},
                {"cell_type": "markdown", "source": ["Some text"]},
            ]
        }
        with pytest.raises(Exit) as exc_info:
            Profiler.convert_ipynb_to_py(json.dumps(notebook_content))
        assert exc_info.value.exit_code == 1


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


class TestProfilerMemoryPrecision:
    """Test the get_memory_precision method."""

    DEFAULT_THRESHOLD = 10485767

    def test_get_memory_precision_default(self, mock_file: Path):
        """Test memory precision with default value."""
        profiler = Profiler(path=mock_file, precision=0)
        result = profiler.get_memory_precision()
        assert result.startswith("--allocation-sampling-window=")
        # Should contain a prime number around 10MB
        threshold = int(result.split("=")[1])
        low, high = 10000000, 11000000
        assert low <= threshold <= high  # Around 10MB range

    def test_get_memory_precision_high_verbosity(self, mock_file: Path):
        """Test memory precision with high verbosity (small threshold)."""
        profiler = Profiler(path=mock_file, precision=3)
        result = profiler.get_memory_precision()
        assert result.startswith("--allocation-sampling-window=")
        # Should be smaller than default (10MB / 2^3 = ~1.25MB)
        threshold = int(result.split("=")[1])
        low = 1e6
        assert low < threshold < self.DEFAULT_THRESHOLD

    def test_get_memory_precision_low_verbosity(self, mock_file: Path):
        """Test memory precision with low verbosity (large threshold)."""
        profiler = Profiler(path=mock_file, precision=-3)

        result = profiler.get_memory_precision()

        assert result.startswith("--allocation-sampling-window=")
        # Should be larger than default (10MB * 2^3 = ~80MB)
        threshold = int(result.split("=")[1])
        high = 9e7
        assert high > threshold > self.DEFAULT_THRESHOLD

    def test_get_memory_precision_clamp_values(self, mock_file: Path):
        """Test that precision values are clamped to a range."""
        with patch("fixingahole.profiler.utils.Colour.print") as mock_color_print:
            for test_val in [-25, 25]:
                mock_color_print.reset_mock()
                profiler = Profiler(path=mock_file, precision=test_val)
                profiler.get_memory_precision()
                limit = profiler.precision_limit
                warning = f"Warning: -{limit} <= precision <= {limit}"
                mock_color_print.assert_called_with(warning)

    def test_adjusted_memory_precision_clamp_values(self, mock_file: Path):
        """Test that precision values are clamped to a range."""
        for test_val in [-25, 25]:
            profiler = Profiler(path=mock_file, precision=test_val)
            profiler.precision_limit = test_val * 2
            result = profiler.get_memory_precision()
            threshold = int(result.split("=")[1])
            default = 10485767
            assert (default / 2 ** abs(test_val)) < threshold < (default * 2 ** (abs(test_val) + 1))

    def test_get_memory_precision_valid_range_no_warning(self, mock_file: Path):
        """Test that valid precision values don't trigger warnings."""
        with patch("fixingahole.profiler.utils.Colour.print") as mock_color_print:
            profiler = Profiler(path=mock_file, precision=3)
            profiler.get_memory_precision()
            mock_color_print.assert_not_called()


class TestProfilerCodePreparation:
    """Test the prepare_code_for_profiling method."""

    def test_prepare_code_for_profiling_basic(self, tmp_path: Path):
        """Test basic code preparation for profiling."""
        test_file = tmp_path / "test_script.py"
        test_code = "print('hello world')\nx = 1\ny = 2"
        test_file.write_text(test_code)

        profiler = Profiler(path=test_file)
        profiler.prepare_code_for_profiling()

        # Check that the profile file was created and contains expected content
        profile_content = profiler.profile_file.read_text()

        # Should contain logging setup
        assert "import logging" in profile_content
        assert f"log_file = Path(r'{profiler.log_file}')" in profile_content
        assert f"logging.basicConfig(filename=log_file, level=logging.{profiler.loglevel.name})" in profile_content

        # Should contain scalene profiler setup
        assert "from scalene import scalene_profiler" in profile_content
        assert "scalene_profiler.start()" in profile_content
        assert "scalene_profiler.stop()" in profile_content

        # Should contain original code
        assert "print('hello world')" in profile_content
        assert "x = 1" in profile_content
        assert "y = 2" in profile_content

        # Should contain dividers
        assert "####################################" in profile_content

    def test_prepare_code_for_profiling_with_noplots(self, tmp_path: Path):
        """Test code preparation with noplots option enabled."""
        test_file = tmp_path / "test_script.py"
        test_code = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()"
        test_file.write_text(test_code)

        profiler = Profiler(path=test_file, noplots=True)
        profiler.prepare_code_for_profiling()
        profile_content = profiler.profile_file.read_text()

        # Should contain plot mocking
        assert "from unittest.mock import patch, MagicMock" in profile_content
        assert "patch_plt = patch('matplotlib.pyplot.show', new=MagicMock())" in profile_content
        assert "patch_plotly = patch('plotly.graph_objects.Figure.show', new=MagicMock())" in profile_content
        assert "patch_plt.start()" in profile_content
        assert "patch_plotly.start()" in profile_content
        assert "patch_plt.stop()" in profile_content
        assert "patch_plotly.stop()" in profile_content

    def test_prepare_code_for_profiling_with_jupyter_notebook(self, tmp_path: Path):
        """Test code preparation with Jupyter notebook file."""
        test_file = tmp_path / "test_notebook.ipynb"
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello from notebook')", "import numpy as np"],
                }
            ]
        }
        test_file.write_text(json.dumps(notebook_content))

        profiler = Profiler(path=test_file)
        profiler.prepare_code_for_profiling()
        profile_content = profiler.profile_file.read_text()

        # Should contain converted notebook content
        assert "print('hello from notebook')" in profile_content
        assert "import numpy as np" in profile_content

    def test_prepare_code_for_profiling_with_warning_loglevel(self, mock_file: Path):
        """Test code preparation with warning log level."""
        profiler = Profiler(path=mock_file, loglevel=LogLevel.WARNING)

        profiler.prepare_code_for_profiling()
        profile_content = profiler.profile_file.read_text()

        # Should contain warning capture
        assert "logging.captureWarnings(True)" in profile_content

    def test_prepare_code_for_profiling_with_error_loglevel(self, tmp_path: Path):
        """Test code preparation with error log level (no warnings)."""
        test_file = tmp_path / "test_script.py"
        test_file.write_text("print('hello world')")

        profiler = Profiler(path=test_file, loglevel=LogLevel.ERROR)
        profiler.prepare_code_for_profiling()
        profile_content = profiler.profile_file.read_text()

        # Should NOT contain warning capture
        assert "logging.captureWarnings(True)" not in profile_content


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
