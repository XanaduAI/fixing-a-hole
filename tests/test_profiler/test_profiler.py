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
from unittest.mock import patch

import pytest

from fixingahole import LogLevel
from fixingahole.profiler.profiler import Profiler, ProfilerException, ResourceUsage
from fixingahole.profiler.utils import FindPathException, find_path


class TestResourceUsage:
    """Test ResourceUsage context manager behaviour."""

    def test_report_raises_before_exit(self):
        """Accessing .report before __exit__ raises RuntimeError."""
        with ResourceUsage() as usage, pytest.raises(RuntimeError, match="before context exit"):
            _ = usage.report

    def test_report_contains_walltime(self):
        """Report includes wall time when walltime is non-zero."""
        with ResourceUsage() as usage:
            pass
        assert "Wall Time" in usage.report


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
        with pytest.raises(FindPathException) as exc_info:
            find_path(filename)
        assert exc_info.value.exit_code == 1

    def test_no_files(self):
        """Test that no files are found and return with an error."""
        filename = "find_me.py"
        with pytest.raises(FindPathException) as exc_info:
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
        with pytest.raises(FindPathException) as exc_info:
            find_path(dirname)
        assert exc_info.value.exit_code == 1

    def test_no_dirs(self):
        """Test that no dirs are found and return with an error."""
        dirname = "find_me"
        with pytest.raises(FindPathException) as exc_info:
            find_path(dirname)
        assert exc_info.value.exit_code == 1

    def test_find_files_in_dir(self, root_dir: Path):
        """Test that the correct files are found inside the directory."""
        path: Path = Path(root_dir) / "deeply" / "nested"
        filenames: list[Path] = [Path(path / file).resolve() for file in ["find_me.py", "find_me_too.py"]]
        path.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            filename.touch()
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

    def test_init_with_bad_path(self, tmp_path: Path):
        """Test profiler initialization with path that doesn't exist."""
        for stem in ["file_does_not_exist.py", "dir_does_not_exist"]:
            with pytest.raises(ProfilerException) as exc_info:
                Profiler(tmp_path / stem)
            err_code = 127
            assert exc_info.value.exit_code == err_code

    def test_init_with_dir(self, mock_dir: Path):
        """Test profiler initialization with path that is a directory."""
        with pytest.raises(ProfilerException) as exc_info:
            Profiler(mock_dir)
        err_code = 1
        assert exc_info.value.exit_code == err_code

    def test_init_custom_output_dir(self, mock_file: Path, root_dir: Path, non_local_dir: Path):
        """Test profiler initialization with in_place=True and custom output_dir."""
        custom_output = non_local_dir / "custom_output"
        custom_output.mkdir(parents=True, exist_ok=False)
        profiler = Profiler(mock_file, output_dir=custom_output)

        # Verify python_file is within root_dir.
        assert profiler.python_file.is_relative_to(root_dir)
        assert not profiler.python_file.is_relative_to(non_local_dir)

        # Verify output_file is in the custom directory and not within root_dir
        assert profiler.output_file == custom_output / "profile_results.txt"
        assert profiler.output_file.exists()
        assert not profiler.output_file.is_relative_to(root_dir)

        # Verify profile_root is the custom directory and not within root_dir
        assert profiler.profile_root == custom_output
        assert not profiler.profile_root.is_relative_to(root_dir)


class TestProfilerProperties:
    """Test the Profiler properties."""

    def test_output_file_property_setter(self, mock_file: Path):
        """Test the output_file property setter."""
        profiler = Profiler(mock_file)

        new_output = mock_file.parent / "new_output.txt"
        profiler.output_file = new_output

        assert profiler.output_file == new_output
        # Should be created by touch()
        assert new_output.exists()

        # Also test that it works with a string path.
        profiler = Profiler(mock_file)

        new_output_str = str(mock_file.parent / "new_output.txt")
        profiler.output_file = new_output_str

        assert profiler.output_file == Path(new_output_str)
        # Should be created by touch()
        assert Path(new_output_str).exists()

    def test_output_path_property_within_root(self, mock_file: Path, root_dir: Path):
        """Test output_path property when output_file is within the current directory."""
        profiler = Profiler(mock_file)
        # output_path should be relative to the current directory (root)
        with patch("pathlib.Path.cwd", return_value=root_dir):
            output_path = profiler.output_path
            assert not output_path.is_absolute()

    def test_properties_outside_root(self, mock_file: Path, root_dir: Path, non_local_dir: Path):
        """Test profile properties when output_file is outside the current directory."""
        external_dir = non_local_dir / "external"
        external_dir.mkdir(exist_ok=True, parents=True)

        profiler = Profiler(mock_file, log_level=LogLevel.INFO)
        profiler.output_file = external_dir / "output.txt"

        # output_path should return absolute path when outside the current directory (root)
        with patch("pathlib.Path.cwd", return_value=root_dir):
            output_path = profiler.output_path
            assert output_path == profiler.output_file
            assert not profiler.path_to_summary.is_relative_to(root_dir)
            assert not profiler.profile_path.is_relative_to(root_dir)
            assert not profiler.log_path.is_relative_to(root_dir)

    def test_output_summary_names(self, mock_file: Path):
        """Test the output_summary names."""
        profiler = Profiler(mock_file)

        new_output = mock_file.parent / "new_output.txt"
        assert profiler.output_summary.name == profiler.output_file.name.replace("results", "summary")
        profiler.output_file = new_output
        assert profiler.output_summary.name == "profile_summary.txt"

    def test_log_file_property(self, mock_file: Path):
        """Test the log_file property."""
        profiler = Profiler(mock_file)

        log_file = profiler.log_file
        assert log_file.name == "profile_logs.log"
        assert profiler.profile_root in log_file.parents

    def test_log_path_property(self, mock_file: Path, root_dir: Path):
        """Test the log_path property."""
        profiler = Profiler(mock_file)

        # Should be relative to ROOT_DIR
        with patch("pathlib.Path.cwd", return_value=root_dir):
            log_path = profiler.log_path
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
        with pytest.raises(ProfilerException) as exc_info:
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
        with pytest.raises(ProfilerException) as exc_info:
            Profiler.convert_ipynb_to_py(json.dumps(notebook_content))
        assert exc_info.value.exit_code == 1


class TestProfilerConfig:
    """Test the ProfilerConfig protocol and custom configuration."""

    def test_profiler_config_missing_required_attributes(self):
        """Test that ProfilerConfig validation catches missing required attributes."""

        class EmptyConfig:
            """Config that sets nothing."""

            def setup(self, profiler: Profiler) -> None:
                """Do nothing."""

        config = EmptyConfig()
        with pytest.raises(ProfilerException) as exc_info:
            Profiler(config)

        # Should mention multiple missing attributes
        error_msg = exc_info.value.message
        assert "ProfilerConfig.setup() must set `python_file` property." in error_msg
