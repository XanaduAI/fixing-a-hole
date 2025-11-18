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
"""Tests for the Fixing-A-Hole config and init."""

from pathlib import Path
from unittest.mock import patch

from fixingahole import config


class TestConfig:
    """Test Fixing-A-Hole config."""

    def test_detect_virtualenv(self):
        """Test detecting the virtual env."""
        assert config._detect_virtualenv()

    def test_find_pyproject(self):
        """Test finding the pyproject.toml."""
        pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
        assert config._find_pyproject() == pyproject_path

    def test_find_pyproject_outside_of_repo(self, tmp_path: Path):
        """Test finding the pyproject.toml."""
        pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
        with patch("fixingahole.config.Path.cwd") as cwd:
            cwd.side_effect = [tmp_path]
            assert config._find_pyproject() == pyproject_path

    def test_get_root_dir(self):
        """Test setting the root directory as the current working directory."""
        assert config._get_root_dir() == Path.cwd()
