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

import pytest

from fixingahole import config
from fixingahole.config import Duration, DurationOption


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
        assert config._get_root_dir(config={}) == Path.cwd()


class TestDurationOption:
    """Basic tests for the DurationOption in the config."""

    def test_enum_values(self):
        """Basic test that the DurationOption Enum works."""
        assert DurationOption.absolute.value == "absolute"
        assert DurationOption.relative.value == "relative"
        bad_value = "bad_value"
        with pytest.raises(ValueError, match=f"'{bad_value}' is not a valid DurationOption"):
            DurationOption(bad_value)


class TestDuration:
    """Tests for the Duration singleton class.

    ****These tests need to run in this order or they will break subsequent tests.****
    """

    def test_invalid_value_exits(self):
        """Test that the program halts when the Duration is not a DurationOption."""
        config.Duration._instance = None
        with pytest.raises(SystemExit):
            Duration("invalid")

    def test_singleton_absolute(self):
        """Test test the Duration object is a singleton, when absolute."""
        # Reset singleton for test isolation
        config.Duration._instance = None
        d1 = Duration("absolute")
        d2 = Duration("absolute")
        assert d1 is d2
        assert Duration.is_absolute(), Duration.is_absolute()
        assert not Duration.is_relative(), Duration.is_relative()

    def test_singleton_relative(self):
        """Test test the Duration object is a singleton, when relative."""
        # Reset singleton for test isolation
        config.Duration._instance = None
        d1 = Duration("relative")
        d2 = Duration("relative")
        assert d1 is d2
        assert Duration.is_relative()
        assert not Duration.is_absolute()
