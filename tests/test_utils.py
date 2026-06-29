# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Fixing-A-Hole utilities."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fixingahole.profiler.utils import (
    FileWatcher,
    FindPathException,
    LogLevel,
    Spinner,
    date,
    find_path,
    format_time,
    installed_modules,
    memory_with_units,
    precision_to_allocation_window,
)


def test_date():
    """Test that date() returns a string."""
    assert isinstance(date(), str)


class TestLogLevel:
    """Test the LogLevel enum."""

    @pytest.mark.parametrize(
        ("level", "should_catch"),
        [
            (LogLevel.NONE, False),
            (LogLevel.NOTSET, True),
            (LogLevel.DEBUG, True),
            (LogLevel.INFO, True),
            (LogLevel.WARNING, True),
            (LogLevel.ERROR, True),
            (LogLevel.CRITICAL, True),
        ],
    )
    def test_capture_output(self, level: LogLevel, should_catch: bool):
        """Test capture_output returns correct value for each level."""
        assert level.capture_output() == should_catch

    def test_should_catch_warnings_emits_deprecation(self):
        """should_catch_warnings() emits DeprecationWarning and delegates to capture_output()."""
        with pytest.warns(DeprecationWarning, match="should_catch_warnings.*deprecated"):
            result = LogLevel.WARNING.should_catch_warnings()
        assert result == LogLevel.WARNING.capture_output()


class TestFormatTime:
    """Test the format_time function."""

    def test_format_seconds_only(self):
        """Test formatting with seconds only."""
        assert format_time(5.123) == "5.123 sec"
        assert format_time(45.678) == "45.678 sec"

    def test_format_minutes_and_seconds(self):
        """Test formatting with minutes and seconds."""
        assert format_time(125) == "02:05"  # 2 minutes, 5 seconds
        assert format_time(3599) == "59:59"  # 59 minutes, 59 seconds

    def test_format_hours_minutes_seconds(self):
        """Test formatting with hours, minutes, and seconds."""
        assert format_time(3661) == " 1:01:01"  # 1 hour, 1 minute, 1 second
        assert format_time(7385) == " 2:03:05"  # 2 hours, 3 minutes, 5 seconds

    def test_format_with_max_val_hours(self):
        """Test formatting when max_val has hours."""
        # Even if duration is small, if max_val has hours, format with hours
        assert format_time(125, max_val=3600) == " 0:02:05"

    def test_format_with_max_val_minutes(self):
        """Test formatting when max_val has minutes."""
        # If max_val has minutes, format with minutes
        assert format_time(45, max_val=200) == "00:45"


class TestPrecisionToAllocationWindow:
    """Test the precision_to_allocation_window utility function."""

    DEFAULT_THRESHOLD = 10485767

    def test_default_precision(self):
        """Default precision (0) yields a prime near 10 MB."""
        result = precision_to_allocation_window(0)
        assert 10_000_000 <= result <= 11_000_000

    def test_positive_precision_smaller_window(self):
        """A positive precision produces a window smaller than the default."""
        result = precision_to_allocation_window(3)
        assert result < self.DEFAULT_THRESHOLD
        assert result > 1_000_000  # still reasonable

    def test_negative_precision_larger_window(self):
        """A negative precision produces a window larger than the default."""
        result = precision_to_allocation_window(-3)
        assert result > self.DEFAULT_THRESHOLD
        assert result < 9e7

    def test_no_warning_within_range(self):
        """Values within range do not emit a warning."""
        with patch("fixingahole.profiler.utils.Colour.warning") as mock_warn:
            precision_to_allocation_window(5)
            mock_warn.assert_not_called()

    def test_returns_prime(self):
        """The returned value is a prime number."""
        from sympy import isprime  # noqa: PLC0415

        for p in range(-3, 4):
            assert isprime(precision_to_allocation_window(p))


class TestMemoryWithUnits:
    """Test the memory_with_units function."""

    def test_memory_mb_to_gb(self):
        """Test converting MB to GB."""
        result = memory_with_units(2048, unit="MB", digits=2)
        assert "2.00 GB" in result

    def test_memory_mb_stays_mb(self):
        """Test memory stays in MB."""
        result = memory_with_units(500, unit="MB", digits=1)
        assert "500.0 MB" in result

    def test_memory_bytes(self):
        """Test small memory values in bytes."""
        result = memory_with_units(0.0001, unit="MB", digits=2)
        assert "bytes" in result


class TestFindPath:
    """Test the find_path function."""

    def test_find_path_with_subfolder_only_no_files(self, root_dir: Path):
        """Test find_path raises Exit when subfolder_only=True and no files found."""
        # Create a directory with no matching files
        test_dir = root_dir / "test_folder"
        test_dir.mkdir()
        (test_dir / "wrong_file.txt").write_text("content")

        with pytest.raises(FindPathException) as exc_info:
            find_path("test_folder", in_dir="", return_suffix=".py", subfolder_only=True)
        assert exc_info.value.exit_code == 1


class TestInstalledModules:
    """Test the installed_modules function."""

    def test_installed_modules_returns_set(self):
        """Test that installed_modules returns a set of module names."""
        modules = installed_modules()
        assert isinstance(modules, set)
        assert len(modules) > 0
        # Some common modules that should exist
        assert any(mod in modules for mod in ["pip", "pytest", "setuptools"])

    def test_installed_modules_with_missing_name_metadata(self):
        """Test installed_modules handles distributions missing Name metadata."""
        # Create a mock distribution without Name in metadata but with files not containing egg-info
        mock_dist = Mock()
        mock_dist.metadata = {}
        mock_dist.files = [Path("some/file.py")]

        with patch("fixingahole.profiler.utils.importlib.metadata.distributions") as mock_distributions:
            mock_distributions.return_value = [mock_dist]
            modules = installed_modules()
            # Should return empty set since no valid modules
            assert modules == set()


class TestSpinner:
    """Test the Spinner class."""

    def test_spinner_normal_creation(self, monkeypatch: pytest.MonkeyPatch):
        """Test that Spinner creates normally when COLOURS_DISABLE_PRINT is not set."""
        monkeypatch.delenv("COLOURS_DISABLE_PRINT", raising=False)
        with Spinner("Testing", style="blue", speed=0.5) as spinner:
            assert spinner is not None


class TestFileWatcher:
    """Test the FileWatcher class."""

    def test_file_watcher_start_and_stop(self, tmp_path: Path):
        """Test starting and stopping the file watcher."""
        test_file = tmp_path / "watch_me.txt"
        test_file.write_text("initial content")

        callback_called: list[bool] = []

        def callback() -> None:
            callback_called.append(True)

        watcher = FileWatcher(test_file, callback)
        watcher.start()

        # Give the observer time to start
        time.sleep(0.1)

        # Modify the file
        test_file.write_text("modified content")

        # Wait for the callback to be triggered
        time.sleep(0.5)

        watcher.stop()

        # Callback should have been called
        assert len(callback_called) > 0

    def test_file_watcher_ignores_directory_events(self, tmp_path: Path):
        """Test that file watcher ignores directory modification events."""
        test_file = tmp_path / "watch_dir" / "file.txt"
        test_file.parent.mkdir()
        test_file.write_text("content")
        # Wait for file contents to be written.
        time.sleep(0.1)

        callback_called: list[bool] = []

        def callback() -> None:
            callback_called.append(True)

        watcher = FileWatcher(test_file, callback)
        watcher.start()

        time.sleep(0.1)

        # Modify the directory (not the file)
        (test_file.parent / "other_file.txt").write_text("other")

        time.sleep(0.3)
        watcher.stop()

        # Callback should not be called for other files
        # (It might be called if the test environment monitors all changes, but ideally shouldn't)
        # The main test is that it doesn't crash
        assert len(callback_called) == 0
