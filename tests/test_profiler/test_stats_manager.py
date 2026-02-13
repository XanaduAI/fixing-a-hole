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
"""Tests for the StatisticsManager class."""

import json
from pathlib import Path
from typing import Never
from unittest.mock import MagicMock, patch

import pytest

from fixingahole.profiler.profile_summary import ProfileSummary
from fixingahole.profiler.stats_manager import StatisticsManager, _get_dirty_files, _get_used_dirty_files  # noqa: PLC2701


@pytest.fixture
def profile_summary_obj(example_json: Path) -> ProfileSummary:
    """Load a real ProfileSummary from example JSON."""
    return ProfileSummary(example_json)


class TestStatisticsManagerInit:
    """Test StatisticsManager initialization."""

    def test_init_creates_empty_manager(self):
        """Test that initialization creates an empty manager."""
        manager = StatisticsManager()
        assert manager.count == 0
        assert len(manager.function_data) == 0


class TestGitDiffHelperFunctions:
    """Test git diff helper functions."""

    def test_get_dirty_files(self):
        """Test that git diff can find dirty files."""
        mock_repo = MagicMock()

        # Mock unstaged changes (diff against working tree)
        unstaged_change = MagicMock()
        unstaged_change.a_path = "file1.py"

        # Mock staged changes (diff against HEAD)
        staged_change = MagicMock()
        staged_change.a_path = "file2.py"

        # Set up diff method to return appropriate changes
        mock_repo.index.diff.side_effect: list[list[MagicMock]] = [[unstaged_change], [staged_change]]

        result = _get_dirty_files(mock_repo)
        assert result == {"file1.py", "file2.py"}
        assert mock_repo.index.diff.call_count == 2

    def test_get_used_dirty_files(self):
        """Test that the dirty files from git diff are used by the profiling."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True

        # Mock dirty files
        dirty_change = MagicMock()
        dirty_change.a_path = "fixingahole/profiler/stats_manager.py"
        mock_repo.index.diff.side_effect: list[list[MagicMock]] = [[dirty_change], []]

        # Mock profiling data with files
        data: dict[str, dict[str, float]] = {
            "fixingahole/profiler/stats_manager.py:function1": {"user": 1.0},
            "fixingahole/cli/main.py:function2": {"user": 2.0},
        }

        result = _get_used_dirty_files(mock_repo, data)
        assert result == ["fixingahole/profiler/stats_manager.py"]

    def test_get_used_dirty_files_no_dirty_files(self):
        """Test that empty list is returned when repo is clean."""
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = False

        data: dict[str, dict[str, float]] = {"some/file.py:function": {"user": 1.0}}
        result = _get_used_dirty_files(mock_repo, data)
        assert result == []


class TestStatisticsManagerInsert:
    """Test StatisticsManager insert method."""

    def test_insert_single_summary(self, profile_summary_obj: ProfileSummary):
        """Test inserting a single profile summary."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        assert manager.count == 1
        assert len(manager.function_data) > 0
        # Check that function data was stored
        assert all(len(funcs) == 1 for funcs in manager.function_data.values())

    def test_insert_multiple_summaries(self, profile_summary_obj: ProfileSummary):
        """Test inserting multiple profile summaries."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)
        manager.insert(profile_summary_obj)

        assert manager.count == 2
        # Each function should have 2 entries
        assert all(len(funcs) == 2 for funcs in manager.function_data.values())


class TestStatisticsManagerAverage:
    """Test StatisticsManager average method."""

    def test_average_single_run(self, profile_summary_obj: ProfileSummary):
        """Test computing average with a single run."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        avg = manager.average()

        assert len(avg) > 0
        # With a single run, average should equal the original values
        measures: list[str] = ["user", "system", "memory"]
        for values in avg.values():
            assert "count" in values
            assert isinstance(values["count"], int)
            for measure in measures:
                assert measure in values
                assert isinstance(values[measure], float)

    def test_average_multiple_runs(self, profile_summary_obj: ProfileSummary):
        """Test computing average with multiple identical runs."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)
        manager.insert(profile_summary_obj)

        avg = manager.average()

        # With identical runs, average should equal original values
        first_func = profile_summary_obj.data.functions[0]
        key = next(iter(manager.function_data))

        assert avg[key]["user"] == pytest.approx(first_func.user_time)
        assert avg[key]["system"] == pytest.approx(first_func.system_time)
        assert avg[key]["memory"] == pytest.approx(first_func.peak_memory)


class TestStatisticsManagerStd:
    """Test StatisticsManager std method."""

    def test_std_single_run(self, profile_summary_obj: ProfileSummary):
        """Test computing std with a single run (should be 0)."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        std = manager.std()

        assert len(std) > 0
        # With a single run, std should be 0
        for values in std.values():
            assert values["user_std"] == 0.0
            assert values["system_std"] == 0.0
            assert values["memory_std"] == 0.0

    def test_std_identical_runs(self, profile_summary_obj: ProfileSummary):
        """Test computing std with multiple identical runs."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)
        manager.insert(profile_summary_obj)

        std = manager.std()

        # With identical runs, std should be 0
        for values in std.values():
            assert values["user_std"] == 0.0
            assert values["system_std"] == 0.0
            assert values["memory_std"] == 0.0


class TestStatisticsManagerStats:
    """Test StatisticsManager stats method."""

    def test_stats_combines_avg_and_std(self, profile_summary_obj: ProfileSummary):
        """Test that stats returns both average and std."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        stats = manager.stats()

        assert len(stats) > 0
        for metrics in stats.values():
            assert "user" in metrics
            assert "system" in metrics
            assert "memory" in metrics
            # Each metric should have avg and std
            for metric in ["user", "system", "memory"]:
                assert "avg" in metrics[metric]
                assert "std" in metrics[metric]
                assert isinstance(metrics[metric]["avg"], float)
                assert isinstance(metrics[metric]["std"], float)

    def test_stats_structure(self, profile_summary_obj: ProfileSummary):
        """Test the nested structure of stats output."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)
        manager.insert(profile_summary_obj)

        stats = manager.stats()

        # Verify nested structure
        first_key = next(iter(stats))
        assert isinstance(stats[first_key], dict)
        assert isinstance(stats[first_key]["user"], dict)
        assert stats[first_key]["user"]["std"] == 0.0  # Identical runs


class TestStatisticsManagerSaveAsJson:
    """Test StatisticsManager save_as_json method."""

    def test_save_unsorted_json(self, profile_summary_obj: ProfileSummary, tmp_path: Path):
        """Test saving JSON without sorting."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        stats = manager.stats()
        file_path = tmp_path / "test_unsorted.json"

        StatisticsManager.save_as_json(file_path, stats, sort=False, save_metadata=False)

        assert file_path.exists()
        saved_data = json.loads(file_path.read_text())
        assert len(saved_data) == len(stats)

    def test_save_sorted_json(self, profile_summary_obj: ProfileSummary, tmp_path: Path):
        """Test saving JSON with sorting by user avg."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        stats = manager.stats()
        file_path = tmp_path / "test_sorted.json"

        StatisticsManager.save_as_json(file_path, stats, sort=True, save_metadata=False)

        assert file_path.exists()
        saved_data = json.loads(file_path.read_text())

        # Verify data is sorted by user.avg in descending order
        user_avgs = [data["user"]["avg"] for data in saved_data.values()]
        assert user_avgs == sorted(user_avgs, reverse=True)

    def test_save_with_metadata(self, profile_summary_obj: ProfileSummary, tmp_path: Path):
        """Test saving JSON with along with the git metadata."""
        manager = StatisticsManager()
        manager.insert(profile_summary_obj)

        stats = manager.stats()
        file_path = tmp_path / "test_sorted.json"

        def raise_error() -> Never:
            msg = "This is an mocked error."
            raise TypeError(msg)

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "mocked_branch_name"
        mock_repo.head.object.hexsha = "mocked_commit_sha_abc123def456"
        mock_repo.remotes.origin.url = "https://github.com/xanadu/mocked_repo_name.git"
        mock_repo.is_dirty.side_effect = raise_error
        with patch("fixingahole.profiler.stats_manager.git.Repo") as mock_git_repo:
            mock_git_repo.return_value = mock_repo
            output = StatisticsManager.save_as_json(file_path, stats, sort=False, save_metadata=True)

        assert file_path.exists()
        saved_data = json.loads(file_path.read_text())

        # Verify metadata was saved
        assert "metadata" in output
        assert saved_data == output
        assert output["metadata"]["repo"] == "mocked_repo_name"
        assert output["metadata"]["branch"] == "mocked_branch_name"
        assert output["metadata"]["commit"] == "mocked_commit_sha_abc123def456"
        assert output["metadata"]["used_dirty_files"] == "Failed to save git used_dirty_files."
        assert isinstance(output["metadata"]["utc_time"], str)
        assert len(output["metadata"]["utc_time"]) == 15  # of the form 20251231_123456
