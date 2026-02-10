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

import pytest

from fixingahole.profiler.profile_summary import ProfileSummary
from fixingahole.profiler.stats_manager import StatisticsManager


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
        for values in avg.values():
            assert "user" in values
            assert "system" in values
            assert "memory" in values
            assert all(isinstance(v, float) for v in values.values())

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
