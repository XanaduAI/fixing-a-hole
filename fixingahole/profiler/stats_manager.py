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
"""Statistics Manager for Profile Results when Benchmarking."""

import json
import math
from collections import defaultdict
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import git
from colours import Colour

from fixingahole import Config
from fixingahole.profiler.utils import date

if TYPE_CHECKING:
    from collections.abc import Callable

    from fixingahole.profiler.profile_summary import ProfileDetails, ProfileSummary


def _get_dirty_files(repo: git.Repo) -> set[str]:
    """Check the git status and return the filenames of uncommitted changes."""
    dirty_files = set()

    # 1. Check for unstaged changes (Diff between Index and Working Tree)
    # passing None compares the Index to the Working Tree
    dirty_files.update(item.a_path for item in repo.index.diff(None))

    # 2. Check for staged changes (Diff between Index and HEAD)
    # We compare the HEAD commit to the current Index
    dirty_files.update(item.a_path for item in repo.index.diff("HEAD"))

    return dirty_files


def _get_used_dirty_files(repo: git.Repo, data: dict) -> list[str]:
    """Compute the intersection of files with uncommitted changes and the profiled files."""
    if not repo.is_dirty():
        return []

    used_files: set[str] = {f.split(":").pop(0) for f in set(data)}
    dirty_files = _get_dirty_files(repo)
    return list(dirty_files.intersection(used_files))


def _mean(values: list[float], count: int | None) -> float:
    """Compute the mean (average) given a list of values."""
    count: int = count if count is not None else len(values)
    return sum(value for value in values) / count


def _std(values: list[float], count: int | None, mean: float | None = None) -> float:
    """Compute the sample standard deviation given a list of values."""
    count: int = count if count is not None else len(values)
    mean: float = mean if mean is not None else _mean(values, count)
    return math.sqrt(sum(pow(value - mean, 2) for value in values) / (count - 1)) if count > 1 else 0.0


def _mean_and_std(values: list[float], count: int | None, mean: float | None = None) -> dict[str, float]:
    """Compute the mean and sample standard deviation given a list of values."""
    count: int = count if count is not None else len(values)
    mean: float = mean if mean is not None else _mean(values, count)
    std: float = math.sqrt(sum(pow(value - mean, 2) for value in values) / (count - 1)) if count > 1 else 0.0
    return {"avg": mean, "std": std}


class StatisticsManager:
    """Statistics Manager for Profile Results when Benchmarking."""

    def __init__(self) -> None:
        self.count: int = 0
        self.function_data: dict[str, list[ProfileDetails]] = defaultdict(list)

    def insert(self, summary: "ProfileSummary") -> None:
        """Add additional function data to the stats manager."""
        self.count += 1
        for f in summary.data.functions:
            try:
                key = f"{Path(f.file_path).relative_to(Config.root())}:{f.name}"
            except ValueError:  # f.file_path is not in the subpath of Config.root()
                key = f"{Path(f.file_path)}:{f.name}"
            self.function_data[key].append(f)

    def average(self) -> dict[str, dict[str, float]]:
        """Compute the averages for each function."""
        res: dict[str, dict[str, float]] = {}
        for key, funcs in self.function_data.items():
            res[key] = {
                "user_avg": _mean([f.user_time for f in funcs], self.count),
                "system_avg": _mean([f.system_time for f in funcs], self.count),
                "memory_avg": _mean([f.peak_memory for f in funcs], self.count),
                "count": self.count,
            }
        return res

    def std(self) -> dict[str, dict[str, float]]:
        """Compute the standard deviations for each function."""
        res: dict[str, dict[str, float]] = {}
        for key, funcs in self.function_data.items():
            res[key] = {
                "user_std": _std([f.user_time for f in funcs], self.count),
                "system_std": _std([f.system_time for f in funcs], self.count),
                "memory_std": _std([f.peak_memory for f in funcs], self.count),
                "count": self.count,
            }
        return res

    def stats(self) -> dict[str, dict[str, Any]]:
        """Compute the standard deviations for each function."""
        res: dict[str, dict[str, Any]] = {}
        for key, funcs in self.function_data.items():
            res[key] = {
                "user": _mean_and_std([f.user_time for f in funcs], self.count),
                "system": _mean_and_std([f.system_time for f in funcs], self.count),
                "memory": _mean_and_std([f.peak_memory for f in funcs], self.count),
                "count": self.count,
            }
        return res

    @staticmethod
    def save_as_json(filename: Path, data: dict[str, Any], *, save_metadata: bool = True, sort: bool = True) -> dict[str, Any]:
        """Location to save the benchmarking statistics."""
        if not data:
            Colour.warning("Warning: data is empty. Nothing to save.")
            return data

        filename = (Path(filename) if isinstance(filename, str) else filename).resolve()
        if filename.exists():
            file = filename
            with suppress(ValueError):
                file = filename.relative_to(Path.cwd())
            Colour.warning("Warning: %s already exists. Overwriting file.", Colour.purple(file))

        save_data: dict[str, Any] = (
            dict(sorted(data.items(), key=lambda item: item[1].get("user", {}).get("avg", 0), reverse=True)) if sort else {}
        )

        if save_metadata:
            save_data: dict[str, Any] = save_data if sort else deepcopy(data)
            save_data["metadata"]: dict[str, Any] = {}
            metadata: dict[str, Callable] = {
                "repo": lambda repo: Path(str(repo.remotes.origin.url)).stem,
                "branch": lambda repo: repo.active_branch.name,
                "commit": lambda repo: repo.head.object.hexsha,
                "used_dirty_files": lambda repo: _get_used_dirty_files(repo, save_data),
                "utc_time": lambda _: date(),
            }
            for info, method in metadata.items():
                try:
                    repo = git.Repo(Config.root(), search_parent_directories=True)
                    if value := method(repo):
                        save_data["metadata"][info] = value
                except (TypeError, git.InvalidGitRepositoryError, git.exc.NoSuchPathError):
                    save_data["metadata"][info] = f"Failed to save git {info}."

        save_data = save_data or data
        filename.write_text(json.dumps(save_data, indent=1))
        return save_data
