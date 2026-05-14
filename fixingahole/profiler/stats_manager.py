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
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import git
from colours import Colour

from fixingahole import Config
from fixingahole.profiler.utils import date, memory_with_units

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


def _format_stats_duration(avg_sec: float, std_sec: float) -> str:
    """Format an average ± std duration for display in benchmark summaries.

    Selects the most readable unit (hours, minutes, seconds, milliseconds, or
    microseconds) based on the magnitude of ``avg_sec``, then formats both
    values in the same unit.  The ``±`` portion is omitted when ``std_sec``
    is zero (e.g. a single-run summary).
    """
    if avg_sec >= 3600:  # noqa: PLR2004
        val, std, unit = avg_sec / 3600, std_sec / 3600, "hr"
    elif avg_sec >= 60:  # noqa: PLR2004
        val, std, unit = avg_sec / 60, std_sec / 60, "min"
    elif avg_sec >= 1:
        val, std, unit = avg_sec, std_sec, "sec"
    elif avg_sec >= 0.001:  # noqa: PLR2004
        val, std, unit = avg_sec * 1_000, std_sec * 1_000, "ms"
    else:
        val, std, unit = avg_sec * 1_000_000, std_sec * 1_000_000, "\u00b5s"
    if std > 0:
        return f"{val:.3f} \u00b1 {std:.3f} {unit}"
    return f"{val:.3f} {unit}"


def _format_stats_memory(avg_mb: float, std_mb: float) -> str:
    """Format an average ± std memory value with auto-scaled units.

    Uses ``memory_with_units`` for consistent unit scaling.  The ``±`` portion
    is omitted when ``std_mb`` is zero.
    """
    avg_str = memory_with_units(avg_mb)
    if std_mb > 0:
        return f"{avg_str} \u00b1 {memory_with_units(std_mb)}"
    return avg_str


@dataclass(frozen=True)
class StatsProfileDetails:
    """Averaged profiling statistics for a single function across multiple runs.

    Satisfies ``FunctionDataProtocol`` so it can be passed to
    ``build_module_tree`` alongside ``ProfileDetails`` objects.

    ``total_percentage`` holds the *total average runtime in seconds*
    (user + system), not a CPU percentage.  It is repurposed as the sort key
    and filter comparator inside the tree-building utilities; callers must pass
    a ``threshold`` in seconds (not percent) when using stats data.
    """

    name: str
    file_path: str
    avg_user: float
    avg_system: float
    avg_memory: float
    std_user: float
    std_system: float
    std_memory: float

    @cached_property
    def total_percentage(self) -> float:
        """Total average runtime in seconds (user + system). Used as a sort/filter key."""
        return self.avg_user + self.avg_system

    @cached_property
    def has_memory_info(self) -> bool:
        """Whether this function has non-zero average memory usage."""
        return self.avg_memory > 0

    @cached_property
    def peak_memory(self) -> float:
        """Average peak memory in MB."""
        return self.avg_memory

    @cached_property
    def peak_memory_info(self) -> str:
        """Average peak memory formatted with auto-scaled units."""
        return memory_with_units(self.avg_memory)


def _build_stats_details(stats_data: dict[str, Any]) -> dict[str, list[StatsProfileDetails]]:
    """Convert a stats dict to ``StatsProfileDetails`` objects grouped by file path.

    Skips the ``"metadata"`` key and any malformed entries that lack both a
    file path and a function name component.

    Relative file paths (as written by ``StatisticsManager.insert``) are resolved
    to absolute paths using ``Config.root()`` so that ``build_module_tree``'s
    ``os.path.commonpath`` call receives a homogeneous list of absolute paths.
    """
    by_file: dict[str, list[StatsProfileDetails]] = defaultdict(list)
    for key, data in stats_data.items():
        if key == "metadata":
            continue
        # Keys have the form "relative/file/path.py:function_name"
        file_path, sep, name = key.rpartition(":")
        if not sep or not file_path or not name:
            continue
        # Resolve to absolute so build_module_tree's commonpath call works uniformly.
        abs_file_path = file_path if Path(file_path).is_absolute() else str(Config.root() / file_path)
        by_file[abs_file_path].append(
            StatsProfileDetails(
                name=name,
                file_path=abs_file_path,
                avg_user=data["user"]["avg"],
                avg_system=data["system"]["avg"],
                avg_memory=data["memory"]["avg"],
                std_user=data["user"]["std"],
                std_system=data["system"]["std"],
                std_memory=data["memory"]["std"],
            )
        )
    return dict(by_file)


def _get_all_tree_functions(tree_dict: dict[str, Any]) -> list:
    """Collect all function lists from a nested tree structure built by ``build_module_tree``."""
    all_functions = []
    for data in tree_dict.values():
        if data.get("_functions"):
            all_functions.append(data["_functions"])
        if data.get("_children"):
            all_functions.extend(_get_all_tree_functions(data["_children"]))
    return all_functions


def render_stats_tree(
    tree_dict: dict[str, Any],
    prefix: str = "",
    max_func_name_length: int = 50,
    threshold_sec: float = 0.001,
) -> list[str]:
    """Render a module tree built from ``StatsProfileDetails`` with avg \u00b1 std formatting.

    Mirrors the structure of ``render_tree`` in ``profile_summary`` but formats
    each function line as ``avg \u00b1 std`` in human-readable time/memory units
    instead of a CPU percentage.
    """
    lines: list[str] = []
    items = list(tree_dict.items())

    if len(items) > 1:
        # Sort directories/files from largest to smallest total average runtime.
        items = sorted(
            items,
            key=lambda item: sum(f.total_percentage for f in item[1].get("_functions", []))
            + sum(
                f.total_percentage for file_funcs in _get_all_tree_functions(item[1].get("_children", {})) for f in file_funcs
            ),
            reverse=True,
        )

    ang, tee, bar, blk = "\u2514\u2500 ", "\u251c\u2500 ", "\u2502  ", "   "
    for i, (name, data) in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = prefix + (ang if is_last else tee)
        next_prefix = prefix + (blk if is_last else bar)

        functions: list[StatsProfileDetails] = sorted(
            data.get("_functions", []), key=lambda f: f.total_percentage, reverse=True
        )
        functions = [f for f in functions if f.total_percentage >= threshold_sec or f.has_memory_info]
        children: dict[str, Any] = data.get("_children", {})

        if functions:
            total_avg = sum(f.total_percentage for f in functions)
            total_std = sum(f.std_user + f.std_system for f in functions)
            dur = _format_stats_duration(total_avg, total_std)
            lines.append(f"{current_prefix}{name} ({len(functions)} func, {dur} avg)")

            for j, func in enumerate(functions):
                func_is_last = j == len(functions) - 1
                func_prefix = next_prefix + (ang if func_is_last else tee)
                runtime_str = _format_stats_duration(
                    func.avg_user + func.avg_system,
                    func.std_user + func.std_system,
                )
                mem_str = f" ({_format_stats_memory(func.avg_memory, func.std_memory)})" if func.has_memory_info else ""
                lines.append(
                    f"{func_prefix}{func.name:.<{max(max_func_name_length - len(func_prefix), 2)}}{runtime_str}{mem_str}"
                )
            lines.append(next_prefix)

        elif children:
            total_avg = 0.0
            total_std = 0.0
            function_count = 0
            has_memory = False
            for file_funcs in _get_all_tree_functions(children):
                for f in file_funcs:
                    has_memory = has_memory or f.has_memory_info
                    if f.total_percentage >= threshold_sec or f.has_memory_info:
                        total_avg += f.total_percentage
                        total_std += f.std_user + f.std_system
                        function_count += 1
            if function_count == 0 and not has_memory:
                return lines
            dur = _format_stats_duration(total_avg, total_std)
            lines.append(f"{current_prefix}{name} ({function_count} func, {dur} avg)")
            lines.extend(
                render_stats_tree(
                    children,
                    prefix=next_prefix,
                    max_func_name_length=max_func_name_length,
                    threshold_sec=threshold_sec,
                )
            )

    return lines


def generate_stats_summary(
    stats_data: dict[str, Any],
    top_n: int = 10,
    threshold_sec: float = 0.001,
) -> str:
    """Generate a human-readable benchmark summary from a stats data dict.

    The `stats_data` dict is the output of `StatisticsManager.stats()`.
    Each function entry is shown with its average runtime and memory alongside
    the sample standard deviation so that any variability is immediately visible.

    Args:
        stats_data: Mapping of `"file_path:func_name"` keys to stats dicts
            containing `user`, `system`, `memory`, and `count` sub-keys.
            A `metadata` key is ignored if present.
        top_n: Maximum number of functions to list in each ranked section.
        threshold_sec: Functions whose average total runtime (user + system)
            is below this threshold *and* have no memory data are excluded from
            the ranked lists and the module tree.

    """
    from fixingahole.profiler.profile_summary import build_module_tree  # noqa: PLC0415

    function_stats = {k: v for k, v in stats_data.items() if k != "metadata"}
    if not function_stats:
        return "No statistics to summarize.\n"

    by_file = _build_stats_details(function_stats)
    all_funcs: list[StatsProfileDetails] = [f for funcs in by_file.values() for f in funcs]

    count: int = max(v.get("count", 0) for v in function_stats.values())

    # Compute column widths for the ranked lists.
    max_func_name_length = max((len(f.name) for f in all_funcs), default=10)
    max_file_name_length = max((len(Path(fp).name) + 3 for fp in by_file), default=10)
    has_memory_info = any(f.has_memory_info for f in all_funcs)
    dur_width = 28  # wide enough for "XXX.XXX \u00b1 YYY.YYY sec"
    mem_width = 22  # wide enough for "XXX.XXX GB \u00b1 YYY.YYY GB"
    whitespace_width = 7
    width = max_func_name_length + max_file_name_length + max(dur_width, mem_width) + whitespace_width

    plural = f"{count} runs" if count != 1 else "1 run"
    message: list[str] = [f"\nBenchmark Summary ({plural})", "=" * width]

    # --- Top N by average total runtime ---
    ranked = sorted(
        [f for f in all_funcs if f.total_percentage >= threshold_sec],
        key=lambda f: f.total_percentage,
        reverse=True,
    )[:top_n]
    n = len(ranked)
    if n == 0:
        message += ["\nNo functions above the runtime threshold.", "-" * width]
    else:
        message += [
            f"\nTop {f'{n} Functions' if n > 1 else 'Function'} by Average Runtime:",
            "-" * width,
        ]
        for i, func in enumerate(ranked, 1):
            file_name = Path(func.file_path).name
            runtime_str = _format_stats_duration(
                func.avg_user + func.avg_system,
                func.std_user + func.std_system,
            )
            mem_str = f"  ({_format_stats_memory(func.avg_memory, func.std_memory)})" if func.has_memory_info else ""
            message.append(f"{i:2d}. {func.name:<{max_func_name_length}} {runtime_str:<{dur_width}}{mem_str}  ({file_name})")

    # --- Top N by average memory usage ---
    if has_memory_info:
        memory_ranked = sorted(
            [f for f in all_funcs if f.avg_memory > 0],
            key=lambda f: f.avg_memory,
            reverse=True,
        )[:top_n]
        n = len(memory_ranked)
        if memory_ranked:
            message += [
                f"\nTop {f'{n} Functions' if n > 1 else 'Function'} by Average Memory:",
                "-" * width,
            ]
            for i, func in enumerate(memory_ranked, 1):
                file_name = Path(func.file_path).name
                mem_str = _format_stats_memory(func.avg_memory, func.std_memory)
                message.append(f"{i:2d}. {func.name:<{max_func_name_length}} {mem_str:>{mem_width}}  ({file_name})")

    # --- Module tree ---
    message.append("\nFunctions by Module:")
    message.append("-" * width)
    module_tree, depth = build_module_tree(by_file, threshold=threshold_sec)  # type: ignore[arg-type]
    tree_width = max_func_name_length + (depth + 2) * 3
    tree_lines = render_stats_tree(
        module_tree,
        max_func_name_length=tree_width,
        threshold_sec=threshold_sec,
    )
    message.extend(tree_lines)
    message.append("")

    message.extend(["=" * width, "\n"])
    return "\n".join(line.rstrip() for line in message)


class StatsSummary:
    """Summary generator for benchmark statistics produced by ``StatisticsManager``.

    Can be constructed from either an in-memory ``StatisticsManager`` or a
    previously saved statistics JSON file, making it equally useful after a
    live benchmarking run and when post-processing archived results.

    Example::

        # From a live run
        manager = StatisticsManager()
        for _ in range(10):
            manager.insert(profiler.run_profiler())
        print(StatsSummary(manager).summary())

        # From a saved JSON file
        print(StatsSummary(Path("function_stats.json")).summary())
    """

    def __init__(self, source: "StatisticsManager | Path") -> None:
        if isinstance(source, (Path, str)):
            self._stats_data: dict[str, Any] = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            self._stats_data = source.stats()
        function_data = {k: v for k, v in self._stats_data.items() if k != "metadata"}
        self.count: int = next(iter(function_data.values()), {}).get("count", 0)

    def summary(self, top_n: int = 10, threshold_sec: float = 0.001) -> str:
        """Generate a human-readable benchmark summary with avg \u00b1 std statistics.

        Args:
            top_n: Maximum number of functions to list in each ranked section.
            threshold_sec: Functions whose average total runtime is below this
                value (in seconds) and have no memory data are excluded.

        """
        return generate_stats_summary(self._stats_data, top_n=top_n, threshold_sec=threshold_sec)


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
                key = f"{Config.relative_to_root(f.file_path)}:{f.name}"
            except ValueError:  # f.file_path is not in the subpath of Config.root()
                key = f"{Path(f.file_path)}:{f.name}"
            self.function_data[key].append(f)

    def average(self) -> dict[str, dict[str, float]]:
        """Compute the averages for each function."""
        return {
            key: {
                "user_avg": _mean([f.user_time for f in funcs], self.count),
                "system_avg": _mean([f.system_time for f in funcs], self.count),
                "memory_avg": _mean([f.peak_memory for f in funcs], self.count),
                "count": self.count,
            }
            for key, funcs in self.function_data.items()
        }

    def std(self) -> dict[str, dict[str, float]]:
        """Compute the standard deviations for each function."""
        return {
            key: {
                "user_std": _std([f.user_time for f in funcs], self.count),
                "system_std": _std([f.system_time for f in funcs], self.count),
                "memory_std": _std([f.peak_memory for f in funcs], self.count),
                "count": self.count,
            }
            for key, funcs in self.function_data.items()
        }

    def stats(self) -> dict[str, dict[str, Any]]:
        """Compute the standard deviations for each function."""
        return {
            key: {
                "user": _mean_and_std([f.user_time for f in funcs], self.count),
                "system": _mean_and_std([f.system_time for f in funcs], self.count),
                "memory": _mean_and_std([f.peak_memory for f in funcs], self.count),
                "count": self.count,
            }
            for key, funcs in self.function_data.items()
        }

    def summary(self, top_n: int = 10, threshold_sec: float = 0.001) -> str:
        """Generate a human-readable benchmark summary with avg ± std statistics.

        Args:
            top_n: Maximum number of functions to list in each ranked section.
            threshold_sec: Functions whose average total runtime is below this
                value (in seconds) and have no memory data are excluded.

        """
        return generate_stats_summary(self.stats(), top_n=top_n, threshold_sec=threshold_sec)

    @staticmethod
    def save_as_json(filename: Path, data: dict[str, Any], *, save_metadata: bool = True, sort: bool = True) -> dict[str, Any]:
        """Location to save the benchmarking statistics."""
        if not data:
            Colour.warning("Warning: data is empty. Nothing to save.")
            return data

        filename = (Path(filename) if isinstance(filename, str) else filename).resolve()
        if filename.exists():
            Colour.warning("Warning: %s already exists. Overwriting file.", Colour.purple(Config.relative_to_cwd(filename)))

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
                    with git.Repo(Config.root(), search_parent_directories=True) as repo:
                        if value := method(repo):
                            save_data["metadata"][info] = value
                except (TypeError, git.InvalidGitRepositoryError, git.exc.NoSuchPathError):
                    Colour.error("Error: Failed to save git %s", info)
                    save_data["metadata"][info] = f"Failed to save git {info}."

        save_data = save_data or data
        filename.write_text(json.dumps(save_data, indent=1))
        return save_data
