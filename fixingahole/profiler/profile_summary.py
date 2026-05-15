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
"""Profile Results Summarizer.

This module extracts function details from the Scalene data
and presents the data in a tree form for easier interpretation.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fixingahole import Config
from fixingahole.profiler.scalene_json_parser import ProfileData, ProfileDetails
from fixingahole.profiler.utils import format_time, installed_modules


@runtime_checkable
class FunctionDataProtocol(Protocol):
    """Minimal interface for function profiling data consumed by the module tree.

    Both ``ProfileDetails`` (single-run Scalene data) and ``StatsProfileDetails``
    (averaged benchmark data) satisfy this protocol, allowing ``build_module_tree``
    to work with either without modification.

    Note: ``total_percentage`` carries different semantics depending on the
    concrete type — a real CPU percentage for ``ProfileDetails`` and total average
    runtime in seconds for ``StatsProfileDetails``.  It is used only as a sort key
    and filter comparator inside the tree functions, so the unit mismatch is safe as
    long as callers pass a ``threshold`` in the appropriate units.
    """

    @property
    def name(self) -> str:
        """Function name."""
        ...

    @property
    def file_path(self) -> str:
        """Absolute or relative path to the source file."""
        ...

    @property
    def total_percentage(self) -> float:
        """Sort/filter key; unit depends on the concrete type (% or seconds)."""
        ...

    @property
    def has_memory_info(self) -> bool:
        """True when this function has non-zero memory profiling data."""
        ...

    @property
    def peak_memory(self) -> float:
        """Peak heap memory in MB."""
        ...

    @property
    def peak_memory_info(self) -> str:
        """Peak memory formatted with auto-scaled units."""
        ...


def _build_top_n_section(
    funcs: list,
    top_n: int,
    width: int,
    *,
    section_label: str,
    empty_message: str | None,
    filter_fn: Callable[[Any], bool],
    sort_key: Callable[[Any], float],
    format_line: Callable[[int, Any], str],
) -> list[str]:
    """Build the header and function lines for a ranked summary section.

    Returns an empty list when ``empty_message`` is ``None`` and there are no
    qualifying functions (i.e., the section is silently omitted).
    """
    ranked = sorted([f for f in funcs if filter_fn(f)], key=sort_key, reverse=True)[:top_n]
    n = len(ranked)
    if n == 0:
        return [empty_message, "-" * width] if empty_message is not None else []
    lines = [
        f"\nTop {f'{n} Functions' if n > 1 else 'Function'} by {section_label}:",
        "-" * width,
    ]
    lines.extend(format_line(i, f) for i, f in enumerate(ranked, 1))
    return lines


def generate_summary(profile_data: ProfileData, top_n: int = 10, threshold: float = 0.1) -> str:
    """Generate a summary of the profiling results."""
    functions = profile_data.functions
    if not functions:
        return "No functions to summarize.\n"

    has_memory_info = False
    max_func_name_length = 0
    max_file_name_length = 0
    max_lineno_length = 0
    by_file = profile_data.functions_by_file
    for file, file_functions in by_file.items():
        for func in file_functions:
            max_func_name_length = max(
                len(func.name),
                max_func_name_length,
            )
            max_file_name_length = max(
                len(Path(file).name) + 3,
                max_file_name_length,
            )
            max_lineno_length = max(
                len(str(func.line_number)),
                max_lineno_length,
            )
            has_memory_info = has_memory_info or func.has_memory_info

    runtime_width = 6
    mem_width = 8
    whitespace_width = 7
    width = max_func_name_length + max_file_name_length + max(runtime_width, mem_width) + whitespace_width + max_lineno_length
    message: list[str] = ["\nProfile Summary", "=" * width]

    walltime = profile_data.walltime

    def fmt_runtime_line(i: int, func: ProfileDetails) -> str:
        file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
        runtime_info = (
            f"{func.total_percentage:>5.2f}%"
            if Config.is_duration_relative()
            else format_time(func.total_percentage * walltime / 100, walltime)
        )
        return f"{i:2d}. {func.name:<{max_func_name_length}} {runtime_info:<{runtime_width}} ({file_name}:{func.line_number})"

    message.extend(
        _build_top_n_section(
            functions,
            top_n,
            width,
            section_label="Total Runtime",
            empty_message="\nNo functions to summarize by Total Runtime",
            filter_fn=lambda f: f.total_percentage >= threshold,
            sort_key=lambda f: f.total_percentage,
            format_line=fmt_runtime_line,
        )
    )

    if has_memory_info:

        def fmt_memory_line(i: int, func: ProfileDetails) -> str:
            file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
            return f"{i:2d}. {func.name:<{max_func_name_length}} {func.peak_memory_info:>{mem_width}} ({file_name}:{func.line_number})"  # noqa: E501

        message.extend(
            _build_top_n_section(
                functions,
                top_n,
                width,
                section_label="Memory Usage",
                empty_message=None,
                filter_fn=lambda f: f.peak_memory > 0,
                sort_key=lambda f: f.peak_memory,
                format_line=fmt_memory_line,
            )
        )

    message.extend(("\nFunctions by Module:", "-" * width))
    module_tree, depth = build_module_tree(by_file, threshold=threshold)
    tree_width = max_func_name_length + (depth + 2) * 3  # depth + 2 because the minimum tree depth is 2.
    tree = render_tree(module_tree, profile_data.walltime, max_func_name_length=tree_width, threshold=threshold)
    message.extend(tree)
    message.append("")

    message.extend(["=" * width, "\n"])
    return "\n".join(line.rstrip() for line in message)


def build_module_tree(
    by_file_dict: Mapping[str, Sequence[FunctionDataProtocol]],
    threshold: float = 0.1,
) -> tuple[dict[str, Any], int]:
    """Build a hierarchical tree structure from file paths and compute the tree's max depth."""
    modules = installed_modules()
    tree: dict[str, Any] = {}
    files: list[str] = [file for file in by_file_dict if file[0] != "<" and file[-1] != ">"]
    common_root = Path(os.path.commonpath(files if len(files) > 1 else [*files, Config.root()]))
    depth = 0
    for file_path in files:
        file_functions = by_file_dict[file_path]
        if not any(f.total_percentage >= threshold or f.has_memory_info for f in file_functions):
            continue
        d = 1
        parts = Path(file_path).relative_to(common_root).parts
        for i, part in enumerate(parts):
            if part in modules:
                parts = parts[i:]
                break
        current = tree
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {"_functions": [], "_children": {}}
            if i == len(parts) - 1:
                current[part]["_functions"] = file_functions
            else:
                d += 1
                depth = max(d, depth)
                current = current[part]["_children"]
    depth += 1
    return tree, depth


def get_all_functions_in_tree(tree_dict: dict[str, Any]) -> list:
    """Get all function lists from a tree structure."""
    all_functions = []
    for data in tree_dict.values():
        if data.get("_functions"):
            all_functions.append(data["_functions"])
        if data.get("_children"):
            all_functions.extend(get_all_functions_in_tree(data["_children"]))
    return all_functions


def _render_tree_core(
    tree_dict: dict[str, Any],
    format_node_dur: Callable[[list], str],
    format_func_runtime: Callable[[Any], str],
    aggregate_children: Callable[[dict, float], tuple[int, str, bool]],
    *,
    prefix: str = "",
    max_func_name_length: int = 50,
    threshold: float = 0.0,
) -> list[str]:
    """Shared rendering core for ``render_tree`` and ``render_stats_tree``.

    ``aggregate_children`` returns ``(function_count, dur_str, skip)`` where
    ``skip=True`` causes the whole subtree to be omitted from the output.
    """
    lines: list[str] = []
    items = list(tree_dict.items())

    if len(items) > 1:
        items = sorted(
            items,
            key=lambda item: sum(f.total_percentage for f in item[1].get("_functions", []))
            + sum(
                f.total_percentage for file_funcs in get_all_functions_in_tree(item[1].get("_children", {})) for f in file_funcs
            ),
            reverse=True,
        )

    ang, tee, bar, blk = "└─ ", "├─ ", "│  ", "   "
    for i, (name, data) in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = prefix + (ang if is_last else tee)
        next_prefix = prefix + (blk if is_last else bar)

        functions = sorted(data.get("_functions", []), key=lambda f: f.total_percentage, reverse=True)
        functions = [f for f in functions if f.total_percentage >= threshold or f.has_memory_info]
        children: dict[str, Any] = data.get("_children", {})

        if functions:
            dur = format_node_dur(functions)
            lines.append(f"{current_prefix}{name} ({len(functions)} func, {dur})")
            for j, func in enumerate(functions):
                func_is_last = j == len(functions) - 1
                func_prefix = next_prefix + (ang if func_is_last else tee)
                runtime_str = format_func_runtime(func)
                lines.append(f"{func_prefix}{func.name:.<{max(max_func_name_length - len(func_prefix), 2)}}{runtime_str}")
            lines.append(next_prefix)
        elif children:
            function_count, dur, skip = aggregate_children(children, threshold)
            if skip:
                return lines
            lines.append(f"{current_prefix}{name} ({function_count} func, {dur})")
            lines.extend(
                _render_tree_core(
                    children,
                    format_node_dur,
                    format_func_runtime,
                    aggregate_children,
                    prefix=next_prefix,
                    max_func_name_length=max_func_name_length,
                    threshold=threshold,
                )
            )

    return lines


def render_tree(
    tree_dict: dict[str, Any],
    walltime: float,
    prefix: str = "",
    max_func_name_length: int = 50,
    threshold: float = 0.1,
) -> list[str]:
    """Render the module tree with proper indentation."""

    def format_node_dur(functions: list) -> str:
        total_runtime = sum(f.total_percentage for f in functions)
        return (
            f"{total_runtime:.2f}% total"
            if Config.is_duration_relative()
            else format_time(total_runtime * walltime / 100, walltime)
        )

    def format_func_runtime(func: FunctionDataProtocol) -> str:
        peak_mem = f" ({func.peak_memory_info})" if func.has_memory_info else ""
        return (
            f"{func.total_percentage:.>5.2f}%"
            if Config.is_duration_relative()
            else format_time(func.total_percentage * walltime / 100, walltime)
        ) + peak_mem

    def aggregate_children(children: dict, thr: float) -> tuple[int, str, bool]:
        total_runtime = 0.0
        function_count = 0
        has_memory = False
        for file_funcs in get_all_functions_in_tree(children):
            for f in file_funcs:
                has_memory = has_memory or f.has_memory_info
                if f.total_percentage >= thr or f.has_memory_info:
                    total_runtime += f.total_percentage
                    function_count += 1
        skip = function_count == 0 and not has_memory
        dur = (
            f"{total_runtime:.2f}% total"
            if Config.is_duration_relative()
            else format_time(total_runtime * walltime / 100, walltime)
        )
        return function_count, dur, skip

    return _render_tree_core(
        tree_dict,
        format_node_dur,
        format_func_runtime,
        aggregate_children,
        prefix=prefix,
        max_func_name_length=max_func_name_length,
        threshold=threshold,
    )


class ProfileSummary:
    """Parser for summarizing scalene cli profile results files."""

    def __init__(self, filename: Path):
        self.data = ProfileData.from_file(filename)
        self.walltime: float = self.data.walltime
        self.max_memory: str = self.data.max_memory

    def summary(self, top_n: int = 10, threshold: float = 0.1) -> str:
        """Generate a summary of the profiling results."""
        return generate_summary(self.data, top_n, threshold)
