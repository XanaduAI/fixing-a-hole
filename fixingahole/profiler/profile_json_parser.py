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
"""Profile Results Parser.

This module parses profile results files and extracts function names and runtime percentages from
the function summary sections at the bottom of each file's profiling table.
"""

import contextlib
import importlib.metadata
import json
import operator
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from string import printable
from typing import Any

from colours import Colour
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from fixingahole import ROOT_DIR
from fixingahole.profiler.utils import memory_with_units

# Some useful global characters.
# Sparkline characters from lowest to highest
__sparkline_bars = "".join([chr(i) for i in range(9601, 9609)])  # "▁▂▃▄▅▆▇█"
__printable = set(printable)
__table_bar = {
    "vertical": chr(0x2502),  # "│"
    "horizontal": chr(0x2500),  # "─"
    "dbl_horizontal": chr(0x2501),  # "━"
    "top_left": chr(0x2577),  # "╷"
    "bottom_left": chr(0x2575),  # "╵"
    "left_t": chr(0x253C),  # "┼"
    "right_t": chr(0x2574),  # "╴"
    "left_double_t": chr(0x2576),  # "╶"
    "en_dash": chr(0x2013),  # "–" en-dash  # noqa: RUF003
}


@dataclass(frozen=True)
class ProfileDetails:
    """Represents a function's profiling information."""

    name: str
    file_path: str
    line_number: int
    memory_python_percentage: float
    peak_memory: float
    python_percentage: float
    native_percentage: float
    system_percentage: float
    timeline_percentage: float
    copy_mb_per_s: float
    memory_samples: list[tuple[float, float]]

    @cached_property
    def has_memory_info(self) -> bool:
        """Determine if a line used significant memory."""
        return bool(
            self.peak_memory > 0 or self.memory_python_percentage > 0 or self.timeline_percentage > 0 or self.memory_samples
        )

    @cached_property
    def peak_memory_info(self) -> str:
        """Convert the used memory into more sensible units."""
        return memory_with_units(self.peak_memory)

    @cached_property
    def total_percentage(self) -> float:
        """Return the total percentage of the runtime."""
        return self.python_percentage + self.native_percentage + self.system_percentage


@dataclass(frozen=True)
class ProfileData:
    """Holds the parsed profile data."""

    functions: list[ProfileDetails]
    lines: dict[str, list[ProfileDetails]]
    files: dict[str, float]
    walltime: float
    max_memory: float
    samples: list[tuple[float, float]]
    details: dict[str, float]

    @cached_property
    def has_memory_info(self) -> bool:
        """Determine any memory usage data is stored."""
        return any(fn.has_memory_info for fn in self.functions) or any(
            ln.has_memory_info for lines in self.lines.values() for ln in lines
        )

    @cached_property
    def get_functions_by_file(self) -> dict[str, list[ProfileDetails]]:
        """Group functions by file path."""
        result = defaultdict(list)
        for func in self.functions:
            result[func.file_path].append(func)
        return result


def parse_json(filename: str | Path) -> ProfileData:
    """Parse profile results provided as a JSON dictionary."""
    profile_path = Path(filename)
    if not profile_path.exists():
        Colour.print(Colour.RED("Error:"), "profile", Colour.purple(filename), "does not exist.")
        return ProfileData(functions=[], lines={}, files={}, walltime=None, max_memory=None, samples=[], details={})

    content = json.loads(profile_path.read_text(encoding="utf-8"))
    function_profs: list[ProfileDetails] = []
    line_profs: dict[str, list[ProfileDetails]] = defaultdict(list)

    # Extract walltime and max memory
    walltime = content.get("elapsed_time_sec", 0)
    max_memory = memory_with_units(content.get("max_footprint_mb", 0), digits=3)

    files = content.get("files", {}) if isinstance(content, dict) else {}
    file_percentage: dict[str, float] = {}
    for file_path, info in files.items():
        file_percentage[file_path] = info.get("percent_cpu_time", 0)
        lines = info.get("lines", []) if isinstance(info, dict) else []
        for line in lines:
            kwargs = {
                "python_percentage": float(line.get("n_cpu_percent_python", 0)),
                "native_percentage": float(line.get("n_cpu_percent_c", 0)),
                "system_percentage": float(line.get("n_sys_percent", 0)),
                "peak_memory": float(line.get("n_peak_mb", 0)),
                "copy_mb_per_s": float(line.get("n_copy_mb_s", 0)),
                "memory_samples": list(line.get("memory_samples", [])),
                "memory_python_percentage": float(line.get("n_python_fraction", 0)) * 100,
                "timeline_percentage": float(line.get("n_usage_fraction", 0)) * 100,
            }
            if not any(bool(val) for val in kwargs.values()):
                continue
            line_info = {
                "file_path": file_path,
                "name": line.get("line", ""),
                "line_number": int(line.get("lineno", 0)),
            }
            line_profs[file_path].append(ProfileDetails(**(line_info | kwargs)))

        funcs = info.get("functions", []) if isinstance(info, dict) else []
        for fn in funcs:
            kwargs = {
                "file_path": file_path,
                "name": fn.get("line", ""),
                "line_number": int(fn.get("lineno", 0)),
                "python_percentage": float(fn.get("n_cpu_percent_python", 0)),
                "native_percentage": float(fn.get("n_cpu_percent_c", 0)),
                "system_percentage": float(fn.get("n_sys_percent", 0)),
                "peak_memory": float(fn.get("n_peak_mb", 0)),
                "copy_mb_per_s": float(fn.get("n_copy_mb_s", 0)),
                "memory_samples": list(fn.get("memory_samples", [])),
                "memory_python_percentage": float(fn.get("n_python_fraction", 0)) * 100,
                "timeline_percentage": float(fn.get("n_usage_fraction", 0)) * 100,
            }
            function_profs.append(ProfileDetails(**kwargs))

    keys = ["max_footprint_mb", "growth_rate", "start_time_absolute", "start_time_perf"]
    details = {k: content.get(k, -1) for k in keys}

    return ProfileData(
        functions=function_profs,
        lines=line_profs,
        files=file_percentage,
        walltime=walltime,
        max_memory=max_memory,
        samples=content.get("samples", []),
        details=details,
    )


def generate_summary(profile_data: ProfileData, top_n: int = 10, threshold: float = 0.1) -> str:
    """Generate a summary of the profiling results."""
    functions = profile_data.functions
    if not functions:
        return "No functions to summarize.\n"

    has_memory_info = False
    max_func_name_length = 0
    by_file = profile_data.get_functions_by_file
    for file_functions in by_file.values():
        for func in file_functions:
            max_func_name_length = max(
                len(func.name) + 3,
                max_func_name_length,
            )
            has_memory_info = has_memory_info or func.has_memory_info

    width = max_func_name_length + 40
    message = [f"\nProfile Summary ({profile_data.walltime or 0:,.3f}s total)", "=" * width]

    # Top functions by total runtime percentage
    top_functions = get_top_usage(functions, top_n)
    message += [
        f"\nTop {len(top_functions)} Functions by Total Runtime:",
        "-" * width,
    ]
    for i, func in enumerate(top_functions, 1):
        file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
        lineno = func.line_number
        runtime_info = f"{func.total_percentage:>5.1f}%"
        message.append(
            f"{i:2d}. {func.name:<{max_func_name_length}} {runtime_info:<6} ({file_name}:{lineno})",
        )

    # Add memory summary if available
    if has_memory_info:
        memory_functions = get_top_usage(functions, top_n, key=lambda f: f.peak_memory)
        memory_functions = [f for f in memory_functions if f.peak_memory]
        if memory_functions:
            message += [
                f"\nTop {len(memory_functions)} Functions by Memory Usage:",
                "-" * width,
            ]
            for i, func in enumerate(memory_functions, 1):
                file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
                message.append(
                    f"{i:2d}. {func.name:<{max_func_name_length}} {func.peak_memory_info:>8} ({file_name})",
                )

    message.append("\nFunctions by Module:")
    message.append("-" * width)

    module_tree = build_module_tree(by_file)
    tree = render_tree(module_tree, max_func_name_length=max_func_name_length, threshold=threshold)
    message.extend(tree)
    message.append("")

    message.extend(["=" * width, "\n"])
    return "\n".join(line.rstrip() for line in message)


def get_top_usage(
    details: list[ProfileDetails],
    n: int = 10,
    key: Callable = lambda f: f.total_percentage,
) -> list[ProfileDetails]:
    """Get the top N profile details by key."""
    return sorted(details, key=key, reverse=True)[:n]


def build_module_tree(by_file_dict: dict[str, list[ProfileDetails]]) -> dict[str, Any]:
    """Build a hierarchical tree structure from file paths."""
    modules = installed_modules()
    tree: dict[str, Any] = {}
    files = by_file_dict.keys()
    common_root = Path(os.path.commonpath(files if len(files) > 1 else [*files, ROOT_DIR]))
    for file_path, file_functions in by_file_dict.items():
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
                current = current[part]["_children"]
    return tree


def get_all_functions_in_tree(tree_dict: dict[str, Any]) -> list:
    """Get all function lists from a tree structure."""
    all_functions = []
    for data in tree_dict.values():
        if data.get("_functions"):
            all_functions.append(data["_functions"])
        if data.get("_children"):
            all_functions.extend(get_all_functions_in_tree(data["_children"]))
    return all_functions


def render_tree(
    tree_dict: dict[str, Any], prefix: str = "", max_func_name_length: int = 50, threshold: float = 0.1
) -> list[str]:
    """Render the module tree with proper indentation."""
    lines = []
    items = list(tree_dict.items())

    if len(items) > 1:
        items = sorted(
            items,
            key=lambda item: sum(
                f.total_percentage for file_funcs in get_all_functions_in_tree(item[1].get("_children", {})) for f in file_funcs
            ),
            reverse=True,
        )

    ang, tee, bar, blk = "└─ ", "├─ ", "│  ", "   "
    for i, (name, data) in enumerate(items):
        is_last_item = i == len(items) - 1
        current_prefix = prefix + (ang if is_last_item else tee)
        next_prefix = prefix + (blk if is_last_item else bar)

        functions = data.get("_functions", [])
        children = data.get("_children", {})

        if functions:
            total_runtime = sum(f.total_percentage for f in functions)
            file_display = f"{name} ({len(functions)} func, {total_runtime:.2f}% total)"
            lines.append(f"{current_prefix}{file_display}")

            funcs = [
                fn
                for fn in sorted(functions, key=lambda f: f.total_percentage, reverse=True)
                if fn.total_percentage >= threshold
            ]
            for j, func in enumerate(funcs):
                func_is_last = j == len(funcs) - 1
                func_prefix = next_prefix + (ang if func_is_last else tee)
                peak_mem = f" ({func.peak_memory_info})" if func.has_memory_info else ""
                runtime_info = f"{func.total_percentage:.>5.2f}%{peak_mem}"
                lines.append(f"{func_prefix}{func.name:.<{max_func_name_length - len(func_prefix)}}{runtime_info}")
            lines.append(next_prefix)
        elif children:
            total_runtime = sum(f.total_percentage for file_funcs in get_all_functions_in_tree(children) for f in file_funcs)
            function_count = sum(len(file_funcs) for file_funcs in get_all_functions_in_tree(children))
            dir_display = f"{name} ({function_count} func, {total_runtime:.2f}% total)"
            lines.append(f"{current_prefix}{dir_display}")
            lines.extend(render_tree(children, next_prefix, threshold=threshold))

    return lines


def installed_modules() -> set[str]:
    """List of all installed module names in the current Python virtual environment."""
    try:
        return {str(dist.metadata["Name"]).lower() for dist in importlib.metadata.distributions()}
    except KeyError:
        Colour.print(f"Python version: {sys.version}")
        return_set = set()
        for dist in importlib.metadata.distributions():
            if "Name" not in dist.metadata:
                Colour.print("Found a distribution with missing 'Name' metadata.")
                Colour.print(f"  Path hint: {dist.locate_file('')}")
                Colour.print(f"  Available metadata: {list(dist.metadata.keys())}")
            else:
                return_set.add(str(dist.metadata["Name"]).lower())
        return return_set


def generate_sparklines(
    samples: list[tuple[float, float]], start_time: float, elapsed_time_sec: float, max_memory_usage: float, num_bars: int = 30
) -> str:
    """Generate a sparkline from memory samples.

    Args:
        samples: List of [perf_timestamp, memory_mb] pairs.
        start_time: The perf_timestamp start of the sampling data.
        elapsed_time_sec: The total runtime of the profile.
        max_memory_usage: The maximum memory used during profiling.
        num_bars: Number of characters in the sparkline.

    Returns:
        A sparkline string using Unicode block characters.

    """
    if not samples or num_bars < 1:
        return ""

    # Re-sample the data to fit the number of bars.
    bin_width_ns = max(elapsed_time_sec * 1e9, *(s[0] for s in samples)) / (num_bars)
    bin_edges = [i * bin_width_ns for i in range(num_bars + 1)]
    bin_values: list[list[float]] = [[] for _ in range(num_bars)]
    for time, mem_val in samples:
        for i in range(num_bars):
            if bin_edges[i] <= (time - start_time) < bin_edges[i + 1]:
                bin_values[i].append(mem_val)
                break

    # Average the samples in each bin.
    memory_values: list[float] = [0.0 for _ in range(num_bars)]
    for i in range(num_bars):
        memory_values[i] = sum(bin_values[i]) / n if (n := len(bin_values[i])) else 0

    min_mem = min(memory_values)
    mem_range = max_memory_usage - min_mem

    sparkline = []
    for value in memory_values:
        if mem_range == 0:
            idx = 0
        else:
            # Normalize to 0-1 range, then map to character index
            normalized = (value - min_mem) / mem_range
            idx = min(int(normalized * (len(__sparkline_bars) - 1)), len(__sparkline_bars) - 1)
        sparkline.append(__sparkline_bars[idx])

    return "".join(sparkline)


def _top_memory_summary(profile_data: ProfileData) -> str:
    """Generate a summary of the memory usage during the profiling."""
    max_memory_usage = profile_data.details.get("max_footprint_mb", 0)
    growth_rate = profile_data.details.get("growth_rate", 0)
    start_time = profile_data.details.get("start_time_perf", 0)
    elapsed_time_sec = profile_data.walltime
    sparkline = generate_sparklines(profile_data.samples, start_time, elapsed_time_sec, max_memory_usage)
    mem_units = memory_with_units(max_memory_usage, unit="MB", digits=3)
    return f"\n{'':8}Memory usage: {sparkline} (max: {mem_units}, growth rate: {growth_rate:3.1f}%)"


def _memory_summary(details: list[ProfileDetails]) -> list[str]:
    """Add top memory consumption summary."""
    lines: list[str] = ["Top PEAK memory consumption, by line:"]
    for i, line in enumerate(details, 1):
        lines.append(f"({i}) {line.line_number:>5}: {int(line.peak_memory):>5} MB")
    return lines


def initialize_table(profile_data: ProfileData, file_path: str, title: str, width: int = 128) -> Table:
    """Initialize a Rich Table for a file's profile from the data."""
    display_path = str(file_path)
    with contextlib.suppress(ValueError):
        display_path = str(Path(file_path).relative_to(ROOT_DIR))

    en6 = __table_bar["en_dash"] * 6
    en11 = (en6 * 2)[:-1]

    min_width = 85 if profile_data.has_memory_info else 50
    kwargs = {"justify": "right", "no_wrap": True, "min_width": 7, "max_width": 7}
    tbl = Table(box=box.MINIMAL_HEAVY_HEAD, title=title, collapse_padding=True, title_justify="left", min_width=min_width)
    tbl.add_column(Markdown("Line", style="dim"), style="dim", **(kwargs | {"min_width": 4, "max_width": 4}))
    tbl.add_column(Markdown("Time  " + "\n" + "_Python_"), **kwargs)
    tbl.add_column(Markdown(f"{en6}  \n_native_"), **kwargs)
    tbl.add_column(Markdown(f"{en6}  \n_system_"), **kwargs)

    if profile_data.has_memory_info:
        tbl.add_column(Markdown("Memory  \n_Python_"), **kwargs)
        tbl.add_column(Markdown(f"{en6}  \n_peak_"), **kwargs)
        if (sparks := (width - min_width)) > 0:
            spark_size = min(sparks, 14)
            tbl.add_column(Markdown(f"{en11}  \n_timeline_%"), **(kwargs | {"min_width": spark_size, "max_width": spark_size}))
        tbl.add_column(Markdown("Copy  \n_(MB/s)_"), **kwargs)

    tbl.add_column("\n " + display_path, min_width=None, no_wrap=True)
    return tbl


def _file_title(
    file_path: str,
    total_time_pct: float,
    total_time_sec: float,
    walltime: float,
) -> str:
    """Add file header section to lines."""
    # Format time with milliseconds if < 1 second
    time_str = f"{total_time_sec * 1000:.3f}ms" if total_time_sec < 1.0 else f"{total_time_sec:.3f}s"
    return f"{'':3}{file_path}: % of time = {total_time_pct:6.2f}% ({time_str}) out of {walltime:.3f}s."


def _above_threshold(details: ProfileDetails, threshold: float = 0.1) -> bool:
    """Determine if ProfileDetails are all above the threshold."""
    return (
        details.python_percentage >= threshold
        or details.native_percentage >= threshold
        or details.system_percentage >= threshold
        or details.memory_python_percentage >= threshold
    )


def _row_details(
    details: ProfileDetails,
    start_time: float,
    elapsed_time_sec: float,
    max_memory_usage: float,
    profile_has_memory_info: bool,
    width: int = 128,
    threshold: float = 0.1,
) -> list[str]:
    """Organize a single line for the table."""
    ret = [
        f"{details.line_number:4.0f}",
        f"{details.python_percentage:5.1f}%" if details.python_percentage >= threshold else "",
        f"{details.native_percentage:5.1f}%" if details.native_percentage >= threshold else "",
        f"{details.system_percentage:5.1f}%" if details.system_percentage >= threshold else "",
    ]
    if profile_has_memory_info:
        min_width = 85
        digits = 1 if details.memory_python_percentage < 100 else 0  # noqa: PLR2004
        python_percentage = f"{pyper:6.{digits}f}%" if (pyper := details.memory_python_percentage) >= threshold else ""
        heap_mem = details.peak_memory_info if details.peak_memory >= threshold else ""
        copy_velocity = f"{cp_vel:6.0f}" if (cp_vel := details.copy_mb_per_s) >= threshold else ""
        if (width - min_width) > 0:
            sparks = generate_sparklines(
                details.memory_samples, start_time, elapsed_time_sec, max_memory_usage, num_bars=min(8, int(width // 12))
            )
            sparkline = f"{sparks} {tlper:4.1f}%" if (tlper := details.timeline_percentage) >= threshold else ""
            ret.extend([python_percentage, heap_mem, sparkline, copy_velocity])
        else:
            ret.extend([python_percentage, heap_mem, copy_velocity])
    ret.append(details.name)
    return ret


def _not_empty_line(chars: set[str]) -> bool:
    """Determine if a line is 'empty'.

    It does this by checking if any of the characters are in the "printable" set,
    or if it contains any "horizontal" dividers.
    """
    return (
        bool((chars - {" "}) & __printable) or (__table_bar["dbl_horizontal"] in chars) or (__table_bar["horizontal"] in chars)
    )


def capture_table(table: Table, compact_rows: bool = True) -> str:
    """Convert a Rich table into a string."""
    console = Console()
    with console.capture() as capture:
        console.print(table)
    result = Colour.remove_ansi(capture.get())
    if compact_rows:
        result = "\n".join([line for line in result.splitlines() if _not_empty_line(set(line))])
    return result


def generate_text_report(profile_data: ProfileData, threshold: float = 0.1, width: int = 128) -> str:  # noqa: C901
    """Generate a text report from ProfileData matching the original Scalene output format.

    Args:
        profile_data: The parsed profile data.
        threshold: The cut-off value for displaying data.
        width: specify the width of the table.

    Returns:
        A formatted text report string.

    """
    report = []

    # Top-level memory usage summary.
    start_time = profile_data.details.get("start_time_perf", 0)
    max_mem = profile_data.details.get("max_footprint_mb", 0)
    elpsd_time = profile_data.walltime
    if profile_data.has_memory_info:
        report.append(_top_memory_summary(profile_data))

    # Sort files by total runtime percentage (highest first)
    sorted_files = sorted(
        profile_data.files.items(),
        key=operator.itemgetter(1),
        reverse=True,
    )
    for file_path, file_percent in sorted_files:
        total_time_sec = (file_percent / 100.0) * profile_data.walltime if profile_data.walltime else 0
        walltime = profile_data.walltime or 0

        max_path_length = width - (57 if profile_data.has_memory_info else 22)
        display_path = str(file_path)
        with contextlib.suppress(ValueError):
            display_path = str(Path(file_path).relative_to(ROOT_DIR))
        display_path = display_path if len(display_path) <= max_path_length else display_path[: max_path_length - 3] + "…"
        title = _file_title(file_path, file_percent, total_time_sec, walltime)

        if not any(_above_threshold(line, threshold) for line in profile_data.lines[file_path]):
            report.extend([title + "\n"])
            continue

        table = initialize_table(profile_data, file_path, title, width)
        n_col = len(table.columns) - 1
        prev_lineno = -1
        file_has_memory_info = False
        for line in profile_data.lines[file_path]:
            file_has_memory_info = file_has_memory_info or line.has_memory_info
            if _above_threshold(line, threshold):
                if line.line_number - 1 != prev_lineno:
                    table.add_row("...")
                table.add_row(
                    *_row_details(line, start_time, elpsd_time, max_mem, profile_data.has_memory_info, width, threshold)
                )
                prev_lineno = line.line_number

        any_above_threshold = any(_above_threshold(fn, threshold) for fn in profile_data.get_functions_by_file[file_path])
        if profile_data.functions and any_above_threshold:
            table.add_section()
            table.add_row(*[*([""] * n_col), f" function summary for {display_path}"])
            for func in profile_data.get_functions_by_file[file_path]:
                if _above_threshold(func, threshold):
                    table.add_row(
                        *_row_details(func, start_time, elpsd_time, max_mem, profile_data.has_memory_info, width, threshold)
                    )
        report.append(capture_table(table))

        if file_has_memory_info:
            memory_lines = get_top_usage(profile_data.lines[file_path], n=5, key=lambda f: f.peak_memory)
            report.extend(_memory_summary(memory_lines))
        report.append("")

    return "\n".join(report)
