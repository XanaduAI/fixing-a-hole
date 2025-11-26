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

import importlib.metadata
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colours import Colour

from fixingahole import ROOT_DIR
from fixingahole.profiler.utils import memory_with_units


@dataclass(frozen=True)
class FunctionProfile:
    """Represents a function's profiling information."""

    function_name: str
    file_path: str
    line_number: int
    memory_python_percentage: float
    peak_memory: float
    python_percentage: float
    native_percentage: float
    system_percentage: float
    timeline_percentage: float
    copy_mb_per_s: float
    has_memory_info: bool

    @property
    def peak_memory_info(self) -> str:
        """Convert the used memory into more sensible units."""
        return memory_with_units(self.peak_memory)

    @property
    def total_percentage(self) -> float:
        """Return the total percentage of the runtime."""
        return self.python_percentage + self.native_percentage + self.system_percentage


@dataclass(frozen=True)
class ProfileData:
    """Holds the parsed profile data."""

    functions: list[FunctionProfile]
    walltime: float | None
    max_memory: str | None


def create_function_profile(**kwargs: object) -> FunctionProfile:
    """Create a FunctionProfile instance from a dictionary."""
    peak_memory = float(kwargs.get("peak_memory", 0))
    memory_python_percentage = float(kwargs.get("memory_python_percentage", 0))
    timeline_percentage = float(kwargs.get("timeline_percentage", 0))

    return FunctionProfile(
        line_number=kwargs.get("line_number", 0),
        function_name=kwargs.get("function_name", ""),
        file_path=kwargs.get("file_path", ""),
        memory_python_percentage=memory_python_percentage,
        peak_memory=peak_memory,
        timeline_percentage=timeline_percentage,
        copy_mb_per_s=kwargs.get("copy_mb_per_s", 0),
        python_percentage=kwargs.get("python_percentage", 0),
        native_percentage=kwargs.get("native_percentage", 0),
        system_percentage=kwargs.get("system_percentage", 0),
        has_memory_info=bool(peak_memory > 0 or memory_python_percentage > 0 or timeline_percentage > 0),
    )


def parse_json(filename: str | Path) -> ProfileData:
    """Parse profile results provided as a JSON dictionary."""
    profile_path = Path(filename)
    if not profile_path.exists():
        return ProfileData(functions=[], walltime=None, max_memory=None)

    fixingahole_header = "### Add Fixing-A-Hole extras for profiling. ###"
    content = json.loads(profile_path.read_text(encoding="utf-8"))
    functions: list[FunctionProfile] = []

    # Extract walltime and max memory
    walltime = content.get("elapsed_time_sec", -1)
    max_memory = memory_with_units(content.get("max_footprint_mb", -1), digits=3)

    files = content.get("files", {}) if isinstance(content, dict) else {}
    for file_path, info in files.items():
        funcs = info.get("functions", []) if isinstance(info, dict) else []
        for fn in funcs:
            line_offset = 0
            try:
                if Path(file_path).exists() and fixingahole_header in Path(file_path).read_text(encoding="utf-8"):
                    line_offset = 21
            except (OSError, UnicodeDecodeError):
                # File doesn't exist, can't be read, or has encoding issues - skip header check
                pass
            kwargs = {
                "file_path": file_path,
                "line_number": fn.get("lineno", 0) - line_offset,
                "function_name": fn.get("line", "<unknown>"),
                "python_percentage": fn.get("n_cpu_percent_python", 0),
                "native_percentage": fn.get("n_cpu_percent_c", 0),
                "system_percentage": fn.get("n_sys_percent", 0),
                "peak_memory": fn.get("n_peak_mb", 0),
                "copy_mb_per_s": fn.get("n_copy_mb_s", 0),
                "memory_python_percentage": fn.get("n_python_fraction", 0) * 100,
                "timeline_percentage": fn.get("n_usage_fraction", 0) * 100,
            }
            functions.append(create_function_profile(**kwargs))

    return ProfileData(functions=functions, walltime=walltime, max_memory=max_memory)


def generate_summary(profile_data: ProfileData, top_n: int = 10, threshold: float = 0.1) -> str:
    """Generate a summary of the profiling results."""
    functions = profile_data.functions
    if not functions:
        return "No functions to summarize.\n"

    has_memory_info = False
    max_func_name_length = 0
    by_file = get_functions_by_file(functions)
    for file_functions in by_file.values():
        for func in file_functions:
            max_func_name_length = max(
                len(func.function_name) + 3,
                max_func_name_length,
            )
            has_memory_info = has_memory_info or func.has_memory_info

    width = max_func_name_length + 40
    message = [f"\nProfile Summary ({profile_data.walltime or 0:,.3f}s total)", "=" * width]

    # Top functions by total runtime percentage
    top_functions = get_top_functions(functions, top_n)
    message += [
        f"\nTop {len(top_functions)} Functions by Total Runtime:",
        "-" * width,
    ]
    for i, func in enumerate(top_functions, 1):
        file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
        lineno = func.line_number
        runtime_info = f"{func.total_percentage:>5.1f}%"
        message.append(
            f"{i:2d}. {func.function_name:<{max_func_name_length}} {runtime_info:<6} ({file_name}:{lineno})",
        )

    # Add memory summary if available
    if has_memory_info:
        memory_functions = get_top_functions(functions, top_n, key=lambda f: f.peak_memory)
        memory_functions = [f for f in memory_functions if f.peak_memory]
        if memory_functions:
            message += [
                f"\nTop {len(memory_functions)} Functions by Memory Usage:",
                "-" * width,
            ]
            for i, func in enumerate(memory_functions, 1):
                file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
                message.append(
                    f"{i:2d}. {func.function_name:<{max_func_name_length}} {func.peak_memory_info:>8} ({file_name})",
                )

    message.append("\nFunctions by Module:")
    message.append("-" * width)

    module_tree = build_module_tree(by_file)
    tree = render_tree(module_tree, max_func_name_length=max_func_name_length, threshold=threshold)
    message.extend(tree)
    message.append("")

    message.extend(["=" * width, "\n"])
    return "\n".join(line.rstrip() for line in message)


def get_top_functions(
    functions: list[FunctionProfile],
    n: int = 10,
    key: Callable = lambda f: f.total_percentage,
) -> list[FunctionProfile]:
    """Get the top N functions by key."""
    return sorted(functions, key=key, reverse=True)[:n]


def get_functions_by_file(
    functions: list[FunctionProfile],
) -> dict[str, list[FunctionProfile]]:
    """Group functions by file path."""
    result = defaultdict(list)
    for func in functions:
        result[func.file_path].append(func)
    return result


def build_module_tree(by_file_dict: dict[str, list[FunctionProfile]]) -> dict[str, Any]:
    """Build a hierarchical tree structure from file paths."""
    modules = installed_modules()
    tree: dict[str, Any] = {}
    for file_path, file_functions in by_file_dict.items():
        parts = Path(file_path).parts
        for parents in [ROOT_DIR, *ROOT_DIR.parents]:
            try:
                parts = Path(file_path).relative_to(parents).parts
                break
            except ValueError:
                pass

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
                lines.append(
                    f"{func_prefix}{func.function_name:.<{max_func_name_length}}{runtime_info}",
                )
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
