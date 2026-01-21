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
"""Profile Results Parser.

This module parses profile results files and extracts function names and runtime percentages from
the function summary sections at the bottom of each file's profiling table.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from colours import Colour
from typer import Exit

from fixingahole import ROOT_DIR
from fixingahole.profiler.utils import installed_modules, memory_with_units


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

    @classmethod
    def from_scalene_dict(cls, data: dict[str, Any], file_path: str) -> "ProfileDetails":
        """Create ProfileDetails from raw scalene JSON data.

        Args:
            data: Dictionary containing scalene profile data for a line or function
            file_path: The file path to associate with this profile entry

        Returns:
            ProfileDetails instance with properly typed and mapped fields.

        """
        return cls(
            file_path=file_path,
            name=data.get("line", ""),
            line_number=int(data.get("lineno", 0)),
            python_percentage=float(data.get("n_cpu_percent_python", 0)),
            native_percentage=float(data.get("n_cpu_percent_c", 0)),
            system_percentage=float(data.get("n_sys_percent", 0)),
            peak_memory=float(data.get("n_peak_mb", 0)),
            copy_mb_per_s=float(data.get("n_copy_mb_s", 0)),
            memory_samples=list(data.get("memory_samples", [])),
            memory_python_percentage=float(data.get("n_python_fraction", 0)) * 100,
            timeline_percentage=float(data.get("n_usage_fraction", 0)) * 100,
        )

    @cached_property
    def has_memory_info(self) -> bool:
        """Determine if a line used significant memory."""
        return bool(
            self.peak_memory > 0 or self.memory_python_percentage > 0 or self.timeline_percentage > 0 or self.memory_samples
        )

    @cached_property
    def has_data(self) -> bool:
        """Determine if this profile entry contains any meaningful data."""
        return bool(self.total_percentage > 0 or self.has_memory_info or self.copy_mb_per_s > 0)

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
    walltime: float | None
    max_memory: str | None
    samples: list[tuple[float, float]]
    details: dict[str, float]

    @cached_property
    def has_memory_info(self) -> bool:
        """Determine any memory usage data is stored."""
        return any(fn.has_memory_info for fn in self.functions) or any(
            ln.has_memory_info for lines in self.lines.values() for ln in lines
        )

    @cached_property
    def functions_by_file(self) -> dict[str, list[ProfileDetails]]:
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
        raise Exit(code=66)  # Cannot open input: A specified file or input cannot be accessed.

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
            profile = ProfileDetails.from_scalene_dict(line, file_path)
            if profile.has_data:
                line_profs[file_path].append(profile)

        funcs = info.get("functions", []) if isinstance(info, dict) else []
        function_profs.extend(ProfileDetails.from_scalene_dict(fn, file_path) for fn in funcs)

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
    by_file = profile_data.functions_by_file
    for file_functions in by_file.values():
        for func in file_functions:
            max_func_name_length = max(
                len(func.name) + 3,
                max_func_name_length,
            )
            has_memory_info = has_memory_info or func.has_memory_info

    width = max_func_name_length + 40
    message = ["\nProfile Summary", "=" * width]

    # Top functions by total runtime percentage
    top_functions: list[ProfileDetails] = sorted(functions, key=lambda f: f.total_percentage, reverse=True)[:top_n]
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
        memory_functions: list[ProfileDetails] = sorted(functions, key=lambda f: f.peak_memory, reverse=True)[:top_n]
        memory_functions: list[ProfileDetails] = [f for f in memory_functions if f.peak_memory]
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


class ProfileSummary:
    """Parser for summarizing scalene cli profile results files."""

    def __init__(self, filename: str | Path):
        self.data = parse_json(filename)
        self.walltime: float | None = self.data.walltime
        self.max_memory: str | None = self.data.max_memory

    def summary(self, top_n: int = 10, threshold: float = 0.1) -> str:
        """Generate a summary of the profiling results."""
        return generate_summary(self.data, top_n, threshold)
