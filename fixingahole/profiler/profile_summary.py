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
from pathlib import Path
from typing import Any

from fixingahole import Config
from fixingahole.profiler.scalene_json_parser import ProfileData, ProfileDetails
from fixingahole.profiler.utils import format_time, installed_modules


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

    # Top functions by total runtime percentage
    top_functions: list[ProfileDetails] = sorted(functions, key=lambda f: f.total_percentage, reverse=True)[:top_n]
    top_functions = [fn for fn in top_functions if fn.total_percentage >= threshold]
    n: int = len(top_functions)
    if n == 0:
        message += [
            "\nNo functions to summarize by Total Runtime",
            "-" * width,
        ]
    else:
        message += [
            f"\nTop {f'{n} Functions' if n > 1 else 'Function'} by Total Runtime:",
            "-" * width,
        ]
    for i, func in enumerate(top_functions, 1):
        file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
        lineno = func.line_number
        runtime_info = (
            f"{func.total_percentage:>5.2f}%"
            if Config.is_duration_relative()
            else format_time(func.total_percentage * profile_data.walltime / 100, profile_data.walltime)
        )
        message.append(
            f"{i:2d}. {func.name:<{max_func_name_length}} {runtime_info:<{runtime_width}} ({file_name}:{lineno})",
        )

    # Add memory summary if available
    if has_memory_info:
        memory_functions: list[ProfileDetails] = sorted(functions, key=lambda f: f.peak_memory, reverse=True)[:top_n]
        memory_functions: list[ProfileDetails] = [f for f in memory_functions if f.peak_memory]
        n: int = len(memory_functions)
        if memory_functions:
            message += [
                f"\nTop {f'{n} Functions' if n > 1 else 'Function'} by Memory Usage:",
                "-" * width,
            ]
            for i, func in enumerate(memory_functions, 1):
                file_name: str = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
                lineno = func.line_number
                message.append(
                    f"{i:2d}. {func.name:<{max_func_name_length}} {func.peak_memory_info:>{mem_width}} ({file_name}:{lineno})",
                )

    message.append("\nFunctions by Module:")
    message.append("-" * width)
    module_tree, depth = build_module_tree(by_file, threshold=threshold)
    tree_width = max_func_name_length + (depth + 2) * 3  # depth + 2 because the minimum tree depth is 2.
    tree = render_tree(module_tree, profile_data.walltime, max_func_name_length=tree_width, threshold=threshold)
    message.extend(tree)
    message.append("")

    message.extend(["=" * width, "\n"])
    return "\n".join(line.rstrip() for line in message)


def build_module_tree(by_file_dict: dict[str, list[ProfileDetails]], threshold: float = 0.1) -> tuple[dict[str, Any], int]:
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


def render_tree(
    tree_dict: dict[str, Any],
    walltime: float,
    prefix: str = "",
    max_func_name_length: int = 50,
    threshold: float = 0.1,
) -> list[str]:
    """Render the module tree with proper indentation."""
    lines = []
    items = list(tree_dict.items())

    if len(items) > 1:
        # Sort the tree from largest to smallest runtime fraction.
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
        is_last_item = i == len(items) - 1
        current_prefix = prefix + (ang if is_last_item else tee)
        next_prefix = prefix + (blk if is_last_item else bar)

        functions: list[ProfileDetails] = sorted(data.get("_functions", []), key=lambda f: f.total_percentage, reverse=True)
        functions = [f for f in functions if f.total_percentage >= threshold or f.has_memory_info]
        children: dict[str, Any] = data.get("_children", {})

        if functions:
            total_runtime = sum(f.total_percentage for f in functions)
            dur = (
                f"{total_runtime:.2f}% total"
                if Config.is_duration_relative()
                else format_time(total_runtime * walltime / 100, walltime)
            )
            file_display = f"{name} ({len(functions)} func, {dur})"
            lines.append(f"{current_prefix}{file_display}")

            funcs = [
                fn
                for fn in sorted(functions, key=lambda f: f.total_percentage, reverse=True)
                if fn.total_percentage >= threshold or fn.has_memory_info
            ]
            for j, func in enumerate(funcs):
                func_is_last = j == len(funcs) - 1
                func_prefix = next_prefix + (ang if func_is_last else tee)
                peak_mem = f" ({func.peak_memory_info})" if func.has_memory_info else ""
                runtime_info = (
                    f"{func.total_percentage:.>5.2f}%"
                    if Config.is_duration_relative()
                    else format_time(func.total_percentage * walltime / 100, walltime)
                ) + f"{peak_mem}"
                lines.append(f"{func_prefix}{func.name:.<{max(max_func_name_length - len(func_prefix), 2)}}{runtime_info}")
            lines.append(next_prefix)
        elif children:
            total_runtime = 0
            function_count = 0
            has_memory_info = False
            for file_funcs in get_all_functions_in_tree(children):
                for f in file_funcs:
                    total_runtime += f.total_percentage
                    has_memory_info: bool = has_memory_info or f.has_memory_info
                    function_count += 1 if f.total_percentage >= threshold or f.has_memory_info else 0
            if total_runtime < threshold and not has_memory_info:
                return lines
            dur = (
                f"{total_runtime:.2f}% total"
                if Config.is_duration_relative()
                else format_time(total_runtime * walltime / 100, walltime)
            )
            dir_display = f"{name} ({function_count} func, {dur})"
            lines.append(f"{current_prefix}{dir_display}")
            lines.extend(
                render_tree(
                    children,
                    walltime=walltime,
                    prefix=next_prefix,
                    max_func_name_length=max_func_name_length,
                    threshold=threshold,
                )
            )

    return lines


class ProfileSummary:
    """Parser for summarizing scalene cli profile results files."""

    def __init__(self, filename: Path):
        self.data = ProfileData.from_file(filename)
        self.walltime: float = self.data.walltime
        self.max_memory: str = self.data.max_memory

    def summary(self, top_n: int = 10, threshold: float = 0.1) -> str:
        """Generate a summary of the profiling results."""
        return generate_summary(self.data, top_n, threshold)
