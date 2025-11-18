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

import importlib
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fixingahole import ROOT_DIR


class FunctionProfile:
    """Represents a function's profiling information."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the FunctionProfile instance."""
        self.line_number = kwargs.get("line_number", 0)
        self.function_name = kwargs.get("function_name", "")
        self.file_path = kwargs.get("file_path", "")
        self.memory_python_percentage = FunctionProfile._get_as_float(kwargs.get("memory_python_percentage"))
        self.peak_memory = kwargs.get("peak_memory", "")
        self.memory_size = FunctionProfile._parse_memory_size(self.peak_memory)
        self.timeline_percentage = self._get_as_float(kwargs.get("timeline_percentage"))
        self.copy_mb_per_s = self._get_as_float(kwargs.get("copy_mb_per_s"))
        self.total_percentage = sum(self._get_as_float(value) for value in kwargs.get("cpu_percentages", []))
        self.has_memory_info = bool(
            self.peak_memory or self.memory_python_percentage > 0 or self.timeline_percentage > 0,
        )

    @property
    def peak_memory_info(self) -> str:
        """Format the peak memory information, if available."""
        if self.peak_memory:
            value, unit = self.peak_memory[:-1], self.peak_memory[-1]
            return f"{value} {unit}B"
        return ""

    @staticmethod
    def _get_as_float(value_str: str) -> float:
        """Parse a numeric value from a string, discarding units."""
        try:
            return float("".join(c for c in value_str if c.isdigit() or c in ".-"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_memory_size(memory_str: str) -> int:
        """Parse memory size string to bytes for sorting (e.g., '13.98G' -> bytes)."""
        value = FunctionProfile._get_as_float(memory_str)
        units = "".join(c for c in memory_str if not c.isdigit() and c not in ".-").upper()

        # Convert to bytes
        kilobytes = 1024
        multipliers = {
            "": 1,
            "B": 1,
            "K": kilobytes,
            "KB": kilobytes,
            "M": kilobytes**2,
            "MB": kilobytes**2,
            "G": kilobytes**3,
            "GB": kilobytes**3,
        }

        return int(value * multipliers.get(units, 1))


class ProfileParser:
    """Parser for summarizing scalene cli profile results files."""

    def __init__(self, filename: str | Path | None = None) -> None:
        """Parser for summarizing Scalene cli profile results files."""
        self.functions: list[FunctionProfile] = None
        self.walltime: float | None = None
        self.max_memory: str | None = None
        if filename is not None and Path(filename).exists():
            self.functions = self.parse_file(filename)

    def parse_file(self, file_path: str) -> list[FunctionProfile]:
        """Parse a profile results file and return function profiles."""
        return self.parse_content(Path(file_path).read_text(encoding="utf-8"))

    def parse_content(self, content: str) -> list[FunctionProfile]:
        """Parse profile results content and return function profiles."""
        # Replace "table vertical bar" (│, U+2502) and "box drawings light up" (╵, U+2575)
        #  with the "verical bar" (|, U+007C) for parsing.
        # The "vertical bar" should be the easy one to type with a normal keyboard.
        self.get_details_from_profile(content)

        lines = content.replace(chr(0x2502), chr(0x007C)).replace(chr(0x2575), chr(0x007C)).split("\n")
        functions = []
        current_file = None
        in_function_summary = False

        for line in lines:
            # Check for file headers
            if r"% of time" in line:
                current_file = line.strip().split(" ")[0]
                in_function_summary = False
                continue

            # Check for function summary section
            if "function summary for" in line:
                in_function_summary = True
                continue

            # Check for end of function summary (empty line or table border)
            if in_function_summary and not line.strip(" |+-"):
                in_function_summary = False
                continue

            # Parse function lines in summary sections
            if in_function_summary and current_file:
                # The initial groups will only be empty if the line begins/ends with "|"
                groups = [group.strip() for group in line.split("|") if group]
                kwargs = {}
                kwargs["line_number"] = int(groups[0])
                kwargs["cpu_percentages"] = groups[1:4]
                kwargs["function_name"] = groups[-1]
                kwargs["file_path"] = current_file

                # Parse memory information from any additional columns
                remaining_cols = 4
                if len(memory_group := groups[4:]) >= remaining_cols:
                    # Memory Python, Peak, Timeline, and Copy
                    kwargs["memory_python_percentage"] = memory_group[0]
                    kwargs["peak_memory"] = memory_group[1]
                    kwargs["timeline_percentage"] = memory_group[2]
                    kwargs["copy_mb_per_s"] = memory_group[3]

                functions.append(FunctionProfile(**kwargs))

        return functions

    def get_details_from_profile(self, contents: str) -> None:
        """Given a profile output from scalene, return the walltime or memory usage."""
        if self.walltime is None and contents:
            time = re.search(r"out of ((\d+h:)?(\d+m:)?\d+\.\d+)(m)?s", contents)
            if time is not None:
                time_groups = time.groups()
                sec_or_ms = 1 if time_groups[3] is None else 1e-3
                time_string = time_groups[0].replace("h", "").replace("m", "").split(":")
                time_string.reverse()
                units = [sec_or_ms, 60, 3600]
                self.walltime = float(
                    sum(t * u for t, u in zip(map(float, time_string), units, strict=False)),
                )
        if self.max_memory is None:
            max_memory = re.search(r"max:\s(\d+.\d+\s.B)", contents)
            self.max_memory = str(max_memory[1]) if max_memory is not None else "an unknown amount"

    def summary(self, functions: list[FunctionProfile] | None = None, top_n: int = 10) -> str:
        """Generate a summary of the profiling results."""
        # Functions by file and calculate the maximum function name length for proper alignment
        # Also check if any functions have memory information
        functions = self.functions if functions is None else functions
        if functions is None:
            err_msg = "Missing functions to summarize."
            raise ValueError(err_msg)
        by_file = ProfileParser.get_functions_by_file(functions)
        has_memory_info = False
        max_func_name_length = 0
        for file_functions in by_file.values():
            for func in file_functions:
                max_func_name_length = max(
                    len(func.function_name) + 3,
                    max_func_name_length,
                )
                has_memory_info += func.has_memory_info
        width = max_func_name_length + 40
        message = [f"\nProfile Summary ({self.walltime or 0:,.3f}s total)", "=" * width]

        # Top functions by total runtime percentage
        top_functions = ProfileParser.get_top_functions(functions, top_n)
        message += [
            f"\nTop {len(top_functions)} Functions by Total Runtime:",
            "-" * width,
        ]
        for i, func in enumerate(top_functions, 1):
            file_name = func.file_path.split("/")[-1] if "/" in func.file_path else func.file_path
            runtime_info = f"{func.total_percentage:>5.1f}%"
            message.append(
                f"{i:2d}. {func.function_name:<{max_func_name_length}} {runtime_info:<6} ({file_name})",
            )

        # Add memory summary if available
        if has_memory_info:
            # Get top functions by peak memory (convert to bytes for proper sorting)
            memory_functions = ProfileParser.get_top_functions(functions, top_n, key=lambda f: f.memory_size)
            # Ensure that all functions have peak memory values.
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

        # Build and render the module tree
        module_tree = ProfileParser.build_module_tree(by_file)
        tree = ProfileParser.render_tree(module_tree, max_func_name_length=max_func_name_length)
        message.extend(tree)
        message.append("")

        message.extend(["=" * width, "\n"])
        # Remove trailing whitespace.
        message = [line.rstrip() for line in message]
        return "\n".join(message)

    @staticmethod
    def get_top_functions(
        functions: list[FunctionProfile],
        n: int = 10,
        key: Callable = lambda f: f.total_percentage,
    ) -> list[FunctionProfile]:
        """Get the top N functions by total runtime percentage."""
        return sorted(functions, key=key, reverse=True)[:n]

    @staticmethod
    def get_functions_by_file(
        functions: list[FunctionProfile],
    ) -> dict[str, list[FunctionProfile]]:
        """Group functions by file path."""
        result = defaultdict(list)
        for func in functions:
            result[func.file_path].append(func)
        return result

    # Build module tree for hierarchical display
    @staticmethod
    def build_module_tree(by_file_dict: dict[str, list[FunctionProfile]]) -> dict[str, Any]:
        """Build a hierarchical tree structure from file paths."""
        modules = ProfileParser.installed_modules()
        tree: dict[str, Any] = {}
        for file_path, file_functions in by_file_dict.items():
            # Split the path into parts, relative to the lowest common directory.
            parts = Path(file_path).parts
            for parents in [ROOT_DIR, *ROOT_DIR.parents]:
                try:
                    parts = Path(file_path).relative_to(parents).parts
                    break
                except ValueError:
                    pass

            # Try to find the appropriate module.
            for i, part in enumerate(parts):
                if part in modules:
                    parts = parts[i:]
                    break

            # Build the tree structure
            current = tree
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {"_functions": [], "_children": {}}

                if i == len(parts) - 1:  # Last part is the file
                    current[part]["_functions"] = file_functions
                else:
                    current = current[part]["_children"]
        return tree

    @staticmethod
    def get_all_functions_in_tree(tree_dict: dict[str, Any]) -> list:
        """Get all function lists from a tree structure."""
        all_functions = []
        for data in tree_dict.values():
            if data.get("_functions"):
                all_functions.append(data["_functions"])
            if data.get("_children"):
                all_functions.extend(ProfileParser.get_all_functions_in_tree(data["_children"]))
        return all_functions

    @staticmethod
    def render_tree(tree_dict: dict[str, Any], prefix: str = "", max_func_name_length: int = 50) -> list[str]:
        """Render the module tree with proper indentation."""
        lines = []
        items = list(tree_dict.items())

        # Sort the files and modules by total runtime.
        if len(items) > 1:
            items = sorted(
                items,
                key=lambda item: sum(
                    f.total_percentage
                    for file_funcs in ProfileParser.get_all_functions_in_tree(item[1].get("_children", {}))
                    for f in file_funcs
                ),
                reverse=True,
            )

        ang, tee, bar, blk = "└─ ", "├─ ", "│  ", "   "
        for i, (name, data) in enumerate(items):
            is_last_item = i == len(items) - 1

            # Determine the tree characters
            current_prefix = prefix + (ang if is_last_item else tee)
            next_prefix = prefix + (blk if is_last_item else bar)

            # Check if this is a file (has functions) or is a directory
            functions = data.get("_functions", [])
            children = data.get("_children", {})

            if functions:
                # This is a file with functions
                total_runtime = sum(f.total_percentage for f in functions)
                file_display = f"{name} ({len(functions)} func, {total_runtime:.1f}% total)"
                lines.append(f"{current_prefix}{file_display}")

                # Add functions under this file
                for j, func in enumerate(
                    sorted(functions, key=lambda f: f.total_percentage, reverse=True),
                ):
                    func_is_last = j == len(functions) - 1
                    func_prefix = next_prefix + (ang if func_is_last else tee)
                    peak_mem = f" ({func.peak_memory_info})" if func.has_memory_info else ""
                    runtime_info = f"{func.total_percentage:.>5.1f}%{peak_mem}"
                    lines.append(
                        f"{func_prefix}{func.function_name:.<{max_func_name_length}}{runtime_info}",
                    )
                lines.append(next_prefix)
            elif children:
                # This is a directory with children
                total_runtime = sum(
                    f.total_percentage for file_funcs in ProfileParser.get_all_functions_in_tree(children) for f in file_funcs
                )
                function_count = sum(len(file_funcs) for file_funcs in ProfileParser.get_all_functions_in_tree(children))
                dir_display = f"{name} ({function_count} func, {total_runtime:.1f}% total)"
                lines.append(f"{current_prefix}{dir_display}")

                # Recursively render children
                lines.extend(ProfileParser.render_tree(children, next_prefix))

        return lines

    @staticmethod
    def installed_modules() -> set[str]:
        """List of all installed module names in the current Python virtual environment."""
        return {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
