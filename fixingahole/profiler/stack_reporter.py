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
"""Stack Reporter for Profile Results.

This script reports stack traces for the most expensive function calls from a Scalene profile
results JSON file.
"""

import json
import operator
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fixingahole import ROOT_DIR


class StackReporter:
    """Reports stack traces for expensive function calls from Scalene profile results JSON file."""

    def __init__(self, profile_json_path: Path | str) -> None:
        """Report stack traces for expensive function calls."""
        try:
            self.profile_json_path = Path(profile_json_path).resolve()
            self.data = self.load_profile_results(self.profile_json_path)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            err_msg = f"Failed to initialize StackReporter: {e}"
            raise StackReporterError(err_msg) from e

    @staticmethod
    def load_profile_results(json_path: Path) -> dict:
        """Load profile results from a JSON file."""
        with Path.open(json_path) as f:
            return json.load(f)

    def get_top_functions(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the top n functions by total CPU percentage."""
        funcs = []
        for file, data in self.data["files"].items():
            for func in data["functions"]:
                total_percent = func["n_cpu_percent_c"] + func["n_cpu_percent_python"]
                funcs.append({"file": file, "name": func["line"], "total_percent": total_percent})
        # Sort by total_percent descending
        return sorted(funcs, key=operator.itemgetter("total_percent"), reverse=True)[:n]

    def find_stack_traces(self, func_name: str) -> list[dict]:
        """Find all stack traces where the function name appears."""
        return [{"stack": stack[0], **stack[1]} for stack in self.data["stacks"] if func_name in stack[0][-1]]

    @staticmethod
    def combine_stack_traces(traces: list[dict]) -> dict[str, Any]:
        """Gather traces into similar call stacks."""
        combined = defaultdict(
            lambda: {"count": 0, "c_time": 0.0, "python_time": 0.0, "cpu_samples": 0.0},
        )
        for trace in traces:
            key = tuple(trace["stack"][:-1])
            combined[key]["count"] += trace["count"]
            combined[key]["cpu_samples"] += trace["cpu_samples"]
            combined[key]["c_time"] += trace["c_time"]
            combined[key]["python_time"] += trace["python_time"]
        return combined

    def report_stacks_for_top_functions(self, top_n: int = 5) -> str:
        """Top N expensive functions, rendered as a combined reverse tree."""
        top_funcs = self.get_top_functions(n=top_n)
        report: list[str] = []
        for func in top_funcs:
            report.append(f"\n{func['name']}, ({func['total_percent']:.2f}%)")
            traces = self.find_stack_traces(func["name"])
            if not traces:
                report.append("  No stack traces found.\n")
                continue
            # Build combined tree from all stack traces
            tree, call_info = StackReporter.build_combined_reverse_tree(traces)
            tree_lines = self.render_combined_reverse_tree(tree, call_info, is_root=False)
            report.extend([f"  {line}" for line in tree_lines])

        if len(report) > 0:
            width = max(len(line) for line in report) + 5
            report = [
                f"\nStack Trace Summary ({round(self.data['elapsed_time_sec'], 3):.3f}s total)",
                "=" * width,
                *report,
            ]
        return "\n".join(report)

    @staticmethod
    def build_combined_reverse_tree(traces: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a combined reverse tree from all stack traces for a function.

        Returns (tree, call_info) where tree is a nested dict and call_info maps node paths to call
        stats. Stack frames are normalized relative to ROOT_DIR if possible.
        """
        tree = {}
        call_info = {}
        combined_traces = StackReporter.combine_stack_traces(traces)
        for stacks, values in combined_traces.items():
            stack = list(reversed(stacks))  # callers first, callee last
            norm_stack = []
            for frame in stack:
                # Try to normalize file path in frame if present
                # Assume frame format: 'filename function:line;'
                # Rearrange to: 'relative_filepath:line; function'
                file_path, func_name, line_no = re.split(r"[ :]", frame)
                try:
                    rel_path = str(Path(file_path).resolve().relative_to(ROOT_DIR))
                except ValueError:
                    rel_path = str(Path(file_path).resolve())
                norm_stack.append(f"{rel_path}:{line_no} {func_name}")
            current = tree
            path = []
            for frame in norm_stack:
                path.append(frame)
                if frame not in current:
                    current[frame] = {}
                current = current[frame]
            call_info[tuple(path)] = {
                "count": values["count"],
                "cpu_samples": values["cpu_samples"],
                "c_time": values["c_time"],
                "python_time": values["python_time"],
            }
        return tree, call_info

    def render_combined_reverse_tree(
        self,
        tree: dict[str, Any],
        call_info: dict[str, Any],
        prefix: str = "",
        path: list | None = None,
        is_root: bool = True,
    ) -> list[str]:
        """Render the combined reverse tree, merging branches for shared callers."""
        if path is None:
            path = []
        lines = []
        items = list(tree.items())
        ang, tee, bar, blk = "└─ ", "├─ ", "│  ", "   "
        for i, (frame, subtree) in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = prefix + (ang if is_last else tee if not is_root else "")
            next_prefix = prefix + (blk if is_last else bar)
            lines.append(f"{current_prefix}{frame}")
            # If this is a leaf, show call info
            new_path = [*path, frame]
            if not subtree and tuple(new_path) in call_info:
                info = call_info[tuple(new_path)]
                lines.extend([f"{next_prefix}n_calls: {info['count']}", next_prefix])
            else:
                lines.extend(
                    self.render_combined_reverse_tree(
                        subtree,
                        call_info,
                        next_prefix,
                        new_path,
                        is_root=False,
                    ),
                )
        return lines


class StackReporterError(Exception):
    """General exception for StackReporter."""
