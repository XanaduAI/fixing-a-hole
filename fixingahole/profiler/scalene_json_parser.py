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
"""Scalene JSON Results Parser.

This module parses the details from the Scalene JSON data into frozen dataclasses for easier manipulation, but no mutation.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fixingahole.profiler.utils import memory_with_units


def _freeze(value: list | tuple | Mapping) -> tuple | MappingProxyType:
    """Recursively freeze lists as tuples and mappings as read-only proxies."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


@dataclass(frozen=True)
class ProfileDetails:
    """Represents a function or line's profiling information."""

    file_path: str
    walltime: float
    cpu_samples_list: list[float]
    line: str
    lineno: int
    memory_samples: list[float]
    n_avg_mb: float
    n_copy_mb_s: float
    n_core_utilization: float
    n_cpu_percent_c: float
    n_cpu_percent_python: float
    n_gpu_avg_memory_mb: float
    n_gpu_peak_memory_mb: float
    n_gpu_percent: int
    n_growth_mb: float
    n_malloc_mb: float
    n_mallocs: int
    n_peak_mb: float
    n_python_fraction: int
    n_sys_percent: float
    n_usage_fraction: float
    end_function_line: int | None = None
    end_outermost_loop: int | None = None
    end_region_line: int | None = None
    start_function_line: int | None = None
    start_outermost_loop: int | None = None
    start_region_line: int | None = None

    def __post_init__(self) -> None:
        """Freeze mutable sequence fields."""
        object.__setattr__(self, "cpu_samples_list", _freeze(self.cpu_samples_list))
        object.__setattr__(self, "memory_samples", _freeze(self.memory_samples))

    @property
    def name(self) -> str:
        """Interpret the line as the name of a function."""
        return self.line

    @property
    def memory_python_percentage(self) -> float:
        """The percentage of maximum heap memory that was used by Python for a function or line."""
        return self.n_python_fraction * 100

    @property
    def copy_mb_per_s(self) -> float:
        """The memory copy rate, as measured by the heap memory for a function or line."""
        return self.n_copy_mb_s

    @property
    def line_number(self) -> int:
        """The line number of the function or line."""
        return self.lineno

    @property
    def peak_memory(self) -> float:
        """The maximum heap memory usage of the function or line."""
        return self.n_peak_mb

    @property
    def python_percentage(self) -> float:
        """The percentage of the total CPU profiling time taken by Python for this function or line."""
        return self.n_cpu_percent_python

    @property
    def native_percentage(self) -> float:
        """The percentage of the total CPU profiling time taken by native libraries for this function or line."""
        return self.n_cpu_percent_c

    @property
    def system_percentage(self) -> float:
        """The percentage of the total CPU profiling time taken by the system for this function or line."""
        return self.n_sys_percent

    @property
    def timeline_percentage(self) -> float:
        """The proportion of total memory allocations that was used by this function or line."""
        return self.n_usage_fraction * 100

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

    @cached_property
    def total_time(self) -> float:
        """Return the absolute value of the amount of runtime this function took (in seconds)."""
        return self.total_percentage * self.walltime / 100.0

    @cached_property
    def user_time(self) -> float:
        """Return the absolute value of the amount of user runtime this function took (in seconds)."""
        return (self.python_percentage + self.native_percentage) * self.walltime / 100.0

    @cached_property
    def system_time(self) -> float:
        """Return the absolute value of the amount of system runtime this function took (in seconds)."""
        return self.system_percentage * self.walltime / 100.0


@dataclass(frozen=True)
class FileDetails:
    """Represents a file's profiling information."""

    file_path: str
    walltime: float
    imports: list[str]
    percent_cpu_time: float
    functions: list[ProfileDetails]
    lines: list[ProfileDetails]
    leaks: dict[str, dict[str, float]]

    def __post_init__(self) -> None:
        """Freeze mutable nested fields."""
        meta_details: dict[str, Any] = {"file_path": self.file_path, "walltime": self.walltime}
        object.__setattr__(
            self,
            "functions",
            tuple(ProfileDetails(**(func_details | meta_details)) for func_details in self.functions),  # ty:ignore[unsupported-operator]
        )
        object.__setattr__(
            self,
            "lines",
            tuple(ProfileDetails(**(line_details | meta_details)) for line_details in self.lines),  # ty:ignore[unsupported-operator]
        )
        object.__setattr__(self, "imports", _freeze(self.imports))
        object.__setattr__(self, "leaks", _freeze(self.leaks))


@dataclass(frozen=True)
class ProfileData:
    """The Scalene profile data."""

    alloc_samples: int
    args: list[str]
    elapsed_time_sec: float
    entrypoint_dir: str
    filename: str
    files: dict[str, FileDetails]
    gpu: bool
    gpu_device: str
    growth_rate: float
    max_footprint_fname: str | None
    max_footprint_lineno: int | None
    max_footprint_mb: int
    max_footprint_python_fraction: int
    memory: bool
    program: str
    samples: list[tuple[int, float]]
    stacks: list[tuple[list[str], dict[str, float]]]
    start_time_absolute: float
    start_time_perf: float

    def __post_init__(self) -> None:
        """Normalize and freeze mutable nested profile data."""
        files: dict[str, FileDetails] = {}
        for file_path, value in self.files.items():
            meta_details: dict[str, Any] = {"file_path": file_path, "walltime": self.elapsed_time_sec}
            files[file_path] = FileDetails(**(value | meta_details))  # ty:ignore[unsupported-operator]

        object.__setattr__(self, "args", _freeze(self.args))
        object.__setattr__(self, "samples", _freeze(self.samples))
        object.__setattr__(self, "stacks", _freeze(self.stacks))
        object.__setattr__(self, "files", MappingProxyType(files))

    @classmethod
    def from_file(cls, filename: Path) -> "ProfileData":
        """Create ProfileData from raw Scalene JSON data."""
        return cls(**json.loads(Path(filename).read_text(encoding="utf-8")))

    @property
    def walltime(self) -> float:
        """The total elapsed time in seconds of the profiling."""
        return self.elapsed_time_sec

    @property
    def max_memory(self) -> str:
        """The maximum footprint of the heap memory used while profiling."""
        return memory_with_units(self.max_footprint_mb, digits=3)

    @cached_property
    def has_memory_info(self) -> bool:
        """Determine if any memory usage data is stored."""
        return any(
            any(func.has_memory_info for func in data.functions) or any(line.has_memory_info for line in data.lines)
            for data in self.files.values()
        )

    @cached_property
    def lines(self) -> dict[str, list[ProfileDetails]]:
        """All the lines, grouped by file path."""
        lines: dict[str, list[ProfileDetails]] = {}
        for file, data in self.files.items():
            lines[file] = data.lines
        return lines

    @cached_property
    def functions(self) -> list[ProfileDetails]:
        """All of the functions."""
        functions: list[ProfileDetails] = []
        for data in self.files.values():
            functions.extend(data.functions)
        return functions

    @cached_property
    def functions_by_file(self) -> dict[str, list[ProfileDetails]]:
        """All the functions, grouped by file path."""
        result: dict[str, list[ProfileDetails]] = {}
        for file, data in self.files.items():
            result[file] = data.functions
        return result
