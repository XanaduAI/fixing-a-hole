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
"""Tests for profiling parallel workloads (threading and multiprocessing)."""

import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from fixingahole.config import Config, DurationOption, Settings
from fixingahole.profiler import Profiler
from fixingahole.profiler.profiler import SuccessfulExit
from fixingahole.profiler.scalene_json_parser import ProfileData

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
WORKERS = ("matrix_worker", "monte_carlo_worker", "statistical_worker")


# Minimum CPU percentage a worker must record to confirm it was genuinely captured.
_MIN_WORKER_CPU_PCT = 1.0

# Force single-threaded BLAS in subprocesses and in-process sequential baselines so that
# the parallel runs (threads / processes) have a fair single-threaded reference to beat.
_SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _profiled_functions(profiler: Profiler) -> list[str]:
    """Return the names of functions recorded in the profile JSON for the profiled script only."""
    data = ProfileData.from_file(profiler.output_json)
    script_str = str(profiler.python_file)
    return [fn.name for fn in data.functions if fn.file_path == script_str]


def _cpu_usage(profiler: Profiler, function_name: str) -> float:
    """Return total CPU percentage for a named function in the profiled script, or 0 if absent."""
    data = ProfileData.from_file(profiler.output_json)
    script_str = str(profiler.python_file)
    matches = [fn for fn in data.functions if fn.name == function_name and fn.file_path == script_str]
    return sum(fn.total_percentage for fn in matches)


def _run_profiler(
    script_name: str,
    tmp_key: str,
    tmp_path_factory: pytest.TempPathFactory,
    serial: bool = False,
) -> Generator[Profiler, None, None]:
    """Create, run, and yield a Profiler for *script_name*, then restore Config."""
    root_dir = tmp_path_factory.mktemp(tmp_key)
    output_dir = root_dir / "performance"
    output_dir.mkdir()
    prev_settings = Config.settings()
    Config.configure(Settings(root=root_dir, output=output_dir, ignore=[output_dir], duration=DurationOption.relative))
    dest = root_dir / script_name
    dest.write_text((SCRIPTS_DIR / script_name).read_text())
    script_args = ["--serial"] if serial else []
    profiler = Profiler(dest, profile_all=True, python_script_args=script_args)
    with patch.dict(os.environ, _SINGLE_THREAD_ENV), contextlib.suppress(SuccessfulExit):
        profiler.run_profiler(raise_exit=True)
    try:
        yield profiler
    finally:
        Config.configure(prev_settings)


@pytest.fixture(scope="class")
def multithreading_profiler(tmp_path_factory: pytest.TempPathFactory) -> Generator[Profiler, None, None]:
    """Profile the multithreading script once per test class (parallel run)."""
    yield from _run_profiler("multithreading.py", "mt_parallel", tmp_path_factory)


@pytest.fixture(scope="class")
def multithreading_serial_profiler(tmp_path_factory: pytest.TempPathFactory) -> Generator[Profiler, None, None]:
    """Profile the multithreading script once per test class (serial run via --serial)."""
    yield from _run_profiler("multithreading.py", "mt_serial", tmp_path_factory, serial=True)


@pytest.fixture(scope="class")
def multiprocessing_profiler(tmp_path_factory: pytest.TempPathFactory) -> Generator[Profiler, None, None]:
    """Profile the multiprocessing script once per test class (parallel run)."""
    yield from _run_profiler("multiprocessing.py", "mp_parallel", tmp_path_factory)


@pytest.fixture(scope="class")
def multiprocessing_serial_profiler(tmp_path_factory: pytest.TempPathFactory) -> Generator[Profiler, None, None]:
    """Profile the multiprocessing script once per test class (serial run via --serial)."""
    yield from _run_profiler("multiprocessing.py", "mp_serial", tmp_path_factory, serial=True)


class TestMultithreadingProfiling:
    """Tests that the profiler reports CPU usage for each thread worker function."""

    def test_all_worker_functions_are_profiled(self, multithreading_profiler: Profiler):
        """Every thread worker function appears in the profile data."""
        profiled = _profiled_functions(multithreading_profiler)
        for worker in WORKERS:
            assert worker in profiled, f"{worker} not found in profile functions: {profiled}"

    def test_matrix_worker_has_cpu_usage(self, multithreading_profiler: Profiler):
        """matrix_worker records meaningful CPU time."""
        assert _cpu_usage(multithreading_profiler, "matrix_worker") >= _MIN_WORKER_CPU_PCT

    def test_monte_carlo_worker_has_cpu_usage(self, multithreading_profiler: Profiler):
        """monte_carlo_worker records meaningful CPU time across both invocations."""
        assert _cpu_usage(multithreading_profiler, "monte_carlo_worker") >= _MIN_WORKER_CPU_PCT

    def test_statistical_worker_has_cpu_usage(self, multithreading_profiler: Profiler):
        """statistical_worker records meaningful CPU time."""
        assert _cpu_usage(multithreading_profiler, "statistical_worker") >= _MIN_WORKER_CPU_PCT

    def test_parallel_wall_time_less_than_sequential(
        self, multithreading_profiler: Profiler, multithreading_serial_profiler: Profiler
    ):
        """Parallel wall time is less than the serial Scalene run of the same script.

        Both runs are profiled by Scalene with identical settings and single-threaded BLAS,
        so any speedup is attributable purely to thread-level parallelism.

        Note: this test may be flaky on heavily-loaded CI machines where OS scheduling
        prevents threads from running truly concurrently.  If this becomes a recurring
        problem, consider marking it ``pytest.mark.flaky`` or moving it to a nightly suite.
        """
        parallel_time = ProfileData.from_file(multithreading_profiler.output_json).elapsed_time_sec
        serial_time = ProfileData.from_file(multithreading_serial_profiler.output_json).elapsed_time_sec
        assert parallel_time < serial_time, (
            f"Parallel wall time {parallel_time:.2f}s >= serial time {serial_time:.2f}s. "
            "This suggests threads did not run concurrently."
        )


class TestMultiprocessingProfiling:
    """Tests that the profiler handles a multi-process workload.

    Scalene instruments child processes spawned via ``mp.Pool``, so worker
    functions are visible in the profile alongside the main process.
    """

    def test_script_file_is_profiled(self, multiprocessing_profiler: Profiler):
        """The multiprocessing script appears as a profiled file in the output."""
        data = ProfileData.from_file(multiprocessing_profiler.output_json)
        script_str = str(multiprocessing_profiler.python_file)
        assert script_str in data.files, f"Script {script_str!r} not found in profiled files: {list(data.files)}"

    def test_worker_functions_are_profiled(self, multiprocessing_profiler: Profiler):
        """At least one worker function appears in the profile, confirming child-process capture."""
        profiled = _profiled_functions(multiprocessing_profiler)
        assert any(worker in profiled for worker in WORKERS), (
            f"No worker functions found in profile. Profiled functions: {profiled}"
        )

    def test_main_function_has_cpu_usage(self, multiprocessing_profiler: Profiler):
        """``main`` records meaningful CPU time (pool orchestration runs in the main process)."""
        assert _cpu_usage(multiprocessing_profiler, "main") >= _MIN_WORKER_CPU_PCT

    def test_overall_cpu_usage_is_nonzero(self, multiprocessing_profiler: Profiler):
        """Total CPU usage across all profiled functions in the script is positive."""
        data = ProfileData.from_file(multiprocessing_profiler.output_json)
        script_str = str(multiprocessing_profiler.python_file)
        total = sum(fn.total_percentage for fn in data.functions if fn.file_path == script_str)
        assert total > 0

    def test_parallel_wall_time_less_than_sequential(
        self, multiprocessing_profiler: Profiler, multiprocessing_serial_profiler: Profiler
    ):
        """Parallel wall time is less than the serial Scalene run of the same script.

        Both runs are profiled by Scalene with identical settings and single-threaded BLAS,
        so any speedup is attributable purely to process-level parallelism.

        Note: this test may be flaky on heavily-loaded CI machines where OS scheduling
        prevents processes from running truly concurrently.  If this becomes a recurring
        problem, consider marking it ``pytest.mark.flaky`` or moving it to a nightly suite.
        """
        parallel_time = ProfileData.from_file(multiprocessing_profiler.output_json).elapsed_time_sec
        serial_time = ProfileData.from_file(multiprocessing_serial_profiler.output_json).elapsed_time_sec
        assert parallel_time < serial_time, (
            f"Parallel wall time {parallel_time:.2f}s >= serial time {serial_time:.2f}s. "
            "This suggests processes did not run concurrently."
        )
