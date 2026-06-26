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
"""Integrated Scalene Profiler."""

import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO, runtime_checkable

from colours import Colour
from typer import Exit

from fixingahole import Config
from fixingahole.profiler.scalene_flags import scalene_flags_to_kwargs, scalene_kwargs_to_flags
from fixingahole.profiler.utils import FileWatcher, LogLevel, PlottingLibrary, Spinner, date, memory_with_units

if TYPE_CHECKING:
    from resource import struct_rusage

    from fixingahole import ProfileSummary


class Platform(Enum):
    """OS Platforms."""

    MacOS = "darwin"
    Linux = "linux"
    Windows = "windows"
    Unknown = "unknown"


HOST: Platform = next((p for p in Platform if p.value == platform.system().lower()), Platform.Unknown)


class ProfilerException(Exit):
    """Error to raise if the Profiler fails."""

    def __init__(self, message: str = "", *, code: int = 1) -> None:
        self.message = message
        super().__init__(code=code)


class SuccessfulExit(Exit):
    """Exit when the Profiler succeeds."""

    def __init__(self, message: str = "Profiling successful.", *, code: int = 0) -> None:
        self.message = message
        super().__init__(code=code)


class ResourceUsage:
    """Measures wall time and max RSS of the code executed inside the context."""

    def __init__(self) -> None:
        self.rss: str = ""
        self.walltime: float = 0.0
        self.child_rusage: struct_rusage | None = None
        self._t0: float = 0.0
        self._finished: bool = False

    def __enter__(self) -> "ResourceUsage":
        """Start measuring walltime and RSS."""
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop measuring and compute RSS and walltime."""
        self._finished = True
        self.walltime = time.perf_counter() - self._t0
        if HOST is not Platform.Windows and self.child_rusage is not None:
            unit = "KB" if HOST is Platform.Linux else "B"
            self.rss = memory_with_units(self.child_rusage.ru_maxrss, unit=unit, digits=3)

    @property
    def report(self) -> str:
        """Formatted report string, empty if no data is available."""
        if not self._finished:
            msg = "ResourceUsage.report accessed before context exit."
            raise RuntimeError(msg)
        lines = [
            f"\nMax RSS Memory Usage: {self.rss}" if self.rss else "",
            f"\nTotal Wall Time: {self.walltime:.3f} seconds\n" if self.walltime > 0 else "",
        ]
        return "".join(lines)


@runtime_checkable
class ProfilerConfig(Protocol):
    """A protocol for customizing Profiler initialization.

    Users can implement this configuration object to dynamically determine
    paths and settings before the profiler runs. This is useful for advanced
    use cases where you need to programmatically configure the profiler.

    Required Attributes:
        The setup() method **must** set `profiler.python_file` to a Path object.
        This is the Python script to be profiled.

    Optional Attributes:
        You may optionally set these attributes to customize behavior:
        - `profiler.filestem` (str): Base name for output files (defaults to python_file.stem)
        - `profiler.profile_root` (Path): Directory for profiling outputs
        - `profiler.output_file` (Path): Location for the main output file
    """

    def setup(self, profiler: "Profiler") -> None:
        """Perform any necessary setup on the profiler instance.

        Args:
            profiler: The Profiler instance being initialized. You **must** set
                profiler.python_file to the Path of the script to profile.

        """
        ...


class Profiler:
    """Class for managing profiling with Scalene.

    This class provides a wrapper around Scalene profiling with additional
    features like automatic output organization, live updates, and summary generation.

    Attributes:
        python_file (Path): The original Python script to be profiled.
        filestem (str): Base name used for output files (derived from python_file.stem).
        profile_root (Path): Root directory where all profiling outputs are stored.
        output_file (Path): Scalene table output file containing formatted profile results.
        output_json (Path): Scalene JSON output file containing raw profiling data.
        output_summary (Path): Summary text file with condensed profiling insights.
        log_file (Path): Log file capturing stdout/stderr during profiling.
        cpu_only (bool): If True, profile CPU only (no memory profiling).
        detailed (bool): If True, profile all modules including libraries.
        log_level (LogLevel): Logging verbosity during profiling.
        live_update (float): Interval in seconds for live output updates (inf = disabled).
        ignored_folders (list[Path]): Directories to exclude from profiling.
        run_count (int): Number of times run_profiler() has been called on this instance.

    """

    if TYPE_CHECKING:
        # Runtime types - these attributes are always set before use
        python_file: Path
        filestem: str
        profile_root: Path
        _profile_file: Path
        _output_file: Path

    def __init__(
        self,
        path_or_config: Path | ProfilerConfig,
        /,
        *,
        python_script_args: list[str] | None = None,
        log_level: LogLevel = LogLevel.NONE,
        no_plots: list[PlottingLibrary] | None = None,
        ignore_dirs: list[Path] | None = None,
        output_dir: Path | None = None,
        **scalene_kwargs: bool | float | str | list[str],
    ) -> None:
        # Validate that the Scalene flags are all valid.
        scalene_kwargs = scalene_flags_to_kwargs(scalene_kwargs_to_flags(scalene_kwargs))
        # Extract Scalene flags needed for internal logic and remove them from the
        # pass-through kwargs so they are not double-added when building the command.
        has_memory = bool(scalene_kwargs.pop("memory", False))
        has_cpu_only = bool(scalene_kwargs.pop("cpu_only", False))
        self.cpu_only: bool = has_cpu_only or not has_memory
        self.detailed: bool = bool(scalene_kwargs.pop("profile_all", False))
        self.live_update: float = float(scalene_kwargs.get("profile_interval", float("inf")))

        self.platform = HOST

        #  Assert correct python environment.
        self.assert_platform_os()

        self.script_args: list[str] = python_script_args if python_script_args is not None else []
        self.log_level: LogLevel = log_level
        self.no_plots: list[PlottingLibrary] = (
            [PlottingLibrary(no_plots)] if isinstance(no_plots, str) else (no_plots if no_plots is not None else [])
        )
        # These are always set during initialization.
        self.filestem = None  # type:ignore[ty:invalid-assignment]
        self.profile_root = None  # type:ignore[ty:invalid-assignment]
        self._profile_file = None  # type:ignore[ty:invalid-assignment]
        self._output_file = None  # type:ignore[ty:invalid-assignment]
        self._output_name: str = "profile_results"
        self.ignored_folders: list[Path] = ignore_dirs if ignore_dirs is not None else []
        self.run_count: int = 0
        self.scalene_kwargs = scalene_kwargs

        # Run the user defined setup config.
        if isinstance(path_or_config, ProfilerConfig):
            path_or_config.setup(self)
            # Validate that setup() configured the python_file property.
            if not hasattr(self, "python_file") or self.python_file is None:
                msg = "ProfilerConfig.setup() must set `python_file` property."
                Colour.error("Error: %s", msg.replace("python_file", Colour.BOLD("python_file")))
                raise ProfilerException(msg)
        else:
            self.python_file = path_or_config

        self.python_file = Path(self.python_file)
        if not self.python_file.exists():
            Colour.error("Error: %s does not exist.", Colour.purple(self.python_file))
            msg = f"Error: {self.python_file} does not exist."
            raise ProfilerException(msg, code=127)
        if not self.python_file.is_file():
            msg = "Error: can only profile a regular file."
            Colour.error(msg)
            raise ProfilerException(msg)
        if self.python_file.is_dir():
            msg = "Error: cannot profile a directory."
            Colour.error(msg)
            raise ProfilerException(msg)

        # Prepare the results folder by inferring or setting the necessary properties.
        self.filestem = (self.filestem or self.python_file.stem).replace(" ", "_")
        self.profile_root = self.profile_root or (
            Config.output() / self.filestem / date() if output_dir is None else output_dir
        )
        # Ensure profile_root exists and that the _profile_file and output_file are set.
        if not isinstance(self.profile_root, (str, Path)):
            msg = f"Error: the `profile_root` must be either a string or a Path object, not {type(self.profile_root)}"
            Colour.error(msg)
            raise ProfilerException(msg)
        self.profile_root = Path(self.profile_root)
        self.profile_root.mkdir(parents=True, exist_ok=True)
        self._profile_file = (self.profile_root / self.filestem).with_suffix(".py")
        if self._output_file is None:
            self.output_file = self._output_name
        self.prepare_code_for_profiling()

    def assert_platform_os(self) -> None:
        """Explain that memory profiling is not available on Windows."""
        if not self.cpu_only and HOST is Platform.Windows:
            Colour.error("Memory profiling is not available on Windows\nUsing --cpu-only")
            self.cpu_only = True

    @property
    def in_place(self) -> bool:
        """Determine whether or not to profile the script directly or profile a modified a copy.

        Default to profiling "in place", but use a modified copy if needed. A modified copy is needed
          when suppressing plotting, if the file is a Jupyter notebook, or if the log level changes.
        """
        return not (self.no_plots or self.python_file.suffix != ".py" or self.log_level.capture_output())

    @property
    def excluded_folders(self) -> str:
        """Scalene flag to exclude system python directory when profiling all modules."""
        exclude_dir: list[Path] = [
            Path(os.getenv("APPDATA") or sys.prefix) if HOST is Platform.Windows else Path(sys.executable).resolve().parents[1]
        ]
        exclude_dir.extend([folder for folder in Config.ignore() if folder != Config.output()])
        exclude_dir.extend(self.ignored_folders)  # allow users to ignore the OUTPUT_DIR if they want to.
        return f"--profile-exclude {','.join(map(str, exclude_dir))}"

    @property
    def profile_file(self) -> Path:
        """The file being profiled. Either the original file, or a modified copy."""
        return self.python_file if self.in_place else self._profile_file

    @property
    def output_file(self) -> Path:
        """The location of the Scalene output (as a .txt file)."""
        if self._output_file is None:
            return Path.cwd() / "profile_results.txt"
        # Auto-append the run_count if multiple runs are made.
        name = f"{self._output_name}_{self.run_count}" if self.run_count > 0 else self._output_name
        return (self._output_file.parent / name).with_suffix(".txt")

    @output_file.setter
    def output_file(self, value: str | Path) -> None:
        """Location of the Scalene output (as a .txt file)."""
        if isinstance(value, Path):
            self._output_file = value
        elif self.profile_root is None:
            msg = "Error: The `profile_root` must be set before setting the `output_file` with a string."
            Colour.error(msg)
            raise ProfilerException(msg)
        else:
            self._output_file = self.profile_root / value
        # Extract the filename stem to enable auto-appending the run_count if multiple runs are made.
        self._output_name = self._output_file.stem
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.touch()

    @property
    def output_json(self) -> Path:
        """The location of the Scalene JSON output (as a .json file)."""
        return self.output_file.with_suffix(".json")

    @property
    def output_summary(self) -> Path:
        """The location of the summary file (as a .txt file)."""
        if "results" in self.output_file.name:
            return self.output_file.parent / self.output_file.name.replace("results", "summary")
        return self.output_file.with_name("profile_summary.txt")

    @property
    def path_to_summary(self) -> Path:
        """A relative path to the summary file (as a .txt file)."""
        return Config.relative_to_cwd(self.output_summary)

    @property
    def output_path(self) -> Path:
        """A relative path (from repo root) to the Scalene output (as a .txt file)."""
        return Config.relative_to_cwd(self.output_file)

    @property
    def profile_path(self) -> Path:
        """A relative path script being profiled."""
        return Config.relative_to_cwd(self.profile_file)

    @property
    def log_file(self) -> Path:
        """The location of the logs caught during profiling."""
        return self.output_file.with_name("profile_logs.log")

    @property
    def log_path(self) -> Path:
        """A relative path (from the repo root) of the logs caught during profiling."""
        return Config.relative_to_cwd(self.log_file)

    @staticmethod
    def convert_ipynb_to_py(file_contents: str) -> str:
        """Convert an ipynb to a python script."""
        contents = json.loads(file_contents)
        executable = []
        for cell in contents["cells"]:
            if cell["cell_type"] == "code":
                executable.extend(cell["source"] + ["\n"])
        executable = "".join(executable)
        if not executable:
            msg = "Error: notebook does not contain any executable code."
            Colour.error(msg)
            raise ProfilerException(msg)
        return executable

    def prepare_code_for_profiling(self) -> None:
        """Make a copy of the code being profiled.

        Add modifiers if needed by adding prefix and suffix lines to the code.
        """
        code_to_profile: str = self.python_file.read_text()

        if not self.in_place:
            if self.python_file.suffix == ".ipynb":
                code_to_profile = Profiler.convert_ipynb_to_py(code_to_profile)
            code_lines: list[str] = code_to_profile.split("\n")
            profile_prefix: list[str] = []
            profile_suffix: list[str] = []
            if self.log_level.capture_output():
                logger: list[str] = [
                    "import sys",
                    "import logging",
                    "from pathlib import Path",
                    f"log_file = Path(r'{self.log_path}')",
                    f"logging.basicConfig(filename=log_file, level=logging.{self.log_level.name})",
                    "logging.captureWarnings(True)" if self.log_level.level <= LogLevel.WARNING.level else "",
                    "sys.stdout = log_file.open(mode='a')",
                ]
                profile_prefix += logger
            if self.no_plots:
                profile_prefix.append("from unittest.mock import patch, MagicMock")
                for lib in self.no_plots:
                    if lib == PlottingLibrary.matplotlib:
                        code_lines = [line for line in code_lines if "%matplotlib" not in line]
                        profile_prefix += [
                            "patch_plt = patch('matplotlib.pyplot.show', new=MagicMock())",
                            "patch_plt.start()",
                        ]
                        profile_suffix.append("patch_plt.stop()")
                    elif lib == PlottingLibrary.plotly:
                        profile_prefix += [
                            "patch_plotly = patch('plotly.graph_objects.Figure.show', new=MagicMock())",
                            "patch_plotly.start()",
                        ]
                        profile_suffix.append("patch_plotly.stop()")
            prefix_line = "; ".join([ln for ln in profile_prefix if ln])
            suffix_line = "; ".join([ln for ln in profile_suffix if ln])
            lines_to_join: list[str] = []
            if prefix_line:
                lines_to_join.append(prefix_line)
            lines_to_join.extend(code_lines)
            if suffix_line:
                lines_to_join.append(suffix_line)
            code_to_profile = "\n".join(lines_to_join)

        self._profile_file.write_text(code_to_profile, encoding="utf-8")

    def _json_output_exists(self) -> bool:
        try:
            if not (
                self.output_json.exists() and (content := self.output_json.read_text(encoding="utf-8")) and json.loads(content)
            ):
                raise ValueError("JSON file missing or empty")  # noqa: EM101, TRY003, TRY301
        except (json.JSONDecodeError, OSError, ValueError):
            note = (
                " Did you ignore the script's parent?"
                if any(self.python_file.is_relative_to(d) for d in self.ignored_folders)
                else ""
            )
            Colour.error("Error: Scalene JSON file is empty or unavailable.%s", note)
            return False
        return True

    def env(self) -> dict[str, str]:
        """Clean the environment variables."""
        # With Python 3.12, pytest-cov sets `COV_CORE` environment variables which will inject coverage.py into the
        #  Scalene profiler subprocess, where both tracing tools fight over sys.settrace().
        # This conflict is due to changes in CPython's internal tracing infrastructure and causes significant slowdown.
        clean_env: dict[str, str] = {k: v for k, v in os.environ.items() if not k.startswith("COV_CORE")}
        ncols = max(160, len(str(self.profile_file)) + 75)
        return clean_env | {"LINES": "320", "COLUMNS": f"{ncols}", "FIXINGAHOLE_PROFILE": "1"}

    @property
    def _scalene_run_cmd(self) -> list[str]:
        """Build the profiling run command."""
        cmd = [
            f"{sys.executable} -m scalene run",
            "--profile-all" if self.detailed else "",
            self.excluded_folders,
            "--memory" if not self.cpu_only else "--cpu-only",
            f"--program-path {Config.root()}",
            f"--outfile {self.output_json}",
        ]
        cmd.append(str(self.profile_file))
        cmd.extend(scalene_kwargs_to_flags(self.scalene_kwargs))
        if self.script_args != []:
            cmd.append("---")
            cmd.extend(self.script_args)
        cmd_str = " ".join([ln.strip() for ln in cmd if ln]).strip()
        return cmd_str.split()

    def _run_scalene(self, usage: ResourceUsage) -> None:
        """Launch the Scalene subprocess, forwarding stderr while suppressing Scalene's own status lines."""
        proc = subprocess.Popen(
            self._scalene_run_cmd,
            text=True,
            stdout=subprocess.DEVNULL if self.log_level.capture_output() else None,
            stderr=subprocess.PIPE,
            env=self.env(),
        )
        stderr_tail: deque[str] = deque(maxlen=50)

        def _forward_stderr(pipe: TextIO) -> None:
            """Forward stderr to the terminal, suppressing Scalene's own status lines."""
            suppress = ("Scalene: profile saved", "  To view in browser:", "  To view in terminal:")
            prev_suppressed = False
            for line in pipe:
                if line.startswith(suppress):
                    prev_suppressed = True
                elif line == "\n" and prev_suppressed:
                    prev_suppressed = False
                else:
                    prev_suppressed = False
                    stderr_tail.append(line)
                    sys.stderr.write(line)
                    sys.stderr.flush()

        stderr_thread = threading.Thread(target=_forward_stderr, args=(proc.stderr,), daemon=True)
        stderr_thread.start()
        if HOST is not Platform.Windows:
            _, status, child_rusage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            usage.child_rusage = child_rusage
        else:
            proc.wait()
        stderr_thread.join(timeout=2)
        if stderr_thread.is_alive():
            Colour.warning("Warning: stderr reader thread did not finish within timeout.\n")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args, stderr="".join(stderr_tail))

    def json_to_tables(self) -> None:
        """Run the scalene view command to format the output."""
        if not self._json_output_exists():
            return

        result = subprocess.run(
            [sys.executable, "-m", "scalene", "view", "--cli", "--reduced", str(self.output_json)],
            check=False,
            text=True,
            capture_output=True,
            env=self.env(),
        )
        if result.returncode == 0 and result.stdout:
            self.output_file.write_text(Colour.remove_ansi(result.stdout))

    def summarize(self, preamble: str, usage: ResourceUsage) -> tuple[str, "ProfileSummary"]:
        """Gather all the details and logs and concisely present them to the user."""
        from fixingahole.profiler import ProfileSummary, StackReporter  # noqa: PLC0415

        if not self._json_output_exists():
            msg = "Scalene JSON file is empty or unavailable."
            raise ProfilerException(msg)

        profile_data = ProfileSummary(self.output_json)
        self.json_to_tables()
        profile_summary = profile_data.summary()
        memory = "" if self.cpu_only else f" using {profile_data.max_memory} of heap RAM"
        finished = f"\nFinished in {profile_data.walltime or 0:,.3f} seconds{memory}"

        log_info = self.log_file.read_text() if self.log_file.exists() else ""

        level_counts = {
            lvl.name: count
            for lvl in LogLevel
            if lvl is not LogLevel.NONE and (count := len(re.findall(rf"^{lvl.name}\b", log_info, re.MULTILINE))) >= 1
        }
        warning_str = f" ({', '.join(f'{n} {lvl}' for lvl, n in level_counts.items())})" if level_counts else ""
        logs_plain = f"\nCheck logs {self.log_path}{warning_str}\n" if log_info else ""
        logs_colored = f"\nCheck logs {Colour.purple(self.log_path)}{Colour.ORANGE(warning_str)}\n" if log_info else ""

        stack_report = ""
        show_stacks = self.scalene_kwargs.get("stacks", False)
        if show_stacks:
            reporter = StackReporter(self.output_json)
            stack_report = reporter.report_stacks_for_top_functions(top_n=5)

        results = f"{preamble}{finished}{usage.report}{logs_plain}{profile_summary}\n{stack_report}"
        self.output_summary.write_text(results, encoding="utf-8")

        summary = f"{finished}{usage.report}{logs_colored}{profile_summary}"
        desc: str = "the stack traces of top function calls." if show_stacks else "a copy of this summary."
        summary += f"\nSee {Colour.purple(self.path_to_summary)} for {desc}\n"
        return summary, profile_data

    def run_profiler(self, preamble: str = "\n", raise_exit: bool = False) -> "ProfileSummary":
        """Profile the python script using Scalene."""
        Colour.info(f"See {Colour.purple(self.output_path)} for details.")
        try:
            # Profile the code.
            watcher = (
                FileWatcher(file_path=self.output_json, on_change_callback=self.json_to_tables)
                if 0 < self.live_update < float("inf")
                else contextlib.nullcontext()
            )
            with Spinner(), watcher, ResourceUsage() as usage:
                self._run_scalene(usage)

        except subprocess.CalledProcessError as exc:
            # Catch any shell errors and display them.
            # This includes any fatal errors in Python during execution.
            msg = "\nExecution Error:\n "
            for exc_output in [exc.stdout, exc.stderr]:
                if not exc_output:
                    continue
                output = exc_output.decode() if isinstance(exc_output, bytes) else exc_output
                if err := output.strip():
                    msg += err + "\n"
            Colour.error(msg)
            raise ProfilerException(msg) from exc
        except KeyboardInterrupt as ki:
            # Make sure to indicate in the profile_results.txt of the interruption.
            msg = "\n Profiling interrupted by user. \n"
            with contextlib.suppress(KeyboardInterrupt):
                self.json_to_tables()
            self.output_file.write_text(msg + self.output_file.read_text(), encoding="utf-8")
            Colour.error(msg)
            raise ProfilerException(msg) from ki
        else:
            text, summary = self.summarize(preamble, usage)
            Colour.info(text)
            if raise_exit:
                raise SuccessfulExit
            self.run_count += 1
            return summary
