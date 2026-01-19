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

import json
import os
import platform
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from threading import Event, Thread

from colours import Colour
from sympy import nextprime
from typer import Exit

from fixingahole import OUTPUT_DIR, ROOT_DIR
from fixingahole.profiler.utils import LogLevel, Spinner, date


class Platform(Enum):
    """OS Platforms."""

    MacOS = "darwin"
    Linux = "linux"
    Windows = "windows"


class Profiler:
    """Class for managing profiling."""

    def __init__(
        self,
        *,
        path: Path,
        python_script_args: list[str] | None = None,
        cpu_only: bool = False,
        precision: int | str | None = None,
        detailed: bool = False,
        loglevel: LogLevel = LogLevel.CRITICAL,
        noplots: bool = False,
        trace: bool = True,
        **_: dict,
    ) -> None:
        self.cpu_only = cpu_only
        self.precision = int(precision) if precision is not None else 0
        self.platform = None

        #  Assert correct python environment.
        self.assert_platform_os()

        self.script_args = python_script_args if python_script_args is not None else []
        self.detailed = detailed
        self.loglevel = loglevel
        self.noplots = noplots
        self._output_file = Path.cwd() / "profile_results.txt"
        self.cli_inputs = ""
        self._precision_limit = 10
        self.trace = trace

        # Prepare the results folder.
        if path.is_file():
            self.python_file = path
            self.filestem = self.python_file.stem.replace(" ", "_")
            self.profile_root = OUTPUT_DIR / self.filestem / date()
            self.profile_root.mkdir(parents=True, exist_ok=True)
            self.profile_file = self.profile_root / f"{self.filestem}.py"
            self.output_file = self.profile_root / "profile_results.txt"
            self.prepare_code_for_profiling()
        elif path.is_dir():
            pass
        elif not path.exists():
            Colour.print(
                Colour.RED("Error:"),
                Colour.purple(path),
                "does not exist.",
            )
            raise Exit(code=127)

    def assert_platform_os(self) -> None:
        """Explain that memory profiling is not available on Windows."""
        match platform.system().lower():
            case Platform.MacOS.value:
                self.platform = Platform.MacOS
            case Platform.Linux.value:
                self.platform = Platform.Linux
            case Platform.Windows.value:
                self.platform = Platform.Windows

        if not self.cpu_only and self.platform == Platform.Windows:
            Colour.RED.print("Memory profiling is not available on Windows")
            Colour.red.print("Using --cpu")
            self.cpu_only = True
        if self.cpu_only and self.precision != 0:
            Colour.orange.print("--precision option is not used with --cpu")

    @property
    def excluded_folders(self) -> str:
        """Scalene flag to exclude system python directory when profiling all modules."""
        exclude_dir = None
        if platform.system() == "Windows":
            appdata = os.getenv("APPDATA")
            exclude_dir = Path(appdata) if appdata else Path(sys.prefix)
        else:
            exclude_dir = Path(sys.executable).resolve().parents[1]
        # Do not exclude directories that are within the repo.
        return f"--profile-exclude {exclude_dir}" if not exclude_dir.is_relative_to(ROOT_DIR) else ""

    @property
    def output_file(self) -> Path:
        """The location of the Scalene output."""
        return self._output_file

    @output_file.setter
    def output_file(self, value: str | Path) -> None:
        """Location of the Scalene output."""
        self._output_file = Path(value)
        self._output_file.touch()

    @property
    def output_path(self) -> Path:
        """A relative path (from repo root) to the Scalene output."""
        try:
            return self.output_file.relative_to(ROOT_DIR)
        except ValueError:
            # output_file is not in the subpath of ROOT_DIR
            #  OR one path is relative and the other is absolute
            return self.output_file

    @property
    def log_file(self) -> Path:
        """The location of the logs caught during profiling."""
        return self.profile_root / "logs.log"

    @property
    def log_path(self) -> Path:
        """A relative path (from the repo root) of the logs caught during profiling."""
        return self.log_file.relative_to(ROOT_DIR)

    @property
    def precision_limit(self) -> int:
        """The precision limit on the memory allocation threshold."""
        return self._precision_limit

    @precision_limit.setter
    def precision_limit(self, value: int) -> None:
        """Precision limit on the memory allocation threshold."""
        self._precision_limit = abs(int(value))

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
            Colour.ORANGE.print(
                Colour.red_error("Error: notebook does not contain any executable code."),
            )
            raise Exit(code=1)
        return executable

    def get_memory_precision(self) -> str:
        """Given an integer, return the memory allocation size to monitor for."""
        verbosity = (
            max(-self.precision_limit, self.precision) if self.precision < 0 else min(self.precision_limit, self.precision)
        )
        if verbosity != self.precision:
            Colour.orange.print(
                f"Warning: -{self.precision_limit} <= precision <= {self.precision_limit}",
            )
        memory_threshold = 10485767  # ~ 10 MB
        memory_threshold = nextprime(int(memory_threshold / 2**verbosity))
        return f"--allocation-sampling-window={memory_threshold}"

    def prepare_code_for_profiling(self) -> tuple[Path, Path]:
        """Add prefix and suffix lines to the code that will be profiled."""
        code_to_profile = self.python_file.read_text()
        if self.python_file.suffix == ".ipynb":
            code_to_profile = Profiler.convert_ipynb_to_py(code_to_profile)

        code_lines = code_to_profile.split("\n")
        divider = ["\n\n"] + ["####################################"] * 3 + ["\n\n"]
        profile_prefix = []
        profile_suffix = []
        logger = [
            "\n### Add extras for profiling. ###\n",
            "import sys",
            "import logging",
            "from pathlib import Path",
            f"log_file = Path(r'{self.log_file}')",
            f"logging.basicConfig(filename=log_file, level=logging.{self.loglevel.name})",
            "logging.captureWarnings(True)" if self.loglevel.should_catch_warnings() else "",
            "sys.stdout = log_file.open(mode='a')",
        ]
        profile_prefix += logger
        if self.noplots:
            profile_prefix += [
                "### Prevent plots from rendering for profiling. ###",
                "from unittest.mock import patch, MagicMock",
                "patch_plt = patch('matplotlib.pyplot.show', new=MagicMock())",
                "patch_plotly = patch('plotly.graph_objects.Figure.show', new=MagicMock())",
                "patch_plt.start()",
                "patch_plotly.start()",
            ]
            profile_suffix += [
                "patch_plt.stop()",
                "patch_plotly.stop()",
            ]
        profile_prefix += [
            "from scalene import scalene_profiler",
            "scalene_profiler.start()",
        ]
        profile_suffix = ["scalene_profiler.stop()", *profile_suffix]

        code_to_profile = "\n".join(
            profile_prefix + divider + code_lines + divider + profile_suffix,
        )
        self.profile_file.write_text(code_to_profile, encoding="utf-8")

        return self.profile_file, self.output_file

    def get_memory_rss(self, stderr: str) -> str:
        """Max memory resident set size (RSS) that occured while profiling."""
        memory_used = -1.0
        rss_line = "Maximum resident set size (kbytes)" if self.platform == Platform.Linux else "maximum resident set size"

        for line in stderr.splitlines():
            if rss_line in line:
                for part in line.strip().split():
                    try:
                        memory_used = float(part)
                        break
                    except ValueError:
                        pass
        byte_prefix = {
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
        }
        if memory_used > 0:
            if memory_used >= byte_prefix["GB"]:
                memory_usage = memory_used / byte_prefix["GB"]
                unit = "GB"
            elif memory_used >= byte_prefix["MB"]:
                memory_usage = memory_used / byte_prefix["MB"]
                unit = "MB"
            elif memory_used >= byte_prefix["KB"]:
                memory_usage = memory_used / byte_prefix["KB"]
                unit = "KB"
            else:
                unit = "bytes"
            memory_usage = f"{memory_usage:.2f} {unit}"
        else:
            memory_usage = ""
        return memory_usage

    @property
    def _build_cmd(self) -> list[str]:
        """Build the profiling run command."""
        sampling_detail = self.get_memory_precision()
        ncols = max(160, len(str(self.profile_file)) + 75)
        usr_bin_time = ""
        if self.platform == Platform.MacOS:
            usr_bin_time = "/usr/bin/time -l"
        elif self.platform == Platform.Linux:
            usr_bin_time = "/usr/bin/time -v"

        cmd = [
            usr_bin_time,
            "python -m scalene run",
            "--reduced-profile --cpu-only",
            "--json --stacks" if self.trace else "",
            f"--profile-all {self.excluded_folders}" if self.detailed else "",
            f"--memory {sampling_detail}" if not self.cpu_only else "",
            f"--program-path {ROOT_DIR} --column-width={ncols}",
            f"--profile-interval=5 --outfile={self.output_file.with_suffix('.json')}",
        ]
        cmd.append(str(self.profile_file))
        if self.script_args != []:
            cmd.append("---")
            cmd.extend(self.script_args)
        cmd_str = " ".join(cmd).strip()
        return cmd_str.split()

    def _watch_output_file(self, stop_event: Event, ncols: int) -> None:
        """Watch JSON output for changes and update the text output.

        Args:
            stop_event: Event to signal when to stop watching
            ncols: Number of columns for formatting the output

        """
        last_mtime = None
        while not stop_event.is_set():
            if self.output_file.with_suffix(".json").exists():
                current_mtime = self.output_file.stat().st_mtime
                if last_mtime is None or current_mtime > last_mtime:
                    last_mtime = current_mtime
                    # Run the scalene view command to format the output
                    result = subprocess.run(
                        f"python -m scalene view --cli --reduced {self.output_file.with_suffix('.json')}".split(),
                        check=False,
                        text=True,
                        capture_output=True,
                        env=os.environ.copy() | {"LINES": "320", "COLUMNS": f"{ncols}", "FIXING_A_HOLE_PROFILE": "1"},
                    )
                    if result.returncode == 0:
                        self.output_file.write_text(Colour.remove_ansi(result.stdout))
            time.sleep(0.5)  # Check for changes every 0.5 seconds

    def _start_watcher(self, ncols: int) -> tuple[Thread, Event]:
        """Start a background thread to watch and update the output file during profiling."""
        stop_event = Event()
        watcher_thread = Thread(target=self._watch_output_file, args=(stop_event, ncols), daemon=True)
        watcher_thread.start()
        return watcher_thread, stop_event

    @staticmethod
    def _stop_watcher(watcher_thread: Thread, stop_event: Event) -> None:
        """Stop the background thread."""
        stop_event.set()
        watcher_thread.join(timeout=1.0)

    def run_profiler(self, preamble: str = "\n") -> None:
        """Profile the python script using Scalene."""
        from fixingahole.profiler import ProfileSummary, StackReporter  # noqa: PLC0415

        ncols = max(160, len(str(self.profile_file)) + 75)
        try:
            # Profile the code.
            with Spinner(f"See {Colour.purple(self.output_path)} for details."):
                watcher_thread, stop_event = self._start_watcher(ncols)
                try:
                    # Run the profiling command
                    capture = subprocess.run(
                        self._build_cmd,
                        check=True,
                        text=True,
                        capture_output=self.loglevel.should_catch_warnings(),
                        env=os.environ.copy() | {"LINES": "320", "COLUMNS": f"{ncols}", "FIXING_A_HOLE_PROFILE": "1"},
                    )
                finally:
                    self._stop_watcher(watcher_thread, stop_event)

        except subprocess.CalledProcessError as exc:
            # Catch any shell errors and display them.
            # This includes any fatal errors in Python during execution.
            for exc_output in [exc.stdout, exc.stderr]:
                if exc_output is None:
                    continue
                output = exc_output.decode() if isinstance(exc_output, bytes) else exc_output
                if output.strip():
                    Colour.print(Colour.RED("\nExecution Error:\n"), Colour.red_error(output.strip()))
            raise Exit(code=1) from exc
        except KeyboardInterrupt as ki:
            # Make sure to indicate in the profile_results.txt of the interruption.
            message = "\n Profiling interrupted by user. \n"
            self.output_file.write_text(message + self.output_file.read_text(), encoding="utf-8")
            Colour.RED.print(message)
            raise Exit(code=1) from ki
        else:
            # Gather all the details and logs and consicely present them to the user.
            profile_data = ProfileSummary(self.output_file.with_suffix(".json"))
            summary = profile_data.summary()
            memory = "" if self.cpu_only else f"using {profile_data.max_memory} of RAM"
            finished = f"Finished in {profile_data.walltime or 0:,.3f} seconds {memory}"

            results = self.output_file.read_text()
            log_info = self.log_file.read_text() if self.log_file.exists() else ""
            n_warns = log_info.count("WARNING")
            warning_str = f" ({n_warns} {'warning' if n_warns == 1 else 'warnings'})" if n_warns > 0 else ""

            if capture.stderr is not None and self.platform != Platform.Windows:
                rss = self.get_memory_rss(capture.stderr)
                rss_report = f"Max RSS Memory Usage: {rss}\n" if rss else ""
            else:
                rss_report = ""

            report = ""
            if self.trace:
                reporter = StackReporter(self.output_file.with_suffix(".json"))
                report = reporter.report_stacks_for_top_functions(top_n=5)

            preamble += f"{finished}.\n"
            preamble += f"Check logs {self.log_path}{warning_str}\n" if log_info else ""
            results = f"{preamble}\n{rss_report}{summary}{results}{report}"
            self.output_file.write_text(results, encoding="utf-8")

            Colour.print(f"{summary}{finished}{Colour.orange(warning_str)}.\n", rss_report, "\n")
            if self.trace:
                Colour.print(report)

            raise Exit(code=0)
