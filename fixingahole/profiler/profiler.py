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
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path

from colours import Colour
from sympy import nextprime
from typer import Exit

from fixingahole import OUTPUT_DIR, ROOT_DIR
from fixingahole.profiler.utils import LogLevel, Spinner, date, memory_with_units


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
        precision: int | None = None,
        detailed: bool = False,
        threshold: float = 1,
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
        self.threshold = threshold
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
            exclude_dir = Path(os.getenv("APPDATA"))
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
            "\n### Add Fixing-A-Hole extras for profiling. ###\n",
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

    def get_usr_bin_time_data(self, stderr: str) -> tuple[str, float]:
        """Max memory resident set size (RSS) and wall time from /usr/bin/time output.

        Returns:
            tuple: (memory_string, walltime_seconds)

        """
        memory_used = -1.0
        walltime = -1.0
        rss_line = "Maximum resident set size (kbytes)" if self.platform == Platform.Linux else "maximum resident set size"

        for line in stderr.splitlines():
            if rss_line in line:
                for part in line.strip().split():
                    try:
                        memory_used = float(part)
                        break
                    except ValueError:
                        pass

            # Parse wall time based on platform
            if self.platform == Platform.Linux and "Elapsed (wall clock) time" in line:
                # Elapsed (wall clock) time (h:mm:ss or m:ss): 0:46.59
                time_parts = line.strip().split()[-1].split(":")
                time_parts.reverse()
                units = [1, 60, 3600]
                walltime = sum(float(t) * u for t, u in zip(time_parts, units, strict=False))
            elif self.platform == Platform.MacOS and " real " in line:
                # 1.55 real  0.65 user  0.32 sys
                walltime = float(line.strip().split()[0])

        unit = "KB" if self.platform == Platform.Linux else "B"
        memory_str = memory_with_units(memory_used, unit=unit, digits=3)
        return memory_str, walltime

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
            "python -m scalene",
            "--reduced-profile --cpu --json",
            "--stacks" if self.trace else "",
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

    def run_profiler(self, preamble: str = "\n") -> None:
        """Profile the python script using Scalene."""
        from fixingahole.profiler import StackReporter, generate_summary, generate_text_report, parse_json  # noqa: PLC0415

        # Fallback to the specified `column_width` if the terminal width cannot be obtained.
        ncols = max(160, len(str(self.profile_file)) + 75)
        ncols = shutil.get_terminal_size(fallback=(ncols, ncols)).columns

        try:
            # Profile the code.
            with Spinner(f"See {Colour.purple(self.output_path)} for details."):
                capture = subprocess.run(
                    self._build_cmd,
                    check=True,
                    text=True,
                    capture_output=self.loglevel.should_catch_warnings(),
                    env=os.environ.copy() | {"LINES": "320", "COLUMNS": f"{ncols}", "FIXING_A_HOLE_PROFILE": "1"},
                )
        except subprocess.CalledProcessError as exc:
            # Catch any shell errors and display them.
            # This includes any fatal errors in Python during execution.
            for exc_output in [exc.stdout, exc.stderr]:
                if exc_output is None:
                    continue
                output = exc_output.decode() if isinstance(exc_output, bytes) else exc_output
                if output.strip():
                    Colour.RED.print("\nExecution Error:")
                    Colour.print(Colour.red_error(output.strip()))
            raise Exit(code=1) from exc
        except KeyboardInterrupt as ki:
            # Make sure to indicate in the profile_results.txt of the interruption.
            message = "\n Profiling interrupted by user. \n"
            self.output_file.write_text(message + self.output_file.read_text(), encoding="utf-8")
            Colour.RED.print(message)
            raise Exit(code=1) from ki
        else:
            # Gather all the details and logs and consicely present them to the user.
            profile_data = parse_json(self.output_file.with_suffix(".json"))
            summary = generate_summary(profile_data, threshold=self.threshold)
            memory = "" if self.cpu_only else f"using {profile_data.max_memory} of RAM"
            finished = f"Finished in {profile_data.walltime or 0:,.3f} seconds {memory}"

            results = self.output_file.read_text()
            log_info = self.log_file.read_text() if self.log_file.exists() else ""
            n_warns = log_info.count("WARNING")
            warn = "warning" if n_warns == 1 else "warnings"
            warning_str = f" ({n_warns} {warn})" if n_warns > 0 else ""

            if capture.stderr is not None and self.platform != Platform.Windows:
                ubt_rss, ubt_walltime = self.get_usr_bin_time_data(capture.stderr)
                rss_report = f"Max RSS Memory Usage: {ubt_rss}\n" if ubt_rss else ""
                rss_report += f"Wall Time: {ubt_walltime:.3f} seconds\n" if ubt_walltime > 0 else ""
            else:
                rss_report = ""

            report = generate_text_report(profile_data, threshold=self.threshold, width=ncols)
            if self.trace:
                reporter = StackReporter(self.output_file.with_suffix(".json"))
                report += reporter.report_stacks_for_top_functions(top_n=5)

            preamble += f"{finished}.\n"
            preamble += f"Check logs {self.log_path}{warning_str}\n" if log_info else ""
            results = f"{preamble}\n{rss_report}{summary}{results}{report}"
            self.output_file.write_text(results, encoding="utf-8")

            Colour.print(f"{summary}{finished}{Colour.orange(warning_str)}.")
            Colour.print(rss_report, "\n")
            if self.trace:
                Colour.print(report)

            raise Exit(code=0)
