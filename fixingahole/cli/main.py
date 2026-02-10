# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command-line entrypoints integrated Scalene profiler Fixing-a-Hole."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from colours import Colour
from typer import Exit

from fixingahole import DURATION, IGNORE_DIRS, OUTPUT_DIR, ROOT_DIR, LogLevel, Profiler, ProfileSummary, StatisticsManager
from fixingahole.config import DurationOption
from fixingahole.profiler.utils import find_path

app = typer.Typer(
    rich_markup_mode="markdown",
    epilog=":copyright: Xanadu Quantum Technologies",
)
ModuleType = type(typer)


def profile(  # noqa: PLR0913
    *,
    filename: Annotated[
        str,
        typer.Argument(
            help="Name of the script or notebook to profile.",
            show_default=False,
        ),
    ],
    python_script_args: typer.Context,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-directory",
            "-o",
            help="Directory to save the results to.",
            show_default=True,
        ),
    ] = OUTPUT_DIR,
    cpu_only: Annotated[
        bool,
        typer.Option(
            "--cpu/--memory",
            "-c/-m",
            help="Profile only the CPU runtime or both CPU and memory usage of the script or notebook.",
            show_default=True,
        ),
    ] = True,
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Also profile how external libraries and modules are used.",
            show_default=True,
        ),
    ] = False,
    precision: Annotated[
        int | None,
        typer.Option(
            "--precision",
            "-p",
            help="Level of memory sampling precision. -10 is fastest, least precise; 10 is slowest, most precise.",
            show_default="0",
            min=-10,
            max=10,
        ),
    ] = None,
    trace: Annotated[
        bool,
        typer.Option(
            "--trace/--no-trace",
            "-t/-nt",
            help="Capture the stack traces for the most expensive function calls.",
            show_default=True,
        ),
    ] = True,
    loglevel: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            "-l",
            help="Log level to capture while profiling.",
            case_sensitive=False,
            show_default=True,
        ),
    ] = LogLevel.WARNING,
    noplots: Annotated[
        bool,
        typer.Option(
            "--no-plots",
            "-np",
            help="Prevent plotting functions from running while profiling a script.",
            show_default=True,
        ),
    ] = False,
    live: Annotated[
        float,
        typer.Option(
            help="Update the profile output every so many seconds as the profiling happens.",
            show_default=True,
            min=1,
        ),
    ] = float("inf"),
    ignore: Annotated[
        list[Path] | None,
        typer.Option(
            "--ignore",
            "-i",
            help="Specific folders to ignore while profiling. Paths are resolved relative to the current directory.",
            show_default=True,
        ),
    ] = None,
    duration: Annotated[
        DurationOption | None,
        typer.Option(
            help="Temporarily set whether the summary shows duration times as 'absolute' or 'relative' values.",
            hidden=True,
        ),
    ] = None,
    in_place: Annotated[
        bool,
        typer.Option(
            "--in-place/--not-in-place",
            help="Profile the script where it is instead from the output directory.",
            show_default=True,
        ),
    ] = False,
    repeat: Annotated[
        int,
        typer.Option(
            "--repeat",
            "-r",
            help="The number of times to profile the script to average the results.",
            show_default=True,
            min=1,
        ),
    ] = 1,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet/",
            "-q/",
            help="Prevent any printed statements (summary outputs) while profiling.",
        ),
    ] = False,
) -> None:
    """Profile a python script or Jupyter notebook."""
    # Find and Prepare script for profiling.
    if in_place and noplots:
        Colour.error("Error: cannot both profile in-place AND suppress plotting.")
        raise Exit(code=1)
    if quiet:
        os.environ["COLOURS_DISABLE_PRINT"] = "true"
    Colour.blue.print("Initializing...")
    if duration is not None:
        DURATION.update(duration.value)
    full_path = (ROOT_DIR / filename).resolve()
    if full_path.exists() and not full_path.is_dir():
        python_file = full_path
    else:
        python_file: Path = find_path(filename, ROOT_DIR, exclude=IGNORE_DIRS)
        if python_file.is_dir():
            Colour.error(Colour.ORANGE("Error: cannot profile a directory."))
            raise typer.Exit(code=1)
    ignore_dirs: list[Path] = [Path(p).resolve() for p in ignore] if ignore is not None else []

    profiler = Profiler(
        path=python_file,
        python_script_args=python_script_args.args,
        cpu_only=cpu_only,
        precision=precision,
        detailed=detailed,
        loglevel=loglevel,
        noplots=noplots,
        trace=trace,
        live_update=live,
        ignore_dirs=ignore_dirs,
        in_place=in_place,
        output_dir=output_dir,
    )

    cli_args = sys.argv
    cli_args[0] = Path(cli_args[0]).name
    preamble = " ".join(cli_args) + "\n"

    Colour.print(
        Colour.blue("Profiling:"),
        Colour.green(profiler.profile_path),
        "for speed." if profiler.cpu_only else "for memory usage.",
    )

    stats = StatisticsManager()
    for _ in range(repeat):
        summary = profiler.run_profiler(preamble=preamble, raise_exit=(repeat == 1))
        if summary is not None:
            stats.insert(summary)
    if stats.count > 1:
        stats_file = profiler.output_file.with_name("function_stats.json")
        stats.save_as_json(stats_file, stats.stats())


# Register the profile function as a command in this CLI
app.command(
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    rich_help_panel="Utilities",
    epilog=":copyright: Xanadu Quantum Technologies",
)(profile)


@app.command(
    no_args_is_help=True,
    rich_help_panel="Utilities",
    epilog=":copyright: Xanadu Quantum Technologies",
)
def summarize(
    filename: Annotated[
        str,
        typer.Argument(
            help="Name of Scalene JSON profile to summarize.",
            show_default=False,
        ),
    ],
    top_n: Annotated[
        int,
        typer.Option(
            "-n",
            help="Show only up to the top n functions.",
            show_default=True,
            min=1,
        ),
    ] = 10,
    threshold: Annotated[
        float,
        typer.Option(
            "-t",
            help="Only report functions with at least this percent of CPU time.",
            show_default=True,
            min=0,
        ),
    ] = 0.1,
) -> str:
    """Summarize a Scalene JSON profile."""
    file = find_path(filename, in_dir=ROOT_DIR)
    summary = ProfileSummary(file).summary(top_n, threshold)
    Colour.print(summary)
    return summary


@app.command(
    no_args_is_help=True,
    rich_help_panel="Utilities",
    epilog=":copyright: Xanadu Quantum Technologies",
)
def stats(
    folder: Annotated[
        str,
        typer.Argument(
            help="Name of the folder containing multiple Scalene JSON profiles to generate stats for.",
            show_default=False,
        ),
    ],
    output_file: Annotated[
        str | None,
        typer.Option(
            "--output-file",
            "-o",
            help="Filename to save the results to within the folder.",
            show_default=True,
        ),
    ] = None,
    metadata: Annotated[
        bool,
        typer.Option(
            help="Capture and save the git repo metadata.",
            show_default=True,
        ),
    ] = True,
    sort: Annotated[
        bool,
        typer.Option(
            help="Sort the statistics by average user time, decending.",
            show_default=True,
        ),
    ] = True,
) -> StatisticsManager:
    """Generate statistics for a group of Scalene JSON profiles."""
    stats = StatisticsManager()
    directory, files = find_path(folder, in_dir=ROOT_DIR, return_suffix=".json", subfolder_only=True)
    for file in files:
        summary = ProfileSummary(file)
        stats.insert(summary)
    stats_file = directory / "function_stats.json" if output_file is None else (directory / output_file).with_suffix(".json")
    data = stats.stats()
    data = stats.save_as_json(stats_file, data, save_metadata=metadata, sort=sort)
    Colour.print(json.dumps(data, indent=2))
    return stats


def version_callback(value: bool) -> None:
    """Print `fixingahole --version`."""
    if value:
        from fixingahole import about  # noqa: PLC0415

        about()
        raise Exit(code=0)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show the current version of Fixing-A-Hole.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Integrated Scalene Profiler CLI."""


if __name__ == "__main__":
    app()
