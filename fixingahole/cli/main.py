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
import sys
from contextlib import suppress
from pathlib import Path
from typing import Annotated

import typer
from colours import Colour
from typer import Exit

from fixingahole import Config, LogLevel, Profiler, ProfileSummary, StatisticsManager
from fixingahole.config import DurationOption
from fixingahole.profiler.utils import FindPathException, find_path

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
            rich_help_panel="Profiling",
        ),
    ],
    python_script_args: typer.Context,
    cpu_only: Annotated[
        bool,
        typer.Option(
            "--cpu/--memory",
            "-c/-m",
            help="Profile only the CPU runtime or both CPU and memory usage of the script or notebook.",
            show_default=True,
            rich_help_panel="Profiling",
        ),
    ] = True,
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Also profile how external libraries and modules are used.",
            show_default=True,
            rich_help_panel="Profiling",
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
            rich_help_panel="Profiling",
        ),
    ] = None,
    trace: Annotated[
        bool,
        typer.Option(
            "--trace/--no-trace",
            "-t/-nt",
            help="Capture the stack traces for the most expensive function calls.",
            show_default=True,
            rich_help_panel="Profiling",
        ),
    ] = True,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            "-l",
            help="Log level to capture while profiling.",
            case_sensitive=False,
            show_default=True,
            rich_help_panel="Preprocessing",
        ),
    ] = LogLevel.WARNING,
    no_plots: Annotated[
        list[str] | None,
        typer.Option(
            "--no-plots",
            "-np",
            help="Comma-separated list of plotting libraries to suppress during profiling, i.e. `matplotlib`, `plotly`",
            show_default=True,
            rich_help_panel="Preprocessing",
        ),
    ] = None,
    live: Annotated[
        float,
        typer.Option(
            help="Update the profile output every so many seconds as the profiling happens.",
            show_default=True,
            min=1,
            rich_help_panel="Profiling",
        ),
    ] = float("inf"),
    ignore: Annotated[
        list[Path] | None,
        typer.Option(
            "--ignore",
            "-i",
            help="Specific folders to ignore while profiling. Paths are resolved relative to the current directory.",
            show_default=True,
            rich_help_panel="Profiling",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-directory",
            "-o",
            help="Directory to save the results to.",
            show_default=str(Config.output() / "<script name>" / "<time stamp>"),
            rich_help_panel="Profiling",
        ),
    ] = None,
    duration: Annotated[
        DurationOption,
        typer.Option(
            help="Temporarily set whether the summary shows duration times as 'absolute' or 'relative' values.",
            hidden=True,
        ),
    ] = DurationOption.relative,
    repeat: Annotated[
        int,
        typer.Option(
            "--repeat",
            "-r",
            help="The number of times to profile the script to average the results.",
            show_default=True,
            min=1,
            rich_help_panel="Benchmarking",
        ),
    ] = 1,
    output_file: Annotated[
        str,
        typer.Option(
            help="Filename to save the statistical results to within the folder.",
            show_default=True,
            rich_help_panel="Benchmarking",
        ),
    ] = "function_stats.json",
    metadata: Annotated[
        bool,
        typer.Option(
            help="Save the git repo metadata with the statistics (repo name, branch name, commit hash, UTC date and time).",
            show_default=True,
            rich_help_panel="Benchmarking",
        ),
    ] = True,
    sort: Annotated[
        bool,
        typer.Option(
            help="Sort the statistics by average user time, descending.",
            show_default=True,
            rich_help_panel="Benchmarking",
        ),
    ] = True,
) -> None:
    """Profile a Python script or Jupyter notebook.

    Any options or arguments needed by the script can be added after these options.
    """
    # Set some configuration.
    if repeat > 1:
        Colour.info(
            "Suppressing info logs when repeat > 1. Instead, see summaries in the output folder %s",
            output_dir if output_dir is not None else Config.output(),
        )
        Colour.set_log_level("warning")
    Config.update_duration(duration.value)

    # Find and Prepare script for profiling.
    Colour.blue.info("Initializing...")
    full_path = (Config.root() / filename).resolve()
    if full_path.exists() and not full_path.is_dir():
        python_file = full_path
    else:
        python_file: Path = find_path(filename, Config.root(), exclude=Config.ignore())
        if python_file.is_dir():
            Colour.error("Error: cannot profile a directory.")
            raise Exit(code=1)

    ignore_dirs: list = []
    if ignore is not None:
        Colour.info("Searching for the following folders to ignore: %s", [str(p) for p in ignore])
        for p in ignore:
            with suppress(FindPathException):
                ignore_dirs.append(find_path(p, in_dir=Config.root()))
        Colour.info("Ignoring: %s", [str(p) for p in ignore_dirs])

    no_plots: list[str] = [lib.strip() for item in no_plots for lib in item.split(",")] if no_plots is not None else []
    profiler = Profiler(
        path=python_file,
        python_script_args=python_script_args.args,
        cpu_only=cpu_only,
        precision=precision,
        detailed=detailed,
        log_level=log_level,
        no_plots=no_plots,
        trace=trace,
        live_update=live,
        ignore_dirs=ignore_dirs,
        output_dir=output_dir,
    )

    cli_args = sys.argv
    cli_args[0] = Path(cli_args[0]).name
    preamble = " ".join(cli_args) + "\n"

    Colour.info(
        "%s %s %s",
        Colour.blue("Profiling:"),
        Colour.green(profiler.profile_path),
        "for speed." if profiler.cpu_only else "for memory usage.",
    )

    stats = StatisticsManager()
    for _ in range(repeat):
        summary = profiler.run_profiler(preamble=preamble, raise_exit=(repeat == 1))
        # When repeat is 1 (the default), the profiler exits immediately after successfully finishing so the program stops here.
        stats.insert(summary)
    if stats.count > 1:
        stats_file = profiler.output_file.with_name(output_file).with_suffix(".json")
        stats.save_as_json(stats_file, stats.stats(), sort=sort, save_metadata=metadata)


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
    file = find_path(filename, in_dir=Config.root())
    summary = ProfileSummary(file).summary(top_n, threshold)
    Colour.info(summary)
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
        str,
        typer.Option(
            "--output-file",
            "-o",
            help="Filename to save the results to within the folder.",
            show_default=True,
        ),
    ] = "function_stats.json",
    metadata: Annotated[
        bool,
        typer.Option(
            help="Capture and save the git repo metadata (repo name, branch name, commit hash, UTC date and time).",
            show_default=True,
        ),
    ] = True,
    sort: Annotated[
        bool,
        typer.Option(
            help="Sort the statistics by average user time, descending.",
            show_default=True,
        ),
    ] = True,
    # This option is hidden. It is useful to use if profiling is done in parallel and the results are in different subfolders.
    subfolder_only: Annotated[
        bool,
        typer.Option(
            "--subfolder-only/--all-subfolders",
            help="Only search the immediate subfolder of `folder` rather than recursively searching all subfolders.",
            show_default=True,
            hidden=True,
        ),
    ] = True,
) -> StatisticsManager:
    """Generate statistics for a group of Scalene JSON profiles."""
    stats = StatisticsManager()
    directory, files = find_path(
        folder,
        in_dir=Config.root(),
        return_suffix=".json",
        subfolder_only=subfolder_only,
    )
    for file in files:
        try:
            summary = ProfileSummary(file)
            stats.insert(summary)
        except (KeyError, TypeError):
            bad_file = file
            with suppress(ValueError):
                bad_file = file.relative_to(Path.cwd())
            Colour.warning("Failed to summarize %s. Probably not a Scalene JSON file.", Colour.purple(bad_file))

    stats_file = (directory / output_file).with_suffix(".json")
    saved_data = stats.save_as_json(stats_file, stats.stats(), save_metadata=metadata, sort=sort)
    Colour.info(json.dumps(saved_data, indent=2))
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
