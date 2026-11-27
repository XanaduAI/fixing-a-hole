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
"""Command-line entrypoints integrated Scalene profiler fixing-a-hole."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from colours import Colour
from typer import Exit

from fixingahole import Config, LogLevel, PlottingLibrary, Profiler, ProfileSummary, StatisticsManager, StatsSummary
from fixingahole.config import DurationOption
from fixingahole.profiler.utils import find_path

app = typer.Typer(epilog="\u00a9 Xanadu Quantum Technologies")


def profile(
    *,
    filename: Annotated[
        str,
        typer.Argument(
            help="Name of the script or notebook to profile.",
            show_default=False,
            rich_help_panel="Profiling",
        ),
    ],
    ctx: typer.Context,
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
    ] = LogLevel.NONE,
    no_plots: Annotated[
        list[PlottingLibrary] | None,
        typer.Option(
            "--no-plots",
            "-np",
            help="Separate entries for each plotting library to suppress during profiling, i.e. `-np matplotlib -np plotly`",
            case_sensitive=False,
            show_default=True,
            rich_help_panel="Preprocessing",
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
        DurationOption | None,
        typer.Option(
            help="Temporarily set whether the summary shows duration times as 'absolute' or 'relative' values.",
            hidden=True,
        ),
    ] = None,
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

    Put fixing-a-hole options before the filename, Scalene arguments after it,
    and script arguments after ---.
    """
    # Set some configuration.
    if repeat > 1:
        Colour.info(
            "Suppressing info logs when repeat > 1. Instead, see summaries in the output folder %s",
            output_dir if output_dir is not None else Config.output(),
        )
        Colour.set_log_level("warning")
    Config.update_duration(duration.value if duration is not None else Config.settings().duration.value)
    all_args = ctx.args
    if "---" in all_args:
        sep = all_args.index("---")
        scalene_args, script_args = all_args[:sep], all_args[sep + 1 :]
    else:
        scalene_args, script_args = all_args, []

    # Find and Prepare script for profiling.
    Colour.blue.info("Initializing...")
    full_path = (Config.root() / filename).resolve()
    python_file = (
        full_path if full_path.is_file() else find_path(filename, Config.root(), exclude=[*Config.ignore(), ".venv", ".git"])
    )
    if python_file.is_dir():
        Colour.error("Error: cannot profile a directory.")
        raise Exit(code=1)

    profiler = Profiler(
        python_file,
        profiler_args=scalene_args,
        script_args=script_args,
        log_level=log_level,
        no_plots=no_plots,
        output_dir=output_dir,
    )

    preamble = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\n"
    Colour.info("%s %s", Colour.blue("Profiling:"), Colour.green(profiler.profile_path))

    stats = StatisticsManager()
    for _ in range(repeat):
        summary = profiler.run_profiler(preamble=preamble, raise_exit=(repeat == 1))
        # When repeat is 1 (the default), the profiler exits immediately after successfully finishing so the program stops here.
        stats.insert(summary)
    if stats.count > 1:
        stats_file = profiler.output_file.with_name(output_file).with_suffix(".json")
        stats.save_as_json(stats_file, stats.stats(), sort=sort, save_metadata=metadata)
        summary_text = stats.summary()
        Colour.info(summary_text)
        stats_file.with_suffix(".txt").write_text(summary_text, encoding="utf-8")


# Register the profile function as a command in this CLI
app.command(
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    rich_help_panel="Utilities",
    epilog="\u00a9 Xanadu Quantum Technologies",
)(profile)


@app.command(
    no_args_is_help=True,
    rich_help_panel="Utilities",
    epilog="\u00a9 Xanadu Quantum Technologies",
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
    ignore = [path for path in Config.ignore() if path != Config.output()]
    file = (
        filepath
        if (filepath := Path(filename).resolve()).is_absolute() and filepath.is_file()
        else find_path(filename, in_dir=Config.root(), exclude=[*ignore, ".venv", ".git"])
    )
    try:
        summary = ProfileSummary(file).summary(top_n, threshold)
    except TypeError:
        summary = StatsSummary(file).summary(top_n, threshold)
    Colour.info(summary)
    return summary


@app.command(
    no_args_is_help=True,
    rich_help_panel="Utilities",
    epilog="\u00a9 Xanadu Quantum Technologies",
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
    ignore = [path for path in Config.ignore() if path != Config.output()]
    directory, files = find_path(
        folder,
        in_dir=Path(folder).resolve().parent,
        exclude=[*ignore, ".venv", ".git"],
        return_suffix=".json",
        subfolder_only=subfolder_only,
    )
    for file in files:
        try:
            summary = ProfileSummary(file)
            stats.insert(summary)
        except (KeyError, TypeError):
            Colour.warning(
                "Failed to summarize %s. Probably not a Scalene JSON file.", Colour.purple(Config.relative_to_cwd(file))
            )

    stats_file = (directory / output_file).with_suffix(".json")
    stats.save_as_json(stats_file, stats.stats(), save_metadata=metadata, sort=sort)
    summary_text = stats.summary()
    Colour.info(summary_text)
    stats_file.with_suffix(".txt").write_text(summary_text, encoding="utf-8")
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
            help="Show the current version of fixing-a-hole.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Integrated Scalene Profiler CLI."""


if __name__ == "__main__":
    app()
