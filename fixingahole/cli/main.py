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

import sys
from pathlib import Path
from typing import Annotated

import typer
from colours import Colour
from typer import Exit

from fixingahole import IGNORE_DIRS, ROOT_DIR, LogLevel, Profiler, ProfileSummary
from fixingahole.profiler.utils import find_path

app = typer.Typer(
    rich_markup_mode="markdown",
    epilog=":copyright: Xanadu Quantum Technologies",
)
ModuleType = type(typer)


def profile(
    *,
    filename: Annotated[
        str,
        typer.Argument(
            help="Name of the script or notebook to profile.",
            show_default=False,
        ),
    ],
    python_script_args: typer.Context,
    cpu_only: Annotated[
        bool,
        typer.Option(
            "--cpu/--memory",
            help="Profile the CPU runtime or the memory usage of the script or notebook.",
            show_default=True,
        ),
    ] = True,
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Also profile how imported libraries and modules are used.",
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
            help="Print the stack traces for the most expensive function calls.",
            show_default=True,
        ),
    ] = True,
    loglevel: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            "-l",
            help="Log level to capture.",
            case_sensitive=False,
            show_default=True,
        ),
    ] = LogLevel.WARNING,
    noplots: Annotated[
        bool,
        typer.Option(
            "--no-plots",
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
            help="Specific folders to ignore while profiling.",
            show_default=True,
        ),
    ] = None,
) -> None:
    """Profile a python script or Jupyter notebook."""
    # Find and Prepare script for profiling.
    Colour.blue.print("Initializing...")
    full_path = (ROOT_DIR / filename).resolve()
    if full_path.exists() and not full_path.is_dir():
        python_file = full_path
    else:
        python_file: Path = find_path(filename, ROOT_DIR, exclude=IGNORE_DIRS)
        if python_file.is_dir():
            Colour.ORANGE.print(Colour.red_error("Error: cannot profile a directory."))
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
    )

    cli_args = sys.argv
    cli_args[0] = Path(cli_args[0]).name
    preamble = " ".join(cli_args) + "\n"

    Colour.print(
        Colour.blue("Profiling:"),
        Colour.green(python_file.relative_to(python_file.parents[1])),
        "for speed." if cpu_only else "for memory usage.",
    )
    profiler.run_profiler(preamble=preamble)


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
            help="Name of the script or notebook to profile.",
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
    summary = ProfileSummary(filename).summary(top_n, threshold)
    Colour.print(summary)
    return summary


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
