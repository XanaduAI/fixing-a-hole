# fixing-a-hole

Run [Scalene](https://github.com/plasma-umass/scalene) on Python scripts and notebooks,
with organized results, function summaries, logging, and statistics over repeated runs.

Scalene controls profiling. `fixing-a-hole` prepares the target, forwards Scalene arguments,
and collects the results.

## Install and run

Requires Python 3.12 or newer. Install into the project you want to profile:

```bash
uv add git+https://github.com/XanaduAI/fixing-a-hole.git
```

Run commands in that project's Python environment. With uv, prefix the commands below with
`uv run`.

```bash
fixingahole profile script.py --cpu-only
```

Omit `--cpu-only` to use Scalene's default profiling modes.

## Command-line usage

```text
fixingahole profile [fixing-a-hole options] <filename> [Scalene arguments] [--- script arguments]
```

For example:

```bash
fixingahole profile --repeat 3 --log-level INFO script.py --cpu-only --- --size 100
```

| Part | Handled by |
| --- | --- |
| `--repeat 3 --log-level INFO`, before the filename | fixing-a-hole |
| `--cpu-only`, after the filename | Scalene |
| `--size 100`, after `---` | Your script |

`fixing-a-hole` options must come before the filename. Everything after `---` reaches the script,
including arguments named `--repeat`, `--log-level`, or `--help`.

Supply a path relative to the configured root or an absolute path. A filename alone also works
when it uniquely identifies a file under that root.

Notebooks use the same command:

```bash
fixingahole profile experiment.ipynb --cpu-only
```

Notebook code cells are converted into a Python script before profiling.

### fixing-a-hole options

| Option | Purpose |
| --- | --- |
| `--output-directory PATH` | Choose the run's output folder. |
| `--log-level INFO` | Capture stdout and configure Python logging at the chosen level for the script. |
| `--no-plots matplotlib` | Suppress plot display; repeat for `matplotlib` and `plotly`. |
| `--repeat N` | Profile the target N times and save averages and standard deviations. |

Without log capture, script output goes to the terminal. With log capture, stderr remains
visible; existing logging handlers may also continue writing to it.

For all `fixing-a-hole` options:

```bash
fixingahole profile --help
```

### Scalene arguments

Pass native Scalene arguments as you would to `scalene run`. Scalene arguments are forwarded
unchanged, without validation, translation, merging, or conflict resolution.

For Scalene's available arguments:

```bash
uv run -m scalene run --help
uv run -m scalene run --help-advanced
```


`fixing-a-hole` supplies these defaults before your arguments:

- `--program-path`: the configured root.
- `--outfile`: the run's `profile_results.json` path.
- `--profile-exclude`: the system Python installation, from `sys.base_prefix`.

An explicit `--profile-exclude` replaces the default exclusion. To include system Python code:

```bash
fixingahole profile script.py --cpu-only --profile-all --profile-exclude=
```

Every argument is passed through, including ones that override output paths or prevent summary
generation. `fixing-a-hole` still expects JSON at its configured result path.

`--profile-interval` controls Scalene's periodic JSON writes. `fixing-a-hole` tables and summaries are
rendered when the run finishes.

## Python usage

Use separate lists for Scalene arguments and script arguments:

```python
from pathlib import Path

from fixingahole import Profiler

profiler = Profiler(
    "script.py",
    profiler_args=["--cpu-only"],
    script_args=["--size", "100"],
    log_level="INFO",
    output_dir=Path("performance/run"),
)
result = profiler.run_profiler()
```

`run_profiler()` returns a `ProfileSummary`. Reuse the same `Profiler` to make additional runs.

For custom setup, pass an object with a `setup(profiler)` method instead of a path. The method
must set `profiler.python_file`; it can also set `filestem`, `profile_root`, and `output_file`.

## Results and statistics

By default, each run is saved under `performance/<script name>/<UTC timestamp>/`.

| File | Contents |
| --- | --- |
| `profile_results.json` | Raw Scalene profile. |
| `profile_results.txt` | Scalene's terminal tables. |
| `profile_summary.txt` | Function summary, run measurements, and available stack traces. |
| `<script name>.py` | Copy of the target, including preparation or notebook conversion. |
| `profile_logs.log` | Captured stdout and Python logs when log capture is enabled. |

The summary includes Scalene's timing and heap measurements, total wall time, and maximum RSS
on macOS and Linux. Heap usage and RSS are separate memory measurements.

See [Understanding memory profiling: heap vs RSS](MEMORY.md) for details on interpreting
these measurements, their limitations, and out-of-memory errors.

Repeated runs save individual profiles plus `function_stats.json` and `function_stats.txt`:

```bash
fixingahole profile --repeat 5 script.py --cpu-only
```

Statistics include function averages, standard deviations, and Git metadata when available.
Use `--no-metadata` before the filename to omit that metadata.

You can also summarize a saved profile or average existing profiles:

```bash
fixingahole summarize path/to/profile_results.json
fixingahole stats path/to/run-folder
fixingahole stats performance/script --all-subfolders
```

`stats` reads JSON profiles directly inside the selected folder by default. Use
`--all-subfolders` to combine profiles from nested run folders (be cautious, as
this may try to combine profiling results from entirely different scripts).

## Configuration

Without configuration, the root is the current directory, output goes to `performance/`, and
summary durations are relative percentages.

To configure a project, add this to `pyproject.toml`:

```toml
[tool.fixingahole]
root = "."
output = "performance"
ignore = ["build", "scratch"]
duration = "relative"
```

| Setting | Purpose |
| --- | --- |
| `root` | Base directory for target lookup and default profiling scope. |
| `output` | Default results directory; relative paths resolve under `root`. |
| `ignore` | Folders skipped during file searches; the output folder is automatically ignored. |
| `duration` | `relative` percentages or `absolute` times in summaries. |

Relative `root` and `ignore` paths in the table resolve beside `pyproject.toml`. `ignore` affects
file searches; use Scalene's `--profile-exclude` to control profiling exclusions.

`FIXINGAHOLE_ROOT`, `FIXINGAHOLE_OUTPUT`, `FIXINGAHOLE_IGNORE` (comma-separated), and
`FIXINGAHOLE_DURATION` override table settings per key. Passing a `Settings` object to
`Config.configure()` sets the configuration explicitly. `Config.configure()` without arguments
reloads project and environment settings.

Inspect the active settings and installed versions:

```bash
fixingahole --version
```
