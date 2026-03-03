# `fixing-a-hole`

Profiling is the process of analyzing the resource usage of code to identify bottlenecks and
potential areas of optimization and improvement. Typical types of resources used by code are CPU
time, memory usage, and disk space. It can be the case that improvements in one area come at the
expense of costs in another.

`fixing-a-hole` uses the [scalene](https://github.com/plasma-umass/scalene) profiler which,
unfortunately, has very limited support on Windows, even for single-threaded CPU usage. Scalene
supports (single- and multi-threaded) CPU and (peak heap) memory usage on macOS, Linux, and WSL
(Windows Subsystem for Linux). It also uses `/usr/bin/time`, when available, as an independent
check on the total walltime and max resident set size (RSS) memory usage.

> [!TIP]
> _"premature optimization is the root of all evil"_ - [Donald Knuth](https://en.wikipedia.org/wiki/Donald_Knuth)

## Usage

### Installing `fixing-a-hole`

`fixing-a-hole` can be installed into your repo using
```bash
uv add https://github.com/XanaduAI/fixing-a-hole.git
```
or
```bash
pip install https://github.com/XanaduAI/fixing-a-hole.git
```
if you're not using [`uv`](https://docs.astral.sh/uv/).

### Configuring `fixing-a-hole`

`fixing-a-hole` works best when configured for profiling a specific code base.
The main settings are `root`, `output`, `ignore`, and `duration`.
- `root` works best when configured to be the path to root of your git repo or code base.
However, there may be circumstances where you want `root` to be your current directory.
- `output` is always defined _relative_ to `root` and is where the profiling results are stored.
- `ignore` is a list of folders to ignore when profiling (the `output` is always ignored). Paths
are resolved relative to the `root` directory unless they're given as absolute paths. Directories
are also only ignored if they exist after being resolved. Ignored folders are also not searched
when looking for scripts to profile.
- `duration` is _either_ "relative" (as a percent of total runtime) or "absolute" and changes how
the resulting times are displayed in summaries. Defaults to "relative".

`fixing-a-hole` resolves global settings with **per-key** precedence:
1. Explicit `Settings` passed to `Config.configure` (overrides everything).
1. Environment variables with `FIXINGAHOLE_` prefix and upper case keys, i.e. `FIXINGAHOLE_ROOT`, etc.
1. `[tool.fixingahole]` in `pyproject.toml`.
1. Built-in defaults.

> [!IMPORTANT]
> When settings are resolved they **fail loudly**. When any configuration source is present, invalid
> values raise an error immediately and are **never** silently replaced by defaults. Only when the
> library finds no configuration at all does it fall back to built-in defaults.

You can pass explicit settings directly from Python. Explicit settings have the highest precedence.

```python
from pathlib import Path
from fixingahole.config import Config, DurationOption, Settings

Config.configure(
    Settings(
        root=Path("/path/to/repo"),
        output="performance",
        ignore=[Path("/path/to/repo/scratch")],
        duration="absolute",
    )
)
```

`Config.configure()` can also be called without arguments at any point to re-read the current
environment variables and `pyproject.toml`. This can be useful when configuration changes after
the initial import.

For environment configuration, `FIXINGAHOLE_IGNORE` accepts a comma-separated list, for example:
`FIXINGAHOLE_IGNORE="build, tmp, .cache"`.
You can also reconfigure from environment variables at runtime:

```python
import os
from fixingahole.config import Config

os.environ["FIXINGAHOLE_IGNORE"] = "scratch,tmp"
os.environ["FIXINGAHOLE_DURATION"] = "relative"

Config.configure()
```

The following is an example configuration for a `pyproject.toml`:
```text
[tool.fixingahole]
root = "/path/to/my/repo/"
output = "profiling/results/"
ignore = ["unfinished_ideas/", "scratch/work/"]
duration = "absolute"
```

You can confirm that your configuration is set up correctly by running
```bash
fixingahole --version
```
in your terminal.

### Scripts and Notebooks

Python scripts `.py` and notebooks `.ipynb` can be profiled using
```bash
fixingahole profile <filename>
```
For example, if you're working on a new method in `my_repo/my_work/my_new_method.ipynb` then
you can profile it using
```bash
fixingahole profile my_new_method.ipynb
```
and so long as `my_new_method.ipynb` is the only file in the repo with that name, it will run it.
Otherwise, you will need to be more specific by calling
`fixingahole profile my_work/my_new_method.ipynb`. You can also always specify the absolute path
to the script.


## Advanced Usage: Custom Profiler Configuration

For more complex use cases, passing static paths to the `Profiler` might not be enough.
You can implement the `ProfilerConfig` protocol to dynamically determine paths and
settings at runtime.

To use this feature, define a class that inherits from `ProfilerConfig` and implements the
`setup(self, profiler: "Profiler") -> None` method.

### Example 1: Environment-Aware Configuration

This configuration adjusts the output directory and logging level based on whether the code is running in a CI environment (like GitHub Actions) or locally.

```python
import os
from pathlib import Path
from fixingahole.profiler import Profiler, ProfilerConfig

class CIConfig(ProfilerConfig):
    def setup(self, profiler: Profiler) -> None:
        # Determine if we are running in CI
        is_ci = os.getenv("CI", "false").lower() == "true"
        profiler.python_file = Path("my_script.py")
        if is_ci:
            # excessive logging and specific artifact path for CI
            profiler.profile_root = Path("./artifacts/profiling")
            # You can also change other profiler settings dynamically
            profiler.detailed = True
        else:
            # simplified local output
            profiler.profile_root = Path("./local_profiles")
        # Ensure directory exists and set output file
        profiler.profile_root.mkdir(parents=True, exist_ok=True)
        profiler.output_file = profiler.profile_root / "results.txt"

# Run the profiler with the dynamic config
Profiler(CIConfig()).run_profiler()
```

### Example 2: Auto-Discover Latest Script

This configuration automatically finds and profiles the most recently modified Python script in a specific directory ("experiments"), saving you from updating the path manually every time you switch scripts.

```python
from pathlib import Path
from colours import Colour
from fixingahole.profiler import Profiler, ProfilerConfig

class LatestExperimentConfig(ProfilerConfig):
    def setup(self, profiler: Profiler) -> None:
        experiments_dir = Path("./experiments")
        # Find the most recently modified .py file
        try:
            latest_script = max(experiments_dir.glob("*.py"), key=lambda p: p.stat().st_mtime)
        except ValueError:
            raise FileNotFoundError("No python scripts found in experiments/")
        Colour.print(f"Profiling latest script: {latest_script.name}")

        profiler.python_file = latest_script
        profiler.filestem = latest_script.stem
        profiler.profile_root = Path("./profiling_results")
        profiler.output_file = profiler.profile_root / f"{latest_script.stem}_results.txt"

# Run with the auto-discovery config
Profiler(LatestExperimentConfig()).run_profiler()
```

## Options

To see all the available options for the `fixing-a-hole` profiler, run
```bash
fixingahole profile --help
```
Additional information for each option can also be found below.

```bash
--cpu/--memory (-c/-m)
```
The main options are `--cpu` vs `--memory`. By default, `fixing-a-hole` will try to profile the
RSS memory usage of the script/experiment. _However_, additional CPU overhead is required in order
to determine the _heap_ memory usage of the script. The slowdown varies depending on the script,
but may be as low as 1.2x to as much as 4x or more. Again, it really depends on the script
itself. The _heap_ memory profiling (using the `--memory` flag) provides line-by-line blame for
memory usage.

> [!TIP]
> It's likely (and recommended) that you have run your script or notebook normally before you
> profile it. _Even the fastest code is useless if it doesn't solve the problem._ However, if
> you're concerned with the overhead of memory sampling, run a default `--cpu` test first to
> establish an expectation on how long you may need to wait when using `--memory`.

```bash
--precision (-p)
```
It is possible to alter the memory sampling overhead using the `--precision` flag. By default,
[scalene](https://github.com/plasma-umass/scalene) will highlight lines of code that allocate more
than about 10 MB of memory. This can be modified to be as verbose as about 10 kB (by setting
`--precision=10`) or as vague as about 10 GB (by setting `--precision=-10`). The higher the level
of precision (`≤10`) the slower the profiling might take as more samples are taken. However,
setting the level of precision too low (`≥-10`) _may_ result in an uninformed summary. You will
need to find the right balance for the level of profiling that you are doing. Again, the speed
depends on the script itself.

```bash
--detailed (-d)
```
By default, `fixing-a-hole` will only report CPU and memory usage within the `root` directory
(see how to configure `fixing-a-hole` above). However, if you would also like a report on the
usage by imported modules, such as `scipy`, `numpy`, etc., then use the `--detailed` flag.
This can be used along with `--ignore` to build a report with only the relevant modules.

```bash
--trace (-t)
```
By default, `fixing-a-hole` will build the stack traces for the most expensive function calls.
This helps determine where the most expensive function calls are originating from and helps
distinguish the difference between functions that are expensive to call even once from functions
that are called repeatedly.

```bash
--log-level (-l)
```
By default, `fixing-a-hole` will capture warnings while profiling scripts and save them to a log
file. More or less detailed capture can be specified using the `--log-level` flag. The options
are: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Each level will capture that level of
severity _and higher_. So the default capturing `WARNING` will also capture `ERROR` and
`CRITICAL`. However, if you have a syntax error or something, your code will still crash, not run,
and throw errors during profiling.

```bash
--no-plots (-np)
```
By default, if your script or notebook generates plots, then `fixing-a-hole` will profile that
too. The downside of this is that if a plot is opened and you take 5 seconds to close it, those 5
seconds will count towards how long it took your code to run. If you would like to temporarily
disable generating plots, you can specify which plotting libraries to suppress with the
`--no-plots` flag. This will temporarily prevent your code from generating plots without modifying
your code. Simply provide a separate library for each `--no-plots` flag,
i.e. `-np matplotlib -np plotly`.
The currently supported libraries are `matplotlib` and `plotly`.

```bash
--live
```
If you would like periodic readouts of the profiling _while_ the profiling is happening, then you
can set the `--live` flag to a value (in seconds). However, this may cause additional,
unintentional side effects.

```bash
--ignore (-i)
```
If there are specific folders that you would like to ignore while profiling, you can either
configure them globally (see [configuration](#configuring-fixing-a-hole) above) or you can
specify each directory individually when invoking the profiler,
i.e. `--ignore foo --ignore /home/bar/baz`. These are resolved relative
to the `root` directory you configure, but you can also set absolute paths.

```bash
--repeat (-r)
```
If there is a need to benchmark a script by profiling it repeatedly and then compute the average
and standard deviation of the results, then using the `--repeat` flag will do this for you. See
also the [benchmarking](#benchmarking) section below. Additional options associated with this flag
can be seen with either `fixingahole profile --help` or `fixingahole stats --help`.

## Results

Results generated from `fixingahole profile` are saved in the configured `output`
(`performance/`) directory relative to the configured `root` directory (see how to configure
`fixing-a-hole` above). It is suggested that the `output` directory is not tracked by `git`. Each
script or notebook that you profile in this way are saved by name and the UTC datetime when you
ran the profile. For example, `my_work/my_new_method.ipynb` will be saved in
`performance/my_new_method/20250639_123456/`. Within the folder will be a copy of the code that
was profiled along with the profile results, the profile summary, and any logs that were
generated.

See below for a portion of an example profile of `tests/scripts/advanced.py` and how to interpret
it.

### Benchmarking

Small variations in profiling results and profiling summaries can arise from many different sources
of noise or complications during a profiling session. In order to obtain a more stable measure of
how changes to your code are affecting performance, you can profile the exact same script repeatedly
and compute the average and standard deviation of the results. `fixing-a-hole` will do this
automatically for you if you use the `--repeat` flag when profiling. Additionally, it will also try
to save some additional metadata, such as the git repo name, git branch name, git commit hash, and
the current UTC date and time when the statistics were generated, if possible.

If multiple previous runs of the same code were made before, then these same statistics can be made
using the `fixingahole stats <folder>` command, so long as all of the Scalene JSON files from each
profile are in the same subfolder. See `fixingahole stats --help` for more details.

### Understanding Memory Profiling: Heap vs RSS

When profiling memory usage, `fixing-a-hole` reports two different metrics: **heap memory** and
**RSS (Resident Set Size) memory**. Understanding the difference between these metrics—and their
limitations—is crucial for interpreting profiling results and predicting potential memory issues.

#### Heap Memory

**What Heap Memory measures:**
Heap memory represents the memory dynamically allocated by your program at runtime—primarily
objects created during program execution (lists, dictionaries, numpy arrays, etc.). The heap
profiler (enabled with `--memory`) tracks allocations and deallocations and attributes them to
specific lines of code. Scalene is capable of distinguishing between memory allocations made in
Python and allocations made by libraries (see the original
[paper](https://www.usenix.org/system/files/osdi23-berger.pdf))

**What Heap Memory can tell you:**
- Which lines of code are responsible for allocating memory objects
- Peak heap usage during program execution
- Memory allocation patterns that can guide optimization
- Where to focus efforts to reduce memory consumption in your code

**What Heap Memory cannot tell you:**
- Memory used by the Python interpreter itself (runtime overhead)
- Stack memory used for function calls and local variables
- Memory-mapped files and shared libraries
- Memory fragmentation overhead
- Kernel memory structures associated with your process

**Limitations of Heap Memory Profiling:**
- Incurs significant CPU overhead due to tracking allocations
- Cannot predict if memory usage will trigger an out of memory (OOM) error since it's incomplete

#### RSS Memory

**What RSS Memory measures:**
Resident Set Size (RSS) is the total amount of physical RAM occupied by your process, as reported
by the operating system. This includes all memory pages currently resident in physical memory.

**What RSS Memory can tell you:**
- The total physical memory footprint of your process
- A more realistic view of actual system memory pressure
- Whether your program is approaching available system memory limits
- Memory overhead from all sources (Python, C extensions, libraries, interpreter)

**What RSS Memory cannot tell you:**
- Which lines of code are responsible for memory usage
- Virtual memory that's been swapped to disk
- Shared library memory that's shared with other processes (counted separately per process)
- Memory that has been allocated but not yet paged into physical RAM
- Future memory growth patterns

**Limitations of RSS Memory Profiling:**
- Provides no attribution to specific code locations
- Can be misleading when processes share memory pages
- Doesn't account for swap space usage, which could prevent OOM
- Doesn't include memory that's been allocated virtually but not yet accessed
- May not trigger OOM errors at the reported values due to overcommit and swap

#### Critical Limitations: Neither Metric Predicts OOM Errors

**Neither heap nor RSS memory can reliably determine if a program will cause an out-of-memory
(OOM) error** on a given machine. Here's why:

1. **Virtual Memory and Overcommit**: Modern operating systems use virtual memory and often allow
memory overcommit. A program can allocate more memory than physically available, relying on swap
space and the assumption that not all allocated memory will be accessed simultaneously.

2. **Swap Space**: RSS only measures physical RAM usage. Systems with swap can handle programs that
exceed physical RAM by swapping pages to disk (at the cost of performance). An OOM error typically
occurs only when both physical RAM and swap are exhausted.

3. **Shared Memory**: RSS counts shared library pages separately for each process. The actual
system-wide memory pressure is lower than the sum of all RSS values, making it difficult to
predict when the system will run out of memory.

4. **Memory Fragmentation**: Even if sufficient total memory exists, fragmentation can prevent
large allocations from succeeding.

5. **Dynamic Behavior**: Both metrics are snapshots or peak values during profiling. Real-world
execution patterns, input data sizes, and concurrent processes can cause dramatically different
memory usage.

6. **Operating System Policies**: Different OS kernels have different OOM killer policies and
thresholds. What causes an OOM on one system may not on another with identical RAM.

7. **Incomplete Accounting**: As noted above, heap memory misses interpreter overhead and native
allocations, while RSS misses virtual allocations and swap. Neither gives a complete picture of
total memory requirements.

**Best Practice for Avoiding OOM Errors:**
Instead of relying solely on profiling metrics, consider these approaches:
- Run your code with realistic data sizes in staging environments that match production
- Monitor both RSS and available system memory during execution
- Use memory limits (ulimit, cgroups) to test behavior under constrained resources
- Implement monitoring and alerting for memory usage trends in production
- Design algorithms with memory scaling in mind (streaming, chunking, etc.)
- Test with 2-3x expected data sizes to ensure headroom


### `profile_summary.txt`

The first line in the summary file is the command used to generate the results. This is followed
by the runtime and max heap memory usage (as reported by scalene) as well as the max RSS memory
usage and total wall time (as reported by `/usr/bin/time`, if available). If the `profile_logs.log`
file is not empty, then a summary is printed next. Following that, the main Profile Summary is
given (it was also printed to stdout). Finally, if requested, the Stack Trace Summary is displayed.
The Stack Trace Summary helps to identify whether or not expensive function calls are the result of
one long execution or repeated calls to a less expensive function call.

```text
fixingahole profile advanced.py --memory

Finished in 9.318 seconds using 1.376 GB of heap RAM
Max RSS Memory Usage: 1.958 GB
Total Wall Time: 9.950 seconds

Check logs performance/advanced/20260129_201440/profile_logs.log (6 warnings)

Profile Summary
=========================================================

Top 7 Functions by Total Runtime:
---------------------------------------------------------
 1. data_serialization        74.37% (advanced.py:122)
 2. fourier_analysis          14.19% (advanced.py:84)
 3. statistical_analysis       6.77% (advanced.py:48)
 4. matrix_operations          1.91% (advanced.py:15)
 5. monte_carlo_simulation     0.46% (advanced.py:36)
 6. main                       0.12% (advanced.py:145)
 7. recursive_computation      0.10% (advanced.py:113)

Top 5 Functions by Memory Usage:
---------------------------------------------------------
 1. fourier_analysis            610 MB (advanced.py:84)
 2. data_serialization          207 MB (advanced.py:122)
 3. statistical_analysis         76 MB (advanced.py:48)
 4. monte_carlo_simulation       76 MB (advanced.py:36)
 5. matrix_operations            37 MB (advanced.py:15)

Functions by Module:
---------------------------------------------------------
└─ performance (7 func, 97.92% total)
   └─ advanced (7 func, 97.92% total)
      └─ 20260129_201440 (7 func, 97.92% total)
         └─ advanced.py (7 func, 97.92% total)
            ├─ data_serialization.......74.37% (207 MB)
            ├─ fourier_analysis.........14.19% (610 MB)
            ├─ statistical_analysis......6.77% ( 76 MB)
            ├─ matrix_operations.........1.91% ( 37 MB)
            ├─ monte_carlo_simulation....0.46% ( 76 MB)
            ├─ main......................0.12%
            └─ recursive_computation.....0.10%


=========================================================


Stack Trace Summary
===========================================================================

data_serialization, (71.85%)
  └─ performance/advanced/20260129_201440/advanced.py:181; main
     └─ performance/advanced/20260129_201440/advanced.py:192; <module>
        n_calls: 338


fourier_analysis, (10.15%)
  └─ performance/advanced/20260129_201440/advanced.py:171; main
     └─ performance/advanced/20260129_201440/advanced.py:192; <module>
        n_calls: 13


statistical_analysis, (4.84%)
  └─ performance/advanced/20260129_201440/advanced.py:166; main
     └─ performance/advanced/20260129_201440/advanced.py:192; <module>
        n_calls: 20


matrix_operations, (1.55%)
  └─ performance/advanced/20260129_201440/advanced.py:156; main
     └─ performance/advanced/20260129_201440/advanced.py:192; <module>
        n_calls: 6


monte_carlo_simulation, (0.40%)
  └─ performance/advanced/20260129_201440/advanced.py:161; main
     └─ performance/advanced/20260129_201440/advanced.py:192; <module>
        n_calls: 1

===========================================================================
```


### `profile_results.txt`

This file shows the summary from [scalene](https://github.com/plasma-umass/scalene), see also the
original [paper](https://www.usenix.org/system/files/osdi23-berger.pdf) for technical details.

We first see the total memory usage and memory growth rate (the scalene documentation isn't clear
on what "growth rate" is, or how to best interpret it). For each file that contains a significant
portion of the runtime (≥1%) there is a table with headers which are described by the following:

* **Time Python**: How much time was spent in Python code (percent relative to the total runtime).
* **native**: How much time was spent in non-Python code (e.g., libraries written in C/C++,
compiled numpy, etc.).
* **system**: How much time was spent in the system (e.g., I/O, reading and writing data).
* **Memory Python**: How much of the memory allocation happened on the Python side of the code, as
opposed to in non-Python code (e.g., libraries written in C/C++, compiled numpy, etc.).
* **peak**: The highest total memory allocation over the profile period.
* **timeline / %**: Memory consumption generated by this line over the program runtime, and the
percentages of total memory activity this line represents.
* **Copy (MB/s)**: The amount of megabytes being copied per second. Large and frequent memory
copies can be computationally expensive and can significantly slow down your program.

For each file there is a table that shows the most resource intensive lines of code, below each
table is summary of functions in that file which are the most resource intensive. There may also
be a warning identifying a possible memory leak, this may be useful, but the feature is currently
marked as experimental.

```text
     Memory usage: (max: 767.800 MB, growth rate:   5%)
  /home/ubuntu/fixing-a-hole/performance/advanced/20260126_145848/advanced.py: % of time = 100.00% (9.308s) out of 9.308s.
      ╷       ╷       ╷       ╷        ╷       ╷           ╷       ╷
      │Time   │–––––– │–––––– │Memory  │–––––– │–––––––––––│Copy   │
 Line │Python │native │system │Python  │peak   │timeline/% │(MB/s) │/home/ubuntu/fixing-a-hole/performance/advanced/20260126_145848/advanced.…
━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━━┿━━━━━━━┿━━━━━━━━━━━┿━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ... │       │       │       │        │       │           │       │
    9 │       │       │   1%  │        │       │           │    26 │import numpy as np
  ... │       │       │       │        │       │           │       │
   40 │       │       │       │   2%   │   76M │   2%      │    12 │    points = rng.uniform(-1, 1, size=(iterations, 2))
   41 │       │       │       │   8%   │  114M │   4%      │       │    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
  ... │       │       │       │        │       │           │       │
   53 │       │       │       │        │   76M │   2%      │       │    normal_data = rng.normal(loc=0, scale=1, size=sample_size)
   54 │       │       │       │  46%   │   76M │   2%      │       │    exponential_data = rng.exponential(scale=2, size=sample_size)
  ... │       │       │       │        │       │           │       │
   60 │       │       │       │   5%   │   76M │   2%      │       │            "std": float(np.std(normal_data)),
   61 │       │       │       │   8%   │   76M │   2%      │    15 │            "median": float(np.median(normal_data)),
  ... │       │       │       │        │       │           │       │
   63 │       │       │       │  40%   │   77M │   2%      │       │                "25": float(np.percentile(normal_data, 25)),
   64 │       │       │       │  10%   │   76M │   2%      │    21 │                "50": float(np.percentile(normal_data, 50)),
   65 │       │       │       │        │   76M │   2%      │     8 │                "75": float(np.percentile(normal_data, 75)),
   66 │       │       │       │        │   76M │   2%      │       │                "95": float(np.percentile(normal_data, 95)),
  ... │       │       │       │        │       │           │       │
   71 │       │       │       │        │   76M │   2%      │       │            "std": float(np.std(exponential_data)),
   72 │       │       │       │        │   76M │   2%      │    16 │            "median": float(np.median(exponential_data)),
  ... │       │       │       │        │       │           │       │
   89 │       │       │       │  26%   │   76M │   2%      │     3 │    t = np.linspace(0, 10, signal_length)
  ... │       │       │       │        │       │           │       │
   91 │       │       │       │   7%   │  153M │  10%      │       │        np.sin(2 * np.pi * 5 * t)
   92 │       │       │       │        │  153M │   6%      │       │        + 0.5 * np.sin(2 * np.pi * 10 * t)
   93 │       │       │       │        │  153M │   6%      │       │        + 0.3 * np.sin(2 * np.pi * 20 * t)
   94 │       │       │       │        │   76M │   2%      │       │        + rng.normal(0, 0.1, signal_length)
  ... │       │       │       │        │       │           │       │
   98 │       │    1% │   1%  │        │  610M │  16%      │       │    fft_result = np.fft.fft(signal)
   99 │       │       │       │        │  229M │   6%      │     9 │    frequencies = np.fft.fftfreq(signal_length, d=0.001)
  ... │       │       │       │        │       │           │       │
  102 │       │       │       │        │  153M │   4%      │       │    power = np.abs(fft_result) ** 2
  103 │       │   11% │       │   3%   │   76M │   2%      │       │    top_indices = np.argsort(power)[-5:]
  ... │       │       │       │        │       │           │       │
  131 │       │       │       │  82%   │   80M │   2%      │       │            "matrix": rng.uniform(0, 100, size=(50, 50)).tolist(),
  ... │       │       │       │        │       │           │       │
  141 │   34% │    9% │       │ 100%   │  160M │   6%      │    45 │        json_str = json.dumps(complex_data)
  142 │    2% │   23% │       │ 100%   │  400M │  10%      │    21 │        _ = json.loads(json_str)
  ... │       │       │       │        │       │           │       │
      ╵       ╵       ╵       ╵        ╵       ╵           ╵       ╵

Function summaries:
  matrix_operations (line 15): 1% Python, 1% native
  monte_carlo_simulation (line 36): 0% Python, 0% native
  statistical_analysis (line 48): 1% Python, 5% native
  fourier_analysis (line 84): 1% Python, 12% native
  recursive_computation (line 113): 0% Python, 0% native
  data_serialization (line 122): 37% Python, 33% native
```
