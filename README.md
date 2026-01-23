# `fixing-a-hole` Profiler

Profiling is the process of analyzing the resource usage of code to identify
bottlenecks and potential areas of optimization and improvement. Typical types
of resources used by code are CPU time, memory usage, and disk space. It can be
the case that improvements in one area come at the expense of costs in another.

`fixing-a-hole` uses the [scalene](https://github.com/plasma-umass/scalene) profiler which,
unfortunately, has very limited support on Windows, even for single-threaded CPU usage.
Scalene supports (single- and multi-threaded) CPU and (peak heap) memory usage on macOS,
Linux, and WSL (Windows Subsystem for Linux).

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

If you're installing `fixing-a-hole` into a repo, you can
configure some defaults in your `pyproject.toml`.
1. The `root` directory determines how to refer to your codebase and
is set as the current working directory (meaning wherever `fixingahole profile`
is invoked from). Setting this to the root of your repo will provide the best
results for profiling code within your repo.
1. The profiling results are saved in the specified `output` directory.
The default is set as `performance/` relative to the `root` directory.
1. Additional directories to `ignore` can be also specified. By
default, the `.git`, `.venv`, and `output` (`performance/`) directories,
relative to `root` are not searched when looking for scripts to profile.

The following is an example configuration:
```text
[tool.fixingahole]
root = "/path/to/my/repo/"
output = "profiling/results/"
ignore = ["unfinished_ideas/", "scratch/work/"]
```


### Scripts and Notebooks

Python scripts `.py` and notebooks `.ipynb` can be profiled using
```bash
fixingahole profile <filename>
```
For example, if you're working on a new method in `my_repo/my_work/my_new_method.ipynb`
then you can profile it using
```bash
fixingahole profile my_new_method.ipynb
```
and so long as `my_new_method.ipynb` is the only file in the repo with that name, it
will run it. Otherwise, you will need to be more specific by calling
`fixingahole profile my_work/my_new_method.ipynb`.
You can also always specify the absolute path to the script.

## Options

To see all the available options for the `fixing-a-hole` profiler, run
```bash
fixingahole profile --help
```

Additional information for each option can also be found below.
```bash
--cpu/--memory
```
The main options are `--cpu` vs `--memory`. By default, `fixing-a-hole` will try to
profile the RSS memory usage of the script/experiment. _However_, additional CPU overhead
is required in order to determine the _heap_ memory usage of the script. The slowdown
varies depending on the script, but may be as low as 1.2x to as much as 4x or more.
Again, it really depends on the script itself.
The _heap_ memory profiling (using the `--memory` flag) provides line-by-line blame for
memory usage.

> [!TIP]
> It's likely (and recommended) that you have run your script or notebook normally
> before you profile it. Even the fastest code is useless if it doesn't solve the
> problem. However, if you're concerned with the overhead of memory sampling,
> run a default `--cpu` test first to establish an expectation on how long you may
> need to wait when using `--memory`.

```bash
--precision=<n>
```
It is possible to alter the memory sampling overhead using the `--precision` flag.
By default, [scalene](https://github.com/plasma-umass/scalene) will highlight lines
of code that allocate more than about 10 MB of memory. This can be modified to be as
verbose as about 325 kB (by setting `--precision=5`) or as vague as about 325 MB
(by setting `--precision=-5`). The higher the level of precision (`≤10`) the slower
the profiling might take as more samples are needed. However, setting the level of
precision too low (`≥-10`) _may_ result in an uninformed summary. You will need to
find the right balance for the level of profiling that you are doing. Again, the
speed depends on the script itself.

```bash
--detailed
```
By default, `fixing-a-hole` will only report CPU and memory usage within the `root`
directory (see how to configure `fixing-a-hole` above). However, if you would also
like a report on the usage by imported modules, such as `scipy`, `numpy`, etc.,
then use the `--detailed` flag.

```bash
--trace
```
By default, `fixing-a-hole` will print the stack traces for the most expensive function calls.
This helps determine where the most expensive function calls are originating from and
helps distinguish the difference between functions that are expensive to call even once
from functions that are called repeatedly.

```bash
--log-level
```
By default, `fixing-a-hole` will capture warnings while profiling scripts and save them to a
log file. More or less detailed capture can be specified using the `--log-level` flag.
The options are: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Each level will
capture that level of severity _and higher_. So the default capturing `WARNING` will
also capture `ERROR` and `CRITICAL`. However, if you have a syntax error or something,
your code will still crash, not run, and throw errors in the terminal.

```bash
--no-plots
```
By default, if your script or notebook generates plots, then `fixing-a-hole` will profile that
too. The downside of this is that if a plot is opened and you take 5 seconds to close
it, those 5 seconds will count towards how long it took your code to run. If you would
like to temporarily disable generating plots, you can profile your code with the
`--no-plots` flag. This will temporarily prevent your code from generating plots without
modifying your code.

```bash
--live
```
If you would like periodic readouts of the profiling _while_ the profiling is happening, then
you can set the `--live` flag to a value (in seconds). However, this may cause additional,
unintentional side effects.

```bash
--ignore
```
If there are specific folders within your repo that you would like to ignore while profiling,
you can either set them globally in your `pyproject.toml` or you can specify each directory
individually when invoking the profiler, i.e. `--ignore foo --ignore bar`. These are resolved
relative to the directory you invoked the profiler from.

## Results

Results generated from `fixingahole profile` are saved in the configured `output` (`performance/`)
directory relative to the configured `root` directory (see how to configure `fixing-a-hole` above).
It is suggested that the `output` directory is not tracked by `git`. Each script or notebook
that you profile in this way are saved by name and the UTC datetime when you ran the profile.
For example, `my_work/my_new_method.ipynb` will be saved in
`performance/my_new_method/20250639_123456/`. Within the folder will be a copy of the
code that was profiled along with the logs and profile results.

See below for a portion of an example profile of `tests/scripts/advanced.py` and how to interpret it.

### `profile_results.txt`

The first line in the results file is the command used to generate the results.
The second line shows the runtime and max heap memory usage.
If the `logs.log` file is not empty, then a summary is printed next.
Following that, the main Profile Summary is given (it was also printed to stdout).

The remainder of the file shows the summary from [scalene](https://github.com/plasma-umass/scalene),
see also the original [paper](https://www.usenix.org/system/files/osdi23-berger.pdf)
for technical details.

We first see the total memory usage (visualized by "sparklines", memory consumption
over the runtime of the profiled code) and memory growth rate (the documentation
isn't clear on what growth rate is, or how to best interpret it).

For each file that contains a significant portion of the runtime (≥1%) there is a
table with headers which are described by the following:
* **Time Python**: How much time was spent in Python code.
* **native**: How much time was spent in non-Python code
(e.g., libraries written in C/C++, numpy, etc.).
* **system**: How much time was spent in the system (e.g., I/O, reading and writing data).
* **Memory Python**: How much of the memory allocation happened on the Python side of the code,
as opposed to in non-Python code (e.g., libraries written in C/C++, numpy, etc.).
* **peak**: The highest total memory allocation over the profile period.
* **timeline / %**: Visualized by "sparklines", memory consumption generated by this line
over the program runtime, and the percentages of total memory activity this line represents.
* **Copy (MB/s)**: The amount of megabytes being copied per second. Large and frequent memory
copies can be computationally expensive and can significantly slow down your program.


```bash
$ fixingahole profile advanced.py --memory
Finished in 9.759 seconds using 767.730 MB of RAM.
Check logs performance/advanced/logs.log (6 warnings)

Profile Summary (9.759s total)
=================================================================

Top 7 Functions by Total Runtime:
-----------------------------------------------------------------
 1. data_serialization         69.8% (advanced.py:142)
 2. fourier_analysis           16.0% (advanced.py:104)
 3. statistical_analysis        7.1% (advanced.py:68)
 4. matrix_operations           2.3% (advanced.py:35)
 5. monte_carlo_simulation      0.9% (advanced.py:56)
 6. recursive_computation       0.1% (advanced.py:133)
 7. main                        0.0% (advanced.py:165)

Top 5 Functions by Memory Usage:
-----------------------------------------------------------------
 1. fourier_analysis            611 MB (advanced.py)
 2. data_serialization          390 MB (advanced.py)
 3. statistical_analysis         77 MB (advanced.py)
 4. monte_carlo_simulation       76 MB (advanced.py)
 5. matrix_operations            36 MB (advanced.py)

Functions by Module:
-----------------------------------------------------------------
└─ performance (7 func, 96.17% total)
   └─ advanced (7 func, 96.17% total)
      └─ 20251201_165709 (7 func, 96.17% total)
         └─ advanced.py (7 func, 96.17% total)
            ├─ data_serialization................................69.81% (390 MB)
            ├─ fourier_analysis..................................15.99% (611 MB)
            ├─ statistical_analysis...............................7.15% ( 77 MB)
            └─ matrix_operations..................................2.25% ( 36 MB)


=================================================================
Finished in 9.759 seconds using 767.730 MB of RAM (6 warnings).
Max RSS Memory Usage: 1.230 GB
Wall Time: 10.350 seconds



        Memory usage: ▁▁▂▃▃▅▁▁▁▃▁▂▂▂▂▂▂▂▂▁▁▂▂▂▁▃▂▂▁▃ (max: 767.730 MB, growth rate: 7.5%)
   /home/ubuntu/fixing-a-hole/tests/scripts/advanced.py: % of time =  99.67% (9.726s) out of 9.759s.
       │Time    │–––––– │––––––│Memory │–––––– │–––––––––––      │Copy   │
  Line │Python  │native │system│Python │peak   │timeline/%       │(MB/s) │ tests/scripts/advanced.py
╺━━━━━━┿━━━━━━━━┿━━━━━━━┿━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━━━━━━━━━━━┿━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ... │        │       │      │       │       │                 │       │
     8 │        │  1.5% │ 1.3% │       │       │                 │    31 │import numpy as np
   ... │        │       │      │       │       │                 │       │
    19 │        │       │      │ 94.2% │ 36 MB │ ▁▁▁▁▁▁▁▁▁  1.0% │       │  matrix_a = rng.uniform(-10, 10, size=(size, size))
    20 │        │       │      │ 73.2% │ 30 MB │                 │       │  matrix_b = rng.uniform(-10, 10, size=(size, size))
   ... │        │       │      │       │       │                 │       │
    30 │        │       │      │  5.2% │ 12 MB │                 │       │  np.linalg.svd(smaller_matrix)
   ... │        │       │      │       │       │                 │       │
    39 │        │       │      │  2.6% │ 74 MB │ ▂▁▁▁▁▁▁▁▁  2.0% │       │  points = rng.uniform(-1, 1, size=(iterations, 2))
    40 │        │       │      │ 12.0% │ 76 MB │ ▃▁▁▁▁▁▁▁▁  3.1% │       │  distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
   ... │        │       │      │       │       │                 │       │
    52 │        │       │      │ 43.0% │ 76 MB │ ▁▁▁▁▁▁▁▁▁  2.1% │       │  normal_data = rng.normal(loc=0, scale=1, size=sample_size)
    53 │        │       │      │ 36.1% │ 76 MB │ ▂▁▁▁▁▁▁▁▁  2.1% │       │  exponential_data = rng.exponential(scale=2, size=sample_size)
   ... │        │       │      │       │       │                 │       │
    59 │        │       │      │  6.4% │ 76 MB │ ▂▁▁▁▁▁▁▁▁  2.1% │       │      "std": float(np.std(normal_data)),
    60 │        │       │      │  9.3% │ 76 MB │ ▃▁▁▁▁▁▁▁▁  2.1% │    13 │      "median": float(np.median(normal_data)),
   ... │        │       │      │       │       │                 │       │
    62 │        │       │      │ 14.7% │ 77 MB │ ▃▁▁▁▁▁▁▁▁  2.1% │     8 │        "25": float(np.percentile(normal_data, 25)),
    63 │        │       │      │ 13.1% │ 76 MB │ ▃▁▁▁▁▁▁▁▁  2.1% │     8 │        "50": float(np.percentile(normal_data, 50)),
   ... │        │       │      │       │       │                 │       │
    77 │        │       │      │ 14.5% │ 15 MB │                 │       │  correlation = np.corrcoef(combined.T)
   ... │        │       │      │       │       │                 │       │
    88 │        │       │      │ 10.4% │ 76 MB │ ▁▃▁▁▁▁▁▁▁  2.1% │    13 │  t = np.linspace(0, 10, signal_length)
   ... │        │       │      │       │       │                 │       │
    90 │        │       │      │ 20.4% │153 MB │ ▁▂▁▁▁▁▁▁▁  4.2% │       │    np.sin(2 * np.pi * 5 * t)
   ... │        │       │      │       │       │                 │       │
    97 │        │  1.0% │ 1.3% │       │611 MB │ ▁▅▁▁▁▁▁▁▁ 16.8% │       │  fft_result = np.fft.fft(signal)
    98 │        │       │      │  1.1% │229 MB │ ▁▅▁▁▁▁▁▁▁  6.3% │       │  frequencies = np.fft.fftfreq(signal_length, d=0.001)
   ... │        │       │      │       │       │                 │       │
   102 │        │ 10.7% │      │       │ 76 MB │ ▁▁▅▁▁▁▁▁▁  2.1% │       │  top_indices = np.argsort(power)[-5:]
   ... │        │       │      │       │       │                 │       │
   130 │        │       │      │ 82.9% │ 80 MB │ ▁▁▂▁▁▁▁▁▁  2.2% │    14 │      "matrix": rng.uniform(0, 100, size=(50, 50)).tolist(),
   ... │        │       │      │       │       │                 │       │
   140 │  35.0% │  7.8% │      │ 99.9% │196 MB │ ▁▁▁▂▂▂▂▂▂ 13.0% │    28 │    json_str = json.dumps(complex_data)
   141 │   1.1% │ 23.3% │      │100.0% │390 MB │ ▁▁▁▁▂▂▂▃▃ 10.7% │    64 │    _ = json.loads(json_str)
╶──────┼────────┼───────┼──────┼───────┼───────┼─────────────────┼───────┼────────────────────────────────────────────────────────────────
       │        │       │      │       │       │                 │       │ function summary for tests/scripts/advanced.py
    14 │        │  1.2% │      │ 44.3% │ 36 MB │ ▁▁▁▁▁▁▁▁▁  2.2% │     7 │matrix_operations
    35 │        │       │      │  8.3% │ 76 MB │ ▂▁▁▁▁▁▁▁▁  5.2% │       │monte_carlo_simulation
    47 │   1.3% │  4.8% │ 1.1% │ 12.3% │ 77 MB │ ▂▃▁▁▁▁▁▁▁ 21.4% │    45 │statistical_analysis
    83 │        │ 10.5% │ 4.5% │  2.9% │611 MB │ ▁▄▅▁▁▁▁▁▁ 44.0% │    13 │fourier_analysis
   121 │  37.0% │ 31.5% │ 1.3% │ 98.5% │390 MB │ ▁▁▂▂▂▂▂▂▃ 25.9% │   106 │data_serialization
Top PEAK memory consumption, by line:
(1)    97:   610 MB
(2)   141:   390 MB
(3)    98:   228 MB
(4)   140:   196 MB
(5)    92:   152 MB
Possible memory leak identified at line 141 (estimated likelihood:  96%, velocity:   4 MB/s)

Stack Trace Summary (9.759s total)
==================================================================================

data_serialization, (68.48%)
  ├─ tests/scripts/advanced.py:180; main
  │  └─ tests/scripts/advanced.py:190; <module>
  │     n_calls: 342
  │
  └─ tests/scripts/advanced.py:133; data_serialization
     └─ tests/scripts/advanced.py:180; main
        └─ tests/scripts/advanced.py:190; <module>
           n_calls: 3


fourier_analysis, (11.47%)
  └─ tests/scripts/advanced.py:191; main
     └─ tests/scripts/advanced.py:190; <module>
        n_calls: 10


statistical_analysis, (6.07%)
  └─ tests/scripts/advanced.py:165; main
     └─ tests/scripts/advanced.py:190; <module>
        n_calls: 16


matrix_operations, (1.74%)
  └─ tests/scripts/advanced.py:155; main
     └─ tests/scripts/advanced.py:190; <module>
        n_calls: 4


monte_carlo_simulation, (0.57%)
  └─ tests/scripts/advanced.py:160; main
     └─ tests/scripts/advanced.py:190; <module>
        n_calls: 2
```

While the upper portion of the table shows the most resource intensive lines of code,
the lower part of the  table shows the functions in that file which are the most
resource intensive. Finally, below the table is a summary of the average and peak
memory consumption by line in the file. There may also be a warning identifying a
possible memory leak, this may be useful, but the feature is currently marked as
experimental.

