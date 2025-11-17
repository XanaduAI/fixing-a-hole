# Fixing-A-Hole Profiler

Profiling is the process of analyzing the resource usage of code to identify
bottlenecks and potential areas of optimization and improvement. Typical types
of resources used by code are CPU time, memory usage, and disk space. It can be
the case that improvements in one area come at the expense of costs in another.

Fixing-A-Hole uses the [scalene](https://github.com/plasma-umass/scalene) profiler which,
unfortunately, has very limited support on Windows, even for single-threaded CPU usage.
Scalene supports (single- and multi-threaded) CPU and (peak heap) memory usage on macOS,
Linux, and WSL (Windows Subsystem for Linux).


> [!TIP]
> _"premature optimization is the root of all evil"_ - [Donald Knuth](https://en.wikipedia.org/wiki/Donald_Knuth)

## Usage

### Scripts and Notebooks

Python scripts `.py` and notebooks `.ipynb` can be profiled using
```bash
fixit profile <filename>
```
For example, if you're working on a new method in `ft-stack/my_work/my_new_method.ipynb`
then you can profile it using
```bash
fixit profile my_new_method.ipynb
```
and so long as `my_new_method.ipynb` is the only file in the repo with that name, it
will run it. Otherwise, you will need to be more specific by calling
`fixit profile my_work/my_new_method.ipynb`.
You can also always specify the absolute path to the script.

### MC Simulations

MC simulations can be profiled when run locally. When creating the input generator,
simply indicate `profile_kwargs.profile = true`. Logging (see below) is not available
while profiling from an input generator.

## Options

To see all the available options for the Fixing-A-Hole profiler, run
```bash
fixit profile --help
```

Additional information for each option can also be found below.
```bash
--cpu/--memory
```
The main options are `--cpu` vs `--memory`. By default, Fixing-A-Hole will profile the
RSS memory usage of the script/experiment. _However_, additional CPU overhead is
required in order to determine the _heap_ memory usage of the script. The slowdown varies
depending on the script, but may be as low as 1.2x to as much as 4x or more.
Again, it really depends on the script itself.

> [!TIP]
> It's likely (and recommended) that you have run your script or notebook normally
> before you profile it. Even the fastest code is useless if it doesn't solve the
> problem. However, if you're concerned with the overhead of memory sampling,
> run a `--cpu` test first to establish an expectation on how long you may need
> to wait when using `--memory`.

```bash
--precision=<n>
```
It is possible to alter the memory sampling overhead using the `--precision` flag.
By default, [scalene](https://github.com/plasma-umass/scalene) will highlight lines
of code that allocate more than about 10 MB of memory. This can be modified to be as
verbose as about 325 kB (by setting `--precision=5`) or as vague as about 325 MB
(by setting `--precision=-5`). The higher the level of precision (`≤5`) the slower
the profiling might take as more samples are needed. However, setting the level of
precision too low (`≥-5`) _may_ result in an uninformed summary. You will need to
find the right balance for the level of profiling that you are doing. Again, the
speed depends on the script itself.

```bash
--detailed
```
By default, the profiler will only report CPU and memory usage within Fixing-A-Hole.
However, if you would also like a report on the usage by imported modules,
such as `scipy`, `numpy`, etc., then use the `--detailed` flag.

```bash
--log-level
```
By default, Fixing-A-Hole will capture warnings while profiling scripts and save them to a
log file. More or less detailed capture can be specified using the `--log-level` flag.
The options are: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Each level will
capture that level of severity _and higher_. So the default capturing `WARNING` will
also capture `ERROR` and `CRITICAL`. However, if you have a syntax error or something,
your code will still crash, not run, and throw errors in the terminal.

```bash
--no-plots
```
By default, if your script or notebook generates plots, then Fixing-A-Hole will profile that
too. The downside of this is that if a plot is opened and you take 5 seconds to close
it, those 5 seconds will count towards how long it took your code to run. If you would
like to temporarily disable generating plots, you can profile your code with the
`--no-plots` flag. This will temporarily prevent your code from generating plots without
modifying your code.

## Results

Results generated from `fixit profile` are saved in the `.scratch/performance/` directory
of the Fixing-A-Hole repo. As such, they are not tracked by `git`. Each script or notebook that
you profile in this way are saved by name and the UTC datetime when you ran the profile.
For example, `/my_work/my_new_method.ipynb` will be saved in
`/performance/my_new_method/20250639_123456/`. Within the folder will be a copy of the
code that was profiled along with the logs and profile results.

Results generated for a local job (for example, using `fpfill`), are saved in the job
directory created for the job. They are saved with a default name, or with the given name.

See below for a portion of an example profile for
`doc/tutorials/run_basics_qec.py` and how to interpret it.

### `profile_results.txt`

The first line in the results file is the command used to generate the results (if it
was called using `fixit`; otherwise, it was called for a specific job and is blank).
The second line shows the runtime and max memory usage.
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
$ fixit profile run_basics_qec.py
Finished in 11.847 seconds using 150.601 MB of RAM.
Check logs .scratch/performance/run_basics_qec/20250717_183538/logs.log (1 warning)


Profile Summary
=====================================================================

Top 4 Functions by Total Runtime:
---------------------------------------------------------------------
 1. CSSCode.symplectic_product     37.0% (ops.py)
 2. PauliOperator.to_binary         6.0% (ops.py)
 3. CSSCode.symp_gram_schmidt       6.0% (stabilizer_code.py)
 4. RepetitionCode                  4.0% (code_library.py)

Top 1 Functions by Memory Usage:
---------------------------------------------------------------------
 1. PauliOperator.to_binary          10 MB (ops.py)

Functions by Module:
---------------------------------------------------------------------
└─ flamingpy (4 func, 53.0% total)
   ├─ dv (2 func, 43.0% total)
   │  └─ ops.py (2 func, 43.0% total)
   │     ├─ CSSCode.symplectic_product....37.0%
   │     └─ PauliOperator.to_binary........6.0% (10 MB)
   │
   └─ code (2 func, 10.0% total)
      ├─ stabilizer_code.py (1 func, 6.0% total)
      │  └─ CSSCode.symp_gram_schmidt......6.0%
      │
      └─ code_library.py (1 func, 4.0% total)
         └─ RepetitionCode.................4.0%


=====================================================================

        Memory usage: ▁▂▂▃▃▄▄▅▅▆▆▇▇██ (max: 150.601 MB, growth rate: 100%)
   /home/ubuntu/ft-stack/flamingpy/dv/ops.py: % of time =  44.58% (5.281s) out of 11.847s.
       ╷       ╷       ╷       ╷       ╷     ╷          ╷       ╷
       │Time   │–––––– │–––––– │Memory │–––––│––––––––––│Copy   │
  Line │Python │native │system │Python │peak │timeline/%│(MB/s) │/home/ubuntu/ft-stack/flamingpy/dv/ops.py
╺━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━━━┿━━━━━┿━━━━━━━━━━┿━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ... │       │       │       │       │     │          │       │
   224 │    3% │       │       │       │     │          │     3 │    x1_stab = ops1[:, : shape[1] // 2]
   225 │    2% │       │       │       │     │          │       │    x2_stab = ops2[:, : shape[1] // 2]
   226 │    2% │       │       │       │     │          │     1 │    z1_stab = ops1[:, shape[1] // 2 :]
   227 │    2% │       │       │       │     │          │     3 │    z2_stab = ops2[:, shape[1] // 2 :]
   228 │   18% │    8% │       │       │     │          │     2 │    symplectic_prod = x1_stab @ z2_stab.T + z1_stab @ x2_stab.T
   ... │       │       │       │       │     │          │       │
   230 │    1% │       │       │       │     │          │     8 │    return symplectic_prod.astype(np.int8)
   ... │       │       │       │       │     │          │       │
   296 │    5% │    1% │       │  99%  │ 10M │▁   7%    │    11 │        ops = csr_array(ops, dtype=np.int8)
   ... │       │       │       │       │     │          │       │
       │       │       │       │       │     │          │       │
╶──────┼───────┼───────┼───────┼───────┼─────┼──────────┼───────┼─────────────────────────────────────────────────────────────────
       │       │       │       │       │     │          │       │function summary for /home/ubuntu/ft-stack/flamingpy/dv/o…
   215 │   27% │   10% │       │       │     │          │    18 │CSSCode.symplectic_product
   286 │    5% │    1% │       │  99%  │ 10M │█   7%    │    11 │PauliOperator.to_binary
       ╵       ╵       ╵       ╵       ╵     ╵          ╵       ╵
Top AVERAGE memory consumption, by line:
(1)   296:    10 MB
Top PEAK memory consumption, by line:
(1)   296:    10 MB
Possible memory leak identified at line 296 (estimated likelihood:  96%, velocity:   4 MB/s)
```

While the upper portion of the table shows the most resource intensive lines of code,
the lower part of the  table shows the functions in that file which are the most
resource intensive. Finally, below the table is a summary of the average and peak
memory consumption by line in the file. There may also be a warning identifying a
possible memory leak, this may be useful, but the feature is currently marked as
experimental.

Please reach out to the Software Tools team in the
[#tooling-for-architecture](https://xanaduhq.slack.com/archives/C04HPDUFN15)
slack channel if you have any additional questions.
