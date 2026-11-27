# Understanding memory profiling: heap vs RSS

[Back to the usage guide](README.md#results-and-statistics).

`fixing-a-hole` reports Scalene's heap measurements when memory profiling is enabled and
maximum resident set size (RSS) on macOS and Linux. These answer different questions:

| Measurement | Useful question |
| --- | --- |
| Scalene's heap profile | Which code allocates memory, and where does allocation growth occur? |
| Maximum RSS | How large was the observed resident-memory peak during this run? |

Use both to investigate memory usage. Neither measurement alone predicts whether another run
will exceed memory constraints.

## Heap memory

Heap memory is dynamically allocated storage, including Python objects and buffers allocated
by native libraries. Scalene samples allocations and deallocations, attributes them to code,
and distinguishes Python from native allocation activity. See Section 3 of the
[Scalene paper](https://www.usenix.org/system/files/osdi23-berger.pdf) for how this works.

Request memory profiling with a native Scalene argument:

```bash
fixingahole profile script.py --memory
```

### What the heap profile tells you

- Which lines and functions are responsible for allocations.
- Where memory usage grows during execution.
- Which allocations contribute to reported peaks.
- Where to investigate retained objects, temporary copies, or opportunities to process data
  in smaller batches.

### Limitations

An allocation profile is not a complete account of resident process memory. It does not
directly describe all mapped pages, thread stacks, allocator capacity, or operating-system
memory accounting.

Scalene uses sampling, so resolution and profiling overhead depend on the workload and
sampling settings. Smaller values of `--allocation-sampling-window` generally provide finer
resolution at higher overhead. Profiling scope also affects which code appears in reports;
excluded code can still contribute to process RSS.

Heap measurements are useful for locating allocation activity, but they do not establish
how much physical RAM the whole workload requires.

## Resident set size (RSS)

RSS measures process memory currently resident in physical RAM. It includes resident anonymous
memory and resident file or shared-memory mappings. This can include heap pages, stacks,
interpreter code, and shared libraries. See the [Linux memory-accounting documentation](https://docs.kernel.org/filesystems/proc.html).

### What maximum RSS tells you

`fixing-a-hole` reports **maximum RSS**, a peak from operating-system resource accounting for
the subprocess it launches. It is not the RSS at the end of the script or a time series.

Use it to compare resident-memory peaks for similar workloads and to investigate memory usage
that is not explained by the allocation profile.

### Limitations

- RSS does not identify which lines allocated memory.
- Swapped-out pages and untouched virtual allocations are not resident memory.
- Shared pages can appear in multiple processes' RSS; adding their values can double-count
  shared memory. On Linux, proportional set size (PSS) apportions shared pages between processes.
- The reported maximum is not a measurement of the peak sum of every concurrently running
  worker. For multiprocessing, examine process or workload-level accounting separately.
- A peak from one run does not describe future inputs, concurrency, or available system memory.

Linux's [resource-usage documentation](https://man7.org/linux/man-pages/man2/getrusage.2.html)
describes the distinction between a process's maximum RSS and child-process accounting.

## Why heap and RSS differ

Allocation tracking and resident-memory accounting measure different things. Python's memory
manager can preallocate and cache storage, so releasing objects need not immediately reduce
resident memory. See [Python's memory-management documentation](https://docs.python.org/3/c-api/memory.html).

RSS also includes resident mappings beyond tracked heap allocations, and memory residency
can change through paging. The two reported peaks may occur at different times and cover
different parts of execution.

Do not subtract peak heap usage from peak RSS and interpret the difference as an exact measure
of interpreter overhead, fragmentation, or a memory leak.

## Why neither metric predicts out-of-memory errors

An out-of-memory (OOM) failure depends on the environment and allocation request, as well as
the workload's memory usage:

1. **Virtual memory and overcommit.** A successful allocation may reserve address space before
   all its pages become resident. Linux supports different overcommit policies, including
   rejecting allocations based on committed memory. See [overcommit accounting](https://www.kernel.org/doc/html/latest/mm/overcommit-accounting.html).
2. **Swap and reclaim.** Available swap and reclaimable pages influence what the operating
   system can satisfy. RSS alone does not describe those resources.
3. **Process and workload limits.** Allocations can fail because of address-space or data limits.
   A container or cgroup can encounter its own memory limit while the host still has free RAM.
   See [allocation failures](https://man7.org/linux/man-pages/man3/malloc.3.html) and
   [Linux cgroup memory limits](https://docs.kernel.org/admin-guide/cgroup-v2.html).
4. **Shared memory and concurrent processes.** Per-process peaks do not describe the complete
   workload's simultaneous demand or competition from other applications.
5. **Allocation layout.** Fragmentation and the size or layout of a requested allocation can
   affect whether it can be satisfied.
6. **Changing workloads.** Larger inputs, additional workers, temporary copies, and different
   execution paths can produce peaks that did not occur in the profiled run.

Free RAM, a low reported heap peak, or a low maximum RSS therefore does not guarantee that
the next allocation or a larger workload will succeed.

## Practical investigation

- Use the heap profile to identify allocation-heavy code and retained data.
- Compare maximum RSS across runs with comparable inputs and settings.
- Test representative inputs and realistic concurrency in the intended deployment environment.
- For memory failures, inspect the actual process or workload limits, swap configuration,
  memory pressure, and operating-system diagnostics.
- For multiprocessing, monitor the workload as a group rather than summing individual peaks.
- Consider streaming or batching when holding the full input or intermediate results is costly.
- Test how memory usage scales with larger inputs and choose headroom from those observations,
  rather than assuming a fixed multiplier guarantees safety.

Record input sizes, worker counts, profiling arguments, and relevant memory limits alongside
results so comparisons have a consistent context.
