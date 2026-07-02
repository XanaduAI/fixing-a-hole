"""Script for testing profiler behaviour with multi-threaded parallel workloads."""

import logging
import threading
from sys import argv

import numpy as np

MAT_SIZE: int = 2_500
SAMPLE_SIZE: int = 10_000_000
logger = logging.getLogger("__testing__")
logger.addHandler(logging.StreamHandler())


def matrix_worker(results: list, index: int, size: int = MAT_SIZE) -> None:
    """Perform matrix multiplication in a thread and store the result."""
    rng = np.random.default_rng()
    logger.info("Thread %d: starting matrix operations (size=%d)", index, size)
    a = rng.uniform(-10, 10, size=(size, size))
    b = rng.uniform(-10, 10, size=(size, size))
    results[index] = np.matmul(a, b)
    logger.info("Thread %d: finished matrix operations", index)


def monte_carlo_worker(results: list, index: int, iterations: int = SAMPLE_SIZE) -> None:
    """Estimate pi via Monte Carlo in a thread and store the result."""
    rng = np.random.default_rng()
    logger.info("Thread %d: starting Monte Carlo (%d iterations)", index, iterations)
    points = rng.uniform(-1, 1, size=(iterations, 2))
    inside = np.sum(np.linalg.norm(points, axis=1) <= 1)
    results[index] = 4 * inside / iterations
    logger.info("Thread %d: pi estimate = %.6f", index, results[index])


def statistical_worker(results: list, index: int, sample_size: int = SAMPLE_SIZE) -> None:
    """Compute basic statistics on random data in a thread."""
    rng = np.random.default_rng()
    logger.info("Thread %d: starting statistical analysis (%d samples)", index, sample_size)
    data = rng.normal(loc=0, scale=1, size=sample_size)
    results[index] = {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
    }
    logger.info("Thread %d: finished statistical analysis", index)


def main() -> None:
    """Run multiple computational tasks concurrently using threads, or serially with ``--serial``."""
    serial = "--serial" in argv
    logger.info("Starting multi-threaded workload (serial=%s). Args: %s", serial, " ".join(argv[1:]))

    n_threads = 4
    results: list = [None] * n_threads

    tasks = [
        threading.Thread(target=matrix_worker, args=(results, 0, MAT_SIZE)),
        threading.Thread(target=monte_carlo_worker, args=(results, 1, SAMPLE_SIZE)),
        threading.Thread(target=statistical_worker, args=(results, 2, SAMPLE_SIZE)),
        threading.Thread(target=monte_carlo_worker, args=(results, 3, SAMPLE_SIZE)),
    ]

    if serial:
        for task in tasks:
            task.run()
    else:
        for t in tasks:
            t.start()
        for t in tasks:
            t.join()

    logger.info("All tasks finished. Results collected: %d", sum(r is not None for r in results))


if __name__ == "__main__":
    main()
