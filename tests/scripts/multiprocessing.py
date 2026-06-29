"""Script for testing profiler behaviour with multi-process parallel workloads."""

import logging
import multiprocessing as mp
from sys import argv

import numpy as np

MAT_SIZE: int = 2_500
SAMPLE_SIZE: int = 10_000_000
logger = logging.getLogger("__testing__")
logger.addHandler(logging.StreamHandler())


def matrix_worker(size: int = MAT_SIZE) -> float:
    """Perform matrix multiplication in a subprocess and return the Frobenius norm."""
    rng = np.random.default_rng()
    a = rng.uniform(-10, 10, size=(size, size))
    b = rng.uniform(-10, 10, size=(size, size))
    result = np.matmul(a, b)
    return float(np.linalg.norm(result, "fro"))


def monte_carlo_worker(iterations: int = SAMPLE_SIZE) -> float:
    """Estimate pi via Monte Carlo in a subprocess."""
    rng = np.random.default_rng()
    points = rng.uniform(-1, 1, size=(iterations, 2))
    inside = np.sum(np.linalg.norm(points, axis=1) <= 1)
    return 4 * inside / iterations


def statistical_worker(sample_size: int = SAMPLE_SIZE) -> dict:
    """Compute basic statistics on random data in a subprocess."""
    rng = np.random.default_rng()
    data = rng.normal(loc=0, scale=1, size=sample_size)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
    }


def main() -> None:
    """Run multiple computational tasks concurrently using processes, or serially with ``--serial``."""
    serial = "--serial" in argv
    logger.info("Starting multi-process workload (serial=%s). Args: %s", serial, " ".join(argv[1:]))

    tasks = [
        (matrix_worker, (MAT_SIZE,)),
        (monte_carlo_worker, (SAMPLE_SIZE,)),
        (statistical_worker, (SAMPLE_SIZE,)),
        (monte_carlo_worker, (SAMPLE_SIZE,)),
    ]

    if serial:
        results = [fn(*args) for fn, args in tasks]
    else:
        with mp.Pool(processes=len(tasks)) as pool:
            async_results = [pool.apply_async(fn, args) for fn, args in tasks]
            results = [r.get() for r in async_results]

    logger.info("All processes finished. Results collected: %d", len(results))
    logger.info("Matrix Frobenius norm: %.4f", results[0])
    logger.info("Pi estimates: %.6f, %.6f", results[1], results[3])
    logger.info("Stats: %s", results[2])


if __name__ == "__main__":
    main()
