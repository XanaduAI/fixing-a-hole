"""Advanced script for testing with multiple computational tasks."""

import json
import logging
import time
from sys import argv

import numpy as np

logger = logging.getLogger("__testing__")
rng = np.random.default_rng()


def matrix_operations(size: int = 2000) -> np.ndarray:
    """Perform intensive matrix operations."""
    logger.info("Starting matrix operations with size %sx%s", size, size)

    # Create large random matrices
    matrix_a = rng.uniform(-10, 10, size=(size, size))
    matrix_b = rng.uniform(-10, 10, size=(size, size))

    # Matrix multiplication
    result = np.matmul(matrix_a, matrix_b)

    # Eigenvalue computation (expensive)
    smaller_matrix = result[:500, :500]
    eigenvalues = np.linalg.eigvals(smaller_matrix)

    # SVD decomposition
    np.linalg.svd(smaller_matrix)

    return eigenvalues


def monte_carlo_simulation(iterations: int = 5_000_000) -> dict:
    """Run Monte Carlo simulation for pi estimation."""
    logger.info("Running Monte Carlo simulation with %s iterations", iterations)

    points = rng.uniform(-1, 1, size=(iterations, 2))
    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    inside_circle = np.sum(distances <= 1)
    pi_estimate = 4 * inside_circle / iterations

    return {"iterations": iterations, "pi_estimate": pi_estimate, "error": abs(pi_estimate - np.pi)}


def statistical_analysis(sample_size: int = 10_000_000) -> dict:
    """Perform various statistical computations."""
    logger.info("Performing statistical analysis on %s samples", sample_size)

    # Generate different distributions
    normal_data = rng.normal(loc=0, scale=1, size=sample_size)
    exponential_data = rng.exponential(scale=2, size=sample_size)

    # Compute statistics
    stats = {
        "normal": {
            "mean": float(np.mean(normal_data)),
            "std": float(np.std(normal_data)),
            "median": float(np.median(normal_data)),
            "percentiles": {
                "25": float(np.percentile(normal_data, 25)),
                "50": float(np.percentile(normal_data, 50)),
                "75": float(np.percentile(normal_data, 75)),
                "95": float(np.percentile(normal_data, 95)),
            },
        },
        "exponential": {
            "mean": float(np.mean(exponential_data)),
            "std": float(np.std(exponential_data)),
            "median": float(np.median(exponential_data)),
        },
    }

    # Correlation between datasets
    combined = np.column_stack([normal_data[:1000000], exponential_data[:1000000]])
    correlation = np.corrcoef(combined.T)
    stats["correlation"] = correlation.tolist()

    return stats


def fourier_analysis(signal_length: int = 10_000_000) -> dict:
    """Perform Fourier transform analysis."""
    logger.info("Performing Fourier analysis on signal of length %s", signal_length)

    # Create a complex signal with multiple frequencies
    t = np.linspace(0, 10, signal_length)
    signal = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 10 * t)
        + 0.3 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 0.1, signal_length)
    )

    # Compute FFT (computationally expensive)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal_length, d=0.001)

    # Find dominant frequencies
    power = np.abs(fft_result) ** 2
    top_indices = np.argsort(power)[-5:]
    dominant_freqs = frequencies[top_indices]

    return {
        "signal_length": signal_length,
        "dominant_frequencies": dominant_freqs.tolist(),
        "total_power": float(np.sum(power)),
    }


def recursive_computation(depth: int = 30, width: int = 100000) -> float:
    """Perform recursive computations with arrays."""
    if depth == 0:
        return float(np.sum(rng.uniform(0, 1, size=width)))

    result = rng.uniform(0, 1, size=width).mean()
    return result + recursive_computation(depth - 1, width) * 0.9


def data_serialization(data_size: int = 1000) -> None:
    """Test JSON serialization with complex nested structures."""
    logger.info("Performing data serialization with %s objects", data_size)

    complex_data = []
    for i in range(data_size):
        entry = {
            "id": i,
            "timestamp": time.time(),
            "matrix": rng.uniform(0, 100, size=(50, 50)).tolist(),
            "metadata": {
                "nested_array": rng.integers(0, 1000, size=100).tolist(),
                "properties": {"value_" + str(j): float(rng.uniform()) for j in range(20)},
            },
        }
        complex_data.append(entry)

    # Serialize multiple times
    for _ in range(5):
        json_str = json.dumps(complex_data)
        _ = json.loads(json_str)


def main() -> None:
    """Complex test function with multiple computational tasks."""
    start_time = time.time()

    logger.info(" ".join(argv[1:]))
    logger.info("Starting advanced profiling test...")

    results = {}

    # Task 1: Matrix operations
    logger.warning("Task 1: Matrix operations")
    eigenvalues = matrix_operations(size=2000)
    results["matrix_eigenvalues_sample"] = eigenvalues[:5].tolist()

    # Task 2: Monte Carlo simulation
    logger.warning("Task 2: Monte Carlo simulation")
    mc_results = monte_carlo_simulation(iterations=5_000_000)
    results["monte_carlo"] = mc_results

    # Task 3: Statistical analysis
    logger.warning("Task 3: Statistical analysis")
    stats = statistical_analysis(sample_size=10_000_000)
    results["statistics"] = stats

    # Task 4: Fourier analysis
    logger.warning("Task 4: Fourier analysis")
    fourier_results = fourier_analysis(signal_length=10_000_000)
    results["fourier"] = fourier_results

    # Task 5: Recursive computation
    logger.warning("Task 5: Recursive computation")
    recursive_result = recursive_computation(depth=30, width=100000)
    results["recursive_sum"] = recursive_result

    # Task 6: Data serialization
    logger.warning("Task 6: Data serialization")
    data_serialization(data_size=1000)

    # Final summary
    elapsed = time.time() - start_time
    results["total_elapsed_seconds"] = elapsed

    logger.info("All tasks completed in %s seconds", elapsed)
    logger.info("Results summary: %s metrics computed", len(results))


main()
