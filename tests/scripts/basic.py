"""Basic script for testing."""

import logging
from sys import argv
from time import sleep

import numpy as np

logger = logging.getLogger("__testing__")
rng = np.random.default_rng()


def main() -> None:
    """Test basic function."""
    logger.info(" ".join(argv[1:]))
    logger.warning("This is a warning.")
    sleep(0.5)
    for i in range(7):
        for _ in range(25):
            a: np.ndarray = rng.uniform(size=(10, 10**i))
            del a


if __name__ == "__main__":
    main()
