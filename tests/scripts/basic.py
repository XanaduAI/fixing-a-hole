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
    _ = rng.uniform(size=10**5)


if __name__ == "__main__":
    main()
