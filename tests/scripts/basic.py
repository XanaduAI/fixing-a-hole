"""Basic script for testing."""

import logging
from sys import argv

import numpy as np

logger = logging.getLogger("__testing__")
rng = np.random.default_rng()


def main() -> None:
    """Test basic function."""
    logger.info(" ".join(argv[1:]))
    logger.warning("This is a warning.")
    _ = rng.uniform(size=10**5)


main()
