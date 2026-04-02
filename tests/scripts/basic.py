"""Basic script for testing."""

import logging
from sys import argv
from time import sleep

import numpy as np

logger = logging.getLogger("__testing__")
logger.addHandler(logging.StreamHandler())
rng = np.random.default_rng()


def main() -> None:
    """Test basic function."""
    logger.info("This is info: %s", " ".join(argv[1:]))
    logger.warning("This is a warning.")
    logger.error("This is an error.")
    logger.critical("This is critical.")

    slp = 0.5
    logger.info("Starting sleep for %s s.", slp)
    sleep(slp)
    logger.info("Slept for %s s.", slp)
    for i in range(7):
        for _ in range(25):
            a: np.ndarray = rng.uniform(size=(10, 10**i))
            logger.debug("This is a debug log: shape%s", a.shape)
            del a


if __name__ == "__main__":
    main()
