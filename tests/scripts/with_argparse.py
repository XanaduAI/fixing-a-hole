"""Basic script for testing how args are parsed."""

import argparse
import logging
from sys import argv

import numpy as np

logger = logging.getLogger("__testing__")
rng = np.random.default_rng()

logger.setLevel(logging.DEBUG)


def main(base: int, exp: int) -> tuple[int,]:
    """Test script with command line args."""
    logger.info(" ".join(argv[1:]))
    logger.warning("This is a warning.")
    rand = rng.uniform(size=base**exp)
    return rand.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scaling law data")
    parser.add_argument("--base", type=int, required=True, help="Base for the number of samples.")
    parser.add_argument("--power", type=int, required=True, help="Exponent for the number of samples")
    args = parser.parse_args()

    base, exp = int(args.base), int(args.power)
    shape, *_ = main(base, exp)
    logger.info("Shape is %s^%s = %s (%s)", base, exp, shape, (base**exp) == shape)
