# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integrated Scalene Profiler and Parser."""

from importlib.metadata import PackageNotFoundError, version

from colours import Colour as _Colour

from fixingahole.config import DURATION, IGNORE_DIRS, OUTPUT_DIR, ROOT_DIR
from fixingahole.profiler.profile_summary import ProfileSummary
from fixingahole.profiler.profiler import Profiler
from fixingahole.profiler.utils import LogLevel

try:
    __version__ = version("fixingahole")
except PackageNotFoundError:
    __version__ = "unknown"


def about() -> None:
    """About info for Fixing-A-Hole."""
    try:
        scalene_version = version("scalene")
    except PackageNotFoundError:
        scalene_version = "unknown"

    ignored_dirs = "".join(["\n   " + str(path) for path in IGNORE_DIRS])
    _Colour.print(
        _Colour.blue("Fixing-A-Hole: an integrated Scalene profiler and parser."),
        f"\n Version: {__version__}",
        f"\n Scalene Version: {scalene_version}",
        _Colour.blue("\nSettings:"),
        f"\n root: {ROOT_DIR}",
        f"\n output: {OUTPUT_DIR}",
        f"\n ignore: [{ignored_dirs}\n]",
    )


__all__ = [
    "DURATION",
    "IGNORE_DIRS",
    "OUTPUT_DIR",
    "ROOT_DIR",
    "LogLevel",
    "ProfileSummary",
    "Profiler",
]
