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
"""Integrated Scalene Profiler Config."""

import os
import sys
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

from colours import Colour


def _detect_virtualenv() -> str:
    """Find the virtual environment path for the current Python executable."""
    # Adapted from https://github.com/astral-sh/uv/blob/44f5a14/python/uv/__main__.py#L7-L23
    # If the VIRTUAL_ENV variable is already set, then just use it.
    value = os.getenv("VIRTUAL_ENV")
    if value:
        return value

    # Otherwise, check if we're in a venv
    venv_marker = Path(sys.prefix) / "pyvenv.cfg"
    if venv_marker.exists():
        return sys.prefix

    return ""


def _find_pyproject() -> Path | None:
    """Search for pyproject.toml starting from cwd and walking up the directory tree."""
    current = Path.cwd()
    while current != current.parent:
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            return pyproject
        current = current.parent
    # Check root directory
    pyproject = current / "pyproject.toml"
    if pyproject.exists():
        return pyproject
    # Check parent directory of virtual environment
    venv_path = _detect_virtualenv()
    if venv_path:
        venv_parent = Path(venv_path).parent
        pyproject = venv_parent / "pyproject.toml"
        if pyproject.exists():
            return pyproject
    # Cannot find pyproject.toml
    return None


@cache
def _get_config() -> dict[str, Any]:
    """Get the Fixing-A-Hole config from pyproject.toml."""
    pyproject_path = _find_pyproject()
    if pyproject_path is None:
        return {}

    with Path.open(pyproject_path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            Colour.print(
                Colour.RED("Error:"),
                f"{exc} while reading",
                Colour.purple(Path(*(pyproject_path.parts[-2:]))),
            )
            sys.exit(5)  # Failure to read or write data.
    tools = data.get("tool", {})
    return tools.get("fixingahole", {})


def _get_root_dir(config: dict[str, Any]) -> Path:
    """Get the ROOT_DIR constant."""
    return Path(config.get("root", Path.cwd())).resolve()


def _get_output_dir(config: dict[str, Any]) -> Path:
    """Get the OUTPUT_DIR constant.

    If the given path is absolute then use it, otherwise assume it's relative to ROOT_DIR.
    """
    output_path = Path(config.get("output", "performance"))
    if output_path.is_absolute():
        return output_path
    return (ROOT_DIR / output_path).resolve()


def _get_ignore_directories(config: dict[str, Any], output_path: Path) -> list[Path]:
    """Get the list of directories to ignore when searching for files."""
    ignore_dirs: str | list[str] = config.get("ignore", [output_path])
    if not isinstance(ignore_dirs, list):
        if isinstance(ignore_dirs, str):
            ignore_dirs: list[Path] = [Path(ignore_dirs)]
        else:
            return [output_path]
    ignore_dirs: list[Path] = [p for path in ignore_dirs if (p := Path(path).resolve()).is_dir()]
    if output_path not in ignore_dirs:
        ignore_dirs.append(output_path)
    return ignore_dirs


ROOT_DIR = _get_root_dir(_get_config())
OUTPUT_DIR = _get_output_dir(_get_config())
IGNORE_DIRS = _get_ignore_directories(_get_config(), OUTPUT_DIR)
