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
from pathlib import Path


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


def _get_root_dir() -> Path:
    """Get the ROOT_DIR constant."""
    pyproject_path = _find_pyproject()
    if pyproject_path is None:
        return Path.cwd()

    with Path.open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    tools = data.get("tool", {})
    config = tools.get("fixingahole", {})
    return Path(config.get("root", ".")).resolve()


ROOT_DIR = _get_root_dir()
