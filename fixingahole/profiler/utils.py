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
"""Integrated Scalene Profiler Utils."""

import datetime
import importlib.metadata
from collections.abc import Callable
from enum import Enum
from pathlib import Path, PurePath
from random import choice
from typing import TYPE_CHECKING, overload

from colours import Colour
from rich._spinners import SPINNERS  # noqa: PLC2701
from rich.live import Live
from rich.spinner import Spinner as rich_Spinner
from typer import Exit
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from fixingahole import ROOT_DIR

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver


class LogLevel(Enum):
    """Valid Log Levels to profile."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def should_catch_warnings(self) -> bool:
        """Determine whether or not to catch warnings during profiling."""
        return self.name in {"DEBUG", "INFO", "WARNING"}


def date() -> str:
    """Return the current UTC date and time."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")


def memory_with_units(memory: float, unit: str = "MB", digits: int = 0) -> str:
    """Convert memory float to string with units."""
    byte_prefix = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    memory_bytes = memory * byte_prefix[unit]
    for prefix in ["GB", "MB", "KB"]:
        if memory_bytes >= byte_prefix[prefix]:
            return f"{memory_bytes / byte_prefix[prefix]:>3.{digits}f} {prefix}"
    return f"{memory_bytes:.{digits}f} bytes"


@overload
def find_path(
    pattern: str | Path,
    in_dir: str | Path = "",
    *,
    exclude: list[str | Path] | None = None,
    return_suffix: None = None,
) -> Path: ...


@overload
def find_path(
    pattern: str | Path,
    in_dir: str | Path = "",
    *,
    exclude: list[str | Path] | None = None,
    return_suffix: str,
) -> tuple[Path, list[Path]]: ...


def find_path(
    pattern: str | Path,
    in_dir: str | Path = "",
    *,
    exclude: list[str | Path] | None = None,
    return_suffix: str | None = None,
) -> Path | tuple[Path, list[Path]]:
    """Find files or directories in the repository.

    Args:
        pattern: Name or pattern to search for
        in_dir: Directory to search within
        exclude: List of patterns to exclude from search
        return_suffix: If the pattern is a directory, returns a tuple of (dir_path, files_in_dir_with_suffix)

    Returns:
        Path object for the found item, or tuple of (Path, list[Path]) if return_contents=True

    """
    if Path(pattern).is_absolute() and Path(pattern).exists():
        return Path(pattern).absolute()

    in_dir = (ROOT_DIR / in_dir).resolve()
    exclude = exclude if exclude is not None else []

    # Process exclude patterns
    exclude_resolved = []
    for x in exclude:
        if "*" in str(x):
            exclude_resolved.extend([folder for folder in in_dir.glob(str(x)) if folder.is_dir()])
        else:
            exclude_resolved.append(x)
    exclude_resolved += [".venv", ".git"]
    exclude_resolved = [(ROOT_DIR / folder).resolve() for folder in exclude_resolved]
    options = [
        path
        for path in in_dir.rglob("*")
        if (not any(path.is_relative_to(folder) for folder in exclude_resolved) and PurePath(path).match(str(pattern)))
    ]

    file_or_folder = "folders" if not Path(pattern).suffix else "files"
    match len(options):
        case 1:
            result: Path = options.pop()
            if return_suffix is not None:
                return result, [path for path in result.rglob(f"*{return_suffix}") if "__pycache__" not in str(path)]
            return result
        case 0:
            Colour.print(
                Colour.red(f"No {file_or_folder} in"),
                f"{Colour.purple(in_dir.name)} with name:",
                f"{Colour.green(pattern)} were found.",
            )
            raise Exit(code=1)
        case _:
            Colour.print(
                Colour.red(f"Many {file_or_folder} with name:"),
                f"{Colour.green(pattern)} were found in",
                f"{Colour.purple(in_dir.name)}.",
                "Please be more specific.",
            )
            for path in options:
                Colour.purple.print(path.relative_to(ROOT_DIR))
            raise Exit(code=1)


def installed_modules() -> set[str]:
    """List of all installed module names in the current Python virtual environment."""
    modules: set[str] = set()
    for dist in importlib.metadata.distributions():
        if "Name" in dist.metadata:
            modules.add(str(dist.metadata["Name"]).lower())
        elif not any("egg-info" in str(file) for file in getattr(dist, "files", [])):
            Colour.print(Colour.ORANGE("\nWarning:"), "module missing 'Name' metadata.")
    return modules


class Spinner(Live):
    """An abstraction of rich.spinner.Spinner with Live context."""

    def __init__(self, message: str = "", *, style: str | None = None, speed: float = 1.0) -> None:
        """Abstraction of rich.spinner.Spinner with Live context."""
        spinner = rich_Spinner(choice(sorted(SPINNERS.keys())), message, style=style, speed=speed)
        super().__init__(spinner, refresh_per_second=20)


class FileWatcher:
    """A file system watcher that triggers a callback when a file changes."""

    def __init__(self, file_path: Path, on_change_callback: Callable) -> None:
        """Initialize the file watcher.

        Args:
            file_path: Path to the file to watch
            on_change_callback: Function to call when the file is modified

        """
        self.file_path = Path(file_path)
        self.callback = on_change_callback
        self.observer: BaseObserver | None = None

    def start(self) -> None:
        """Start watching the file for changes."""

        class FileChangeHandler(FileSystemEventHandler):
            """Handler for file system events."""

            def __init__(self, target_path: str, callback: Callable):
                self.target_path = target_path
                self.callback = callback

            def on_modified(self, event: FileSystemEvent) -> None:
                if not event.is_directory and event.src_path == self.target_path:
                    self.callback()

        handler = FileChangeHandler(str(self.file_path), self.callback)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.file_path.parent), recursive=False)
        self.observer.start()

    def stop(self) -> None:
        """Stop watching the file."""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=1.0)
