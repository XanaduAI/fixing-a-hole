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
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from colours import Colour


class DurationOption(Enum):
    """Duration display options."""

    absolute = "absolute"  # ex. 4.32 seconds
    relative = "relative"  # ex. 73.2%


class ConfigError(Exception):
    """Base exception for configuration errors."""


class ConfigParseError(ConfigError):
    """Raised when config cannot be parsed."""


class ConfigValueError(ConfigError):
    """Raised when a config value is invalid."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_virtualenv() -> str:
    """Return the active virtual environment path, or an empty string.

    Checks the ``VIRTUAL_ENV`` environment variable first, then falls back to
    inspecting ``sys.prefix`` for a ``pyvenv.cfg`` marker.

    Adapted from https://github.com/astral-sh/uv/blob/44f5a14/python/uv/__main__.py#L7-L23
    """
    if value := os.getenv("VIRTUAL_ENV"):
        return value
    if (Path(sys.prefix) / "pyvenv.cfg").exists():
        return sys.prefix
    return ""


def _find_pyproject() -> Path | None:
    """Search for ``pyproject.toml`` starting from ``cwd``, walking up the tree.

    Falls back to the parent directory of the active virtual environment when the
    normal walk reaches the filesystem root without a match.
    """
    current = Path.cwd()
    while True:
        candidate = current / "pyproject.toml"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: parent directory of the virtual environment
    if venv_path := _detect_virtualenv():
        candidate = Path(venv_path).parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _load_pyproject_config() -> tuple[dict[str, Any], Path]:
    """Load ``[tool.fixingahole]`` from ``pyproject.toml`` and return ``(config, base_dir)``.

    Returns an empty config dict and ``cwd`` when no ``pyproject.toml`` is found.

    Raises:
        ConfigParseError: When the discovered ``pyproject.toml`` cannot be parsed.

    """
    pyproject_path = _find_pyproject()
    if pyproject_path is None:
        return {}, Path.cwd()
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        msg = f"{exc} while reading {pyproject_path}"
        raise ConfigParseError(msg) from exc
    return data.get("tool", {}).get("fixingahole", {}), pyproject_path.parent


def _resolve_root_dir(config: dict[str, Any], base_dir: Path) -> Path:
    """Resolve the root directory, relative to *base_dir* when the path is not absolute."""
    root = Path(config.get("root", base_dir))
    return root if root.is_absolute() else (base_dir / root).resolve()


def _resolve_output_dir(config: dict[str, Any], root_dir: Path) -> Path:
    """Resolve the output directory, relative to *root_dir* when the path is not absolute."""
    output = Path(config.get("output", "performance"))
    return output if output.is_absolute() else (root_dir / output).resolve()


def _resolve_ignore_dirs(config: dict[str, Any], output_dir: Path, base_dir: Path) -> list[Path]:
    """Resolve ignore directories, always appending *output_dir* if not already present.

    Invalid or non-existent paths are silently dropped.  Non-string, non-list
    values for the ``ignore`` key fall back to ``[output_dir]``.
    """
    raw: Any = config.get("ignore", [])
    if isinstance(raw, str):
        entries: list[str] = [raw]
    elif isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, str)]
    else:
        return [output_dir]

    dirs: list[Path] = [(p if (p := Path(e)).is_absolute() else base_dir / p).resolve() for e in entries]
    dirs = [p for p in dirs if p.is_dir()]
    if output_dir not in dirs:
        dirs.append(output_dir)
    return dirs


def _build_settings(config_data: dict[str, Any], base_dir: Path) -> "Settings":
    """Build a :class:`Settings` instance from raw config data and a base directory.

    Raises:
        ConfigValueError: When ``duration`` is not a valid :class:`DurationOption`.

    """
    root = _resolve_root_dir(config_data, base_dir)
    output = _resolve_output_dir(config_data, root)
    ignore = _resolve_ignore_dirs(config_data, output, base_dir)
    try:
        duration = DurationOption(config_data.get("duration", "relative"))
    except ValueError as err:
        msg = f"{err}. Choose either 'relative' or 'absolute'. Check 'tool.fixingahole' in pyproject.toml"
        raise ConfigValueError(msg) from err
    return Settings(root=root, output=output, ignore=ignore, duration=duration)


def _env_config_data() -> dict[str, Any]:
    """Return raw config data from ``FIXINGAHOLE_*`` environment variables.

    Path values for ``root`` and ``ignore`` are pre-resolved against ``cwd`` so
    they remain correct when merged with pyproject-sourced config that uses a
    different ``base_dir``.
    """
    data: dict[str, Any] = {}
    cwd = Path.cwd()

    if (val := os.getenv("FIXINGAHOLE_ROOT")) is not None:
        p = Path(val)
        data["root"] = str((p if (p := Path(val)).is_absolute() else (cwd / p)).resolve())
    if (val := os.getenv("FIXINGAHOLE_OUTPUT")) is not None:
        data["output"] = val  # relative to root_dir; handled by _resolve_output_dir
    if (val := os.getenv("FIXINGAHOLE_IGNORE")) is not None:
        data["ignore"] = [
            str((p if (p := Path(entry)).is_absolute() else (cwd / p)).resolve())
            for value in val.split(",")
            if (entry := value.strip())
        ]
    if (val := os.getenv("FIXINGAHOLE_DURATION")) is not None:
        data["duration"] = val

    return data


# ---------------------------------------------------------------------------
# Public Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Resolved Fixing-A-Hole settings."""

    root: Path
    output: Path
    ignore: list[Path]
    duration: DurationOption

    def __post_init__(self) -> None:
        """Validate the correct types on creation."""
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "output", Path(self.root / self.output).resolve())
        object.__setattr__(self, "ignore", [(Path.cwd() / path).resolve() for path in self.ignore])
        object.__setattr__(self, "duration", DurationOption(self.duration))

    @classmethod
    def defaults(cls) -> "Settings":
        """Return default settings when no external configuration is provided."""
        root = Path.cwd().resolve()
        output = (root / "performance").resolve()
        return cls(
            root=root,
            output=output,
            ignore=[output],
            duration=DurationOption.relative,
        )


class Config:
    """Active configuration accessor.

    All settings are exposed as classmethods so callers always read the
    current value even after :meth:`configure` is called again at runtime.
    Call :meth:`configure` to update settings.
    """

    _settings: Settings = Settings.defaults()

    def __new__(cls) -> "Config":
        """Prevent instantiation — :class:`Config` is a class-only namespace."""
        msg = f"{cls.__name__} cannot be instantiated."
        raise TypeError(msg)

    @classmethod
    def settings(cls) -> Settings:
        """Return the currently active :class:`Settings` snapshot."""
        return cls._settings

    @classmethod
    def root(cls) -> Path:
        """Return the configured root directory."""
        return cls._settings.root

    @classmethod
    def output(cls) -> Path:
        """Return the configured output directory."""
        return cls._settings.output

    @classmethod
    def ignore(cls) -> list[Path]:
        """Return the configured ignore directories."""
        return cls._settings.ignore

    @classmethod
    def is_duration_relative(cls) -> bool:
        """Return ``True`` when the active duration mode is relative."""
        return cls._settings.duration == DurationOption.relative

    @classmethod
    def is_duration_absolute(cls) -> bool:
        """Return ``True`` when the active duration mode is absolute."""
        return cls._settings.duration == DurationOption.absolute

    @classmethod
    def update_duration(cls, value: str) -> None:
        """Update only the duration display mode.

        Args:
            value: ``'relative'`` or ``'absolute'``.

        Raises:
            ConfigValueError: When *value* is not a valid :class:`DurationOption`.

        """
        try:
            new_duration = DurationOption(value)
        except ValueError as err:
            msg = f"{err}. Choose either 'relative' or 'absolute'."
            raise ConfigValueError(msg) from err
        cls._settings = replace(cls._settings, duration=new_duration)

    @classmethod
    def configure(cls, explicit: "Settings | None" = None) -> "Settings":
        """Resolve settings and apply them to :class:`Config`.

        Environment variables act as *per-key overrides* on top of
        ``[tool.fixingahole]`` in ``pyproject.toml``; only the keys that are
        present in the environment override their pyproject counterparts — the
        rest continue to come from pyproject or built-in defaults.

        Full precedence, per key: explicit > environment > pyproject > defaults.

        Called automatically at import time.  Call again to reconfigure at runtime.

        When *explicit* is provided it is applied immediately, bypassing all other
        sources.  When any other configuration source is present (environment
        variables or ``[tool.fixingahole]`` in ``pyproject.toml``), invalid values
        raise immediately rather than falling back to defaults — ensuring
        configuration mistakes are never silently ignored.

        Args:
            explicit: A fully-constructed :class:`Settings` object.  When provided,
                all other configuration sources are ignored.

        Returns:
            The resolved and applied :class:`Settings` instance.

        Raises:
            ConfigError: When user-provided configuration is present but invalid.

        """
        if explicit is not None:
            settings = explicit
        else:
            pyproject_data, base_dir = _load_pyproject_config()
            env_data = _env_config_data()
            merged = pyproject_data | env_data
            if merged:
                try:
                    settings = _build_settings(merged, base_dir)
                except ConfigError as err:
                    Colour.error("%s", str(err))
                    raise
            else:
                settings = Settings.defaults()
        cls._settings = settings
        return settings


Config.configure()
