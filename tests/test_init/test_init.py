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
"""Tests for the Fixing-A-Hole config and init."""

import tomllib
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from fixingahole import config
from fixingahole.config import Config, ConfigParseError, ConfigValueError, DurationOption, Settings


class TestConfig:
    """Test Fixing-A-Hole config."""

    def test_detect_virtualenv(self):
        """Test detecting the virtual env."""
        assert config._detect_virtualenv()

    def test_find_pyproject(self):
        """Test finding the pyproject.toml."""
        pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
        assert config._find_pyproject() == pyproject_path

    def test_find_pyproject_outside_of_repo(self, tmp_path: Path):
        """Test finding the pyproject.toml when cwd is outside the repo."""
        pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
        with patch("fixingahole.config.Path.cwd") as cwd:
            cwd.side_effect = [tmp_path]
            assert config._find_pyproject() == pyproject_path

    def test_resolve_root_dir_defaults_to_cwd(self):
        """Test setting the root directory as the current working directory."""
        assert config._resolve_root_dir(config={}, base_dir=Path.cwd()) == Path.cwd()

    def test_detect_virtualenv_returns_sys_prefix_when_marker_exists(self):
        """Test virtualenv detection via pyvenv.cfg when env var is unset."""
        with patch("fixingahole.config.os.getenv", return_value=""), patch("fixingahole.config.Path.exists", return_value=True):
            assert config._detect_virtualenv() == config.sys.prefix

    def test_detect_virtualenv_returns_empty_when_not_in_venv(self):
        """Test virtualenv detection when no env var and no pyvenv marker are found."""
        with (
            patch("fixingahole.config.os.getenv", return_value=""),
            patch("fixingahole.config.Path.exists", return_value=False),
        ):
            assert config._detect_virtualenv() == ""  # noqa: PLC1901

    def test_find_pyproject_at_filesystem_root(self):
        """Test pyproject lookup when cwd is already the filesystem root."""
        with (
            patch("fixingahole.config.Path.cwd", return_value=Path("/")),
            patch("fixingahole.config.Path.exists", return_value=False),
            patch("fixingahole.config._detect_virtualenv", return_value=""),
        ):
            assert config._find_pyproject() is None

    def test_find_pyproject_at_filesystem_root_when_present(self):
        """Test pyproject lookup returns the root pyproject when present."""
        with (
            patch("fixingahole.config.Path.cwd", return_value=Path("/")),
            patch("fixingahole.config.Path.exists", side_effect=[True]),
        ):
            assert config._find_pyproject() == Path("/pyproject.toml")

    def test_find_pyproject_in_virtualenv_parent(self, tmp_path: Path):
        """Test fallback pyproject lookup in the parent directory of virtualenv."""
        venv_path = tmp_path / "some_venv"
        expected = venv_path.parent / "pyproject.toml"
        with (
            patch("fixingahole.config.Path.cwd", return_value=Path("/")),
            patch("fixingahole.config._detect_virtualenv", return_value=str(venv_path)),
            patch("fixingahole.config.Path.exists", side_effect=[False, True]),
        ):
            assert config._find_pyproject() == expected

    def test_load_pyproject_config_returns_empty_when_pyproject_missing(self):
        """Test config loading returns an empty dict when no pyproject is found."""
        with patch("fixingahole.config._find_pyproject", return_value=None):
            cfg, base = config._load_pyproject_config()
        assert cfg == {}
        assert base == Path.cwd()

    def test_load_pyproject_config_raises_on_toml_decode_error(self, tmp_path: Path):
        """Test config loading raises a parse error when TOML parsing fails."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.fixingahole]\n")
        decode_error = tomllib.TOMLDecodeError("bad toml", "", 0)

        with (
            patch("fixingahole.config._find_pyproject", return_value=pyproject),
            patch("fixingahole.config.tomllib.load", side_effect=decode_error),
            pytest.raises(ConfigParseError),
        ):
            config._load_pyproject_config()

    def test_resolve_output_dir_uses_absolute_path_unchanged(self, tmp_path: Path):
        """Test absolute output path is returned directly."""
        absolute_output = tmp_path / "performance"
        assert config._resolve_output_dir({"output": str(absolute_output)}, root_dir=tmp_path) == absolute_output

    def test_resolve_root_dir_uses_base_dir_for_relative_root(self, tmp_path: Path):
        """Test relative root resolves against provided base directory."""
        base_dir = tmp_path / "project"
        expected = (base_dir / "src").resolve()
        assert config._resolve_root_dir({"root": "src"}, base_dir=base_dir) == expected

    def test_resolve_ignore_dirs_non_string_non_list_defaults_to_output(self, tmp_path: Path):
        """Test invalid ignore config type falls back to output directory only."""
        output_path = tmp_path / "performance"
        assert config._resolve_ignore_dirs({"ignore": 123}, output_path, base_dir=tmp_path) == [output_path]

    def test_resolve_ignore_dirs_accepts_string_path(self, tmp_path: Path):
        """Test string ignore directory input is normalised and preserved."""
        ignore_path = tmp_path / "ignore_me"
        output_path = tmp_path / "performance"
        ignore_path.mkdir()
        output_path.mkdir(exist_ok=True)

        result = config._resolve_ignore_dirs({"ignore": str(ignore_path)}, output_path, base_dir=tmp_path)
        assert ignore_path.resolve() in result
        assert output_path in result

    def test_resolve_ignore_dirs_appends_output_when_missing(self, tmp_path: Path):
        """Test output path is appended when not present in ignore directories."""
        ignore_path = tmp_path / "ignore_me"
        output_path = tmp_path / "performance"
        ignore_path.mkdir()
        output_path.mkdir(exist_ok=True)

        result = config._resolve_ignore_dirs({"ignore": [str(ignore_path)]}, output_path, base_dir=tmp_path)
        assert ignore_path.resolve() in result
        assert output_path in result

    def test_resolve_ignore_dirs_resolves_relative_to_base_dir(self, tmp_path: Path):
        """Test relative ignore paths are resolved from config base directory."""
        base_dir = tmp_path / "project"
        ignore_path = base_dir / "ignore_me"
        output_path = base_dir / "performance"
        ignore_path.mkdir(parents=True)
        output_path.mkdir(exist_ok=True)

        result = config._resolve_ignore_dirs({"ignore": ["ignore_me"]}, output_path, base_dir=base_dir)
        assert ignore_path.resolve() in result

    def test_configure_updates_globals(self, tmp_path: Path):
        """Test configure() applies resolved settings to the module-level globals."""
        root = tmp_path / "project"
        output = root / "custom_output"
        ignore = root / "ignored"
        ignore.mkdir(parents=True)

        with patch(
            "fixingahole.config._load_pyproject_config",
            return_value=(
                {
                    "root": str(root),
                    "output": "custom_output",
                    "ignore": [str(ignore)],
                    "duration": "absolute",
                },
                tmp_path,
            ),
        ):
            settings = Config.configure()

        assert settings.root == root.resolve()
        assert settings.output == output.resolve()
        assert ignore.resolve() in settings.ignore
        assert settings.root == Config.root()
        assert settings.output == Config.output()
        assert settings.ignore == Config.ignore()
        assert Config.is_duration_absolute()

    def test_configure_applies_explicit_settings(self, tmp_path: Path):
        """configure(explicit=...) applies the given settings directly to globals."""
        explicit = Settings(
            root=tmp_path,
            output=tmp_path / "out",
            ignore=[tmp_path / "out"],
            duration=DurationOption.absolute,
        )
        settings = Config.configure(explicit=explicit)
        assert settings == explicit
        assert explicit.root == Config.root()
        assert explicit.output == Config.output()
        assert explicit.ignore == Config.ignore()
        assert Config.is_duration_absolute()


class TestConfigure:
    """Tests for configure() settings precedence and per-key merge."""

    def test_explicit_bypasses_all_other_sources(self, tmp_path: Path):
        """Explicit settings bypass env vars and pyproject entirely."""
        explicit = Settings(
            root=tmp_path,
            output=tmp_path / "out",
            ignore=[tmp_path / "out"],
            duration=DurationOption.absolute,
        )
        with patch.dict("fixingahole.config.os.environ", {"FIXINGAHOLE_DURATION": "relative"}, clear=False):
            result = Config.configure(explicit=explicit)
        assert result == explicit

    def test_env_key_overrides_matching_pyproject_key(self, tmp_path: Path):
        """A single env var overrides only that key; all other keys come from pyproject."""
        root = tmp_path / "myrepo"
        output = root / "results"
        ignore = tmp_path / "scratch"
        ignore.mkdir(parents=True)

        pyproject_config = {
            "root": str(root),
            "output": "results",
            "ignore": [str(ignore)],
            "duration": "relative",
        }

        with (
            patch("fixingahole.config._load_pyproject_config", return_value=(pyproject_config, tmp_path)),
            patch("fixingahole.config.os.getenv", side_effect=lambda k, *_: {"FIXINGAHOLE_DURATION": "absolute"}.get(k)),
        ):
            settings = Config.configure()

        assert settings.root == root.resolve()
        assert settings.output == output.resolve()
        assert ignore.resolve() in settings.ignore
        assert settings.duration == DurationOption.absolute  # from env

    def test_all_env_keys_used_when_no_pyproject(self, tmp_path: Path):
        """All settings resolved from env vars when no pyproject is found."""
        root = tmp_path / "repo"
        output = root / "perf"
        ignore = tmp_path / "scratch"
        ignore.mkdir()

        env = {
            "FIXINGAHOLE_ROOT": str(root),
            "FIXINGAHOLE_OUTPUT": "perf",
            "FIXINGAHOLE_IGNORE": str(ignore),
            "FIXINGAHOLE_DURATION": "absolute",
        }

        with (
            patch("fixingahole.config._load_pyproject_config", return_value=({}, tmp_path)),
            patch("fixingahole.config.os.getenv", side_effect=lambda k, *_: env.get(k)),
        ):
            settings = Config.configure()

        assert settings.root == root.resolve()
        assert settings.output == output.resolve()
        assert ignore.resolve() in settings.ignore
        assert settings.duration == DurationOption.absolute

    def test_pyproject_used_when_no_env(self, tmp_path: Path):
        """All settings come from pyproject when no env vars are set."""
        root = tmp_path / "myrepo"
        output = root / "results"
        ignore = tmp_path / "scratch"
        ignore.mkdir()

        pyproject_config = {
            "root": str(root),
            "output": "results",
            "ignore": [str(ignore)],
            "duration": "absolute",
        }

        with (
            patch("fixingahole.config._load_pyproject_config", return_value=(pyproject_config, tmp_path)),
            patch("fixingahole.config.os.getenv", return_value=None),
        ):
            settings = Config.configure()

        assert settings.root == root.resolve()
        assert settings.output == output.resolve()
        assert ignore.resolve() in settings.ignore
        assert settings.duration == DurationOption.absolute

    def test_defaults_used_silently_when_no_config_source(self):
        """Defaults are applied silently when no pyproject and no env vars are set."""
        with (
            patch("fixingahole.config._load_pyproject_config", return_value=({}, Path.cwd())),
            patch("fixingahole.config.os.getenv", return_value=None),
        ):
            settings = Config.configure()

        assert settings == Settings.defaults()

    def test_env_ignore_csv_resolved_against_cwd(self, tmp_path: Path):
        """FIXINGAHOLE_IGNORE entries are split by comma and resolved against cwd."""
        cwd = tmp_path / "project"
        ignore_one = cwd / "ignore_one"
        ignore_two = cwd / "ignore_two"
        ignore_one.mkdir(parents=True)
        ignore_two.mkdir()

        env = {"FIXINGAHOLE_IGNORE": "ignore_one, ignore_two"}

        with (
            patch("fixingahole.config._load_pyproject_config", return_value=({}, cwd)),
            patch("fixingahole.config.os.getenv", side_effect=lambda k, *_: env.get(k)),
            patch("fixingahole.config.Path.cwd", return_value=cwd),
        ):
            settings = Config.configure()

        assert ignore_one.resolve() in settings.ignore
        assert ignore_two.resolve() in settings.ignore


class TestDurationOption:
    """Basic tests for the DurationOption in the config."""

    def test_enum_values(self):
        """Basic test that the DurationOption Enum works."""
        assert DurationOption.absolute.value == "absolute"
        assert DurationOption.relative.value == "relative"
        bad_value = "bad_value"
        with pytest.raises(ValueError, match=f"'{bad_value}' is not a valid DurationOption"):
            DurationOption(bad_value)


class TestConfigDuration:
    """Tests for Config duration management.

    Each test resets the duration via :meth:`Config.update_duration` via the
    autouse fixture so tests remain independent.
    """

    @pytest.fixture(autouse=True)
    def reset_mode(self) -> Generator[None, None, None]:
        """Reset duration to relative before and after each test."""
        Config.update_duration("relative")
        yield
        Config.update_duration("relative")

    def test_default_mode_is_relative(self):
        """Config defaults to relative duration mode."""
        assert Config.is_duration_relative()
        assert not Config.is_duration_absolute()

    def test_update_to_absolute(self):
        """Calling update_duration('absolute') switches the mode to absolute."""
        Config.update_duration("absolute")
        assert Config.is_duration_absolute()
        assert not Config.is_duration_relative()

    def test_update_to_relative(self):
        """Calling update_duration('relative') keeps (or restores) relative mode."""
        Config.update_duration("absolute")
        Config.update_duration("relative")
        assert Config.is_duration_relative()
        assert not Config.is_duration_absolute()

    def test_update_invalid_value_raises(self):
        """update_duration() raises ConfigValueError for unrecognised mode strings."""
        with pytest.raises(ConfigValueError):
            Config.update_duration("invalid")

    def test_mode_unchanged_after_invalid_update(self):
        """Duration mode is not mutated when update_duration() raises."""
        Config.update_duration("absolute")
        with pytest.raises(ConfigValueError):
            Config.update_duration("bad")
        assert Config.is_duration_absolute()

    def test_config_settings_is_settings_instance(self):
        """Config.settings() always returns a Settings instance."""
        assert isinstance(Config.settings(), Settings)

    def test_config_cannot_be_instantiated(self):
        """Config raises TypeError when instantiation is attempted."""
        with pytest.raises(TypeError, match="Config cannot be instantiated"):
            Config()

    def test_update_duration_reflected_in_settings(self):
        """update_duration() changes the duration reported by Config."""
        Config.update_duration("absolute")
        assert Config.is_duration_absolute()
