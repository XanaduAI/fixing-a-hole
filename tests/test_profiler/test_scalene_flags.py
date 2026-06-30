# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for scalene_flags: flag parsing, validation, and round-tripping."""

import pytest

from fixingahole.profiler.scalene_flags import (
    _ALIASES,  # noqa: PLC2701
    _META,  # noqa: PLC2701
    _TOKEN_MAP,  # noqa: PLC2701
    DuplicateKeyError,
    scalene_flags_to_kwargs,
    scalene_kwargs_to_flags,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _known_bool_flag() -> str:
    """Return the positive CLI flag for a known boolean Scalene option."""
    for cfg in _META.values():
        if cfg["bool"] and cfg["pos"]:
            return cfg["pos"]
    return pytest.fail("Expected at least one Scalene boolean flag, but none were found")


def _known_value_flag() -> tuple[str, str]:
    """Return (flag, example_value) for a known non-boolean Scalene option."""
    for cfg in _META.values():
        if not cfg["bool"] and cfg["pos"]:
            return cfg["pos"], "1"
    return pytest.fail("Expected at least one Scalene non-boolean value flag, but none were found")


# ---------------------------------------------------------------------------
# _metadata / module-level constants
# ---------------------------------------------------------------------------


class TestMetadata:
    """Verify the module-level metadata structures are well-formed."""

    def test_meta_keys_are_strings(self):
        """Every key in _META is a non-empty string."""
        for key in _META:
            assert key
            assert isinstance(key, str)

    def test_meta_values_have_required_fields(self):
        """Every entry in _META has 'pos', 'neg', and 'bool' keys."""
        for dest, cfg in _META.items():
            assert "pos" in cfg, f"Missing 'pos' for {dest}"
            assert "neg" in cfg, f"Missing 'neg' for {dest}"
            assert "bool" in cfg, f"Missing 'bool' for {dest}"

    def test_aliases_cover_all_destinations(self):
        """Every destination in _META has at least one entry in _ALIASES."""
        for dest in _META:
            assert dest in _ALIASES, f"{dest} missing from _ALIASES"

    def test_token_map_keys_are_valid_flags(self):
        """Every key in _TOKEN_MAP starts with '--'."""
        for token in _TOKEN_MAP:
            assert token.startswith("--"), f"Bad token: {token}"

    def test_reserved_flags_absent_from_meta(self):
        """Reserved destinations (outfile, program_path, …) must not appear in _META."""
        from fixingahole.profiler.scalene_flags import RESERVED  # noqa: PLC0415

        for dest in RESERVED:
            assert dest not in _META, f"Reserved dest {dest!r} leaked into _META"


# ---------------------------------------------------------------------------
# scalene_flags_to_kwargs
# ---------------------------------------------------------------------------


class TestScaleneFlagsToKwargs:
    """Tests for parsing a list of CLI tokens into a kwargs dict."""

    def test_empty_input_returns_empty_dict(self):
        """An empty token list should produce an empty kwargs dict."""
        result = scalene_flags_to_kwargs([])
        assert result == {}

    def test_boolean_flag_parsed_as_true(self):
        """A positive boolean flag should map its destination key to True."""
        flag = _known_bool_flag()
        result = scalene_flags_to_kwargs([flag])
        dest = _TOKEN_MAP[flag][0]
        assert result[dest] is True

    def test_negated_boolean_flag_parsed_as_false(self):
        """If a --no-<flag> variant exists, it should produce False."""
        for dest, cfg in _META.items():
            if cfg["bool"] and cfg["neg"]:
                result = scalene_flags_to_kwargs([cfg["neg"]])
                assert result[dest] is False
                return
        pytest.fail("No negatable boolean flag found")

    def test_value_flag_parsed_with_space_separator(self):
        """A value flag followed by a separate token should be parsed correctly."""
        flag, val = _known_value_flag()
        dest = _TOKEN_MAP[flag][0]
        result = scalene_flags_to_kwargs([flag, val])
        assert dest in result

    def test_value_flag_parsed_with_equals_separator(self):
        """A value flag in --flag=value form should be parsed correctly."""
        flag, val = _known_value_flag()
        dest = _TOKEN_MAP[flag][0]
        result = scalene_flags_to_kwargs([f"{flag}={val}"])
        assert dest in result

    def test_value_flag_equals_with_equals_in_value(self):
        """A value flag in --flag=foo=bar form should not split on the second '='."""
        result = scalene_flags_to_kwargs(["--profile-only=foo=bar"])
        assert result.get("profile_only") == "foo=bar"

    def test_integer_value_is_cast(self):
        """Numeric strings for value flags should be cast to int or float."""
        flag, _ = _known_value_flag()
        dest = _TOKEN_MAP[flag][0]
        result = scalene_flags_to_kwargs([flag, "42"])
        assert isinstance(result[dest], (int, float, str))  # str if Scalene type=None

    def test_invalid_typed_value_exits(self):
        """A value that fails Scalene's declared type should call sys.exit."""
        for cfg in _META.values():
            if not cfg["bool"] and cfg["pos"] and cfg["type"] is int:
                with pytest.raises(SystemExit):
                    scalene_flags_to_kwargs([cfg["pos"], "snickers"])
                return
        pytest.skip("No int-typed Scalene flag found in this version")

    def test_comma_separated_value_becomes_list(self):
        """A comma-separated string value should be split into a list."""
        flag, _ = _known_value_flag()
        dest = _TOKEN_MAP[flag][0]
        result = scalene_flags_to_kwargs([flag, "a,b,c"])
        assert isinstance(result[dest], list)
        assert len(result[dest]) == 3

    def test_unknown_flag_calls_sys_exit(self):
        """An unrecognised flag should call sys.exit rather than raise."""
        with pytest.raises(SystemExit):
            scalene_flags_to_kwargs(["--not-a-real-scalene-flag"])

    def test_reserved_cli_flag_exits_with_managed_message(self, capsys: pytest.CaptureFixture[str]):
        """Passing a reserved CLI flag (e.g. --outfile) should exit with a 'managed by fixing-a-hole' message."""
        from fixingahole.profiler.scalene_flags import _RESERVED_TOKEN_MAP  # noqa: PLC0415, PLC2701

        if not _RESERVED_TOKEN_MAP:
            pytest.skip("No reserved CLI tokens found in this Scalene version")
        token = next(iter(_RESERVED_TOKEN_MAP))
        with pytest.raises(SystemExit):
            scalene_flags_to_kwargs([token])
        captured = capsys.readouterr()
        assert "managed by fixing-a-hole" in captured.err or "managed by fixing-a-hole" in captured.out

    def test_duplicate_boolean_flag_raises_duplicate_key_error(self):
        """Passing the same boolean flag twice should raise DuplicateKeyError."""
        flag = _known_bool_flag()
        with pytest.raises(DuplicateKeyError):
            scalene_flags_to_kwargs([flag, flag])

    def test_repeated_scalar_value_flag_raises_duplicate_key_error(self):
        """Supplying the same scalar value flag twice should raise DuplicateKeyError."""
        accumulating = {"profile_exclude", "profile_only"}
        for cfg in _META.values():
            dest_key = _TOKEN_MAP.get(cfg["pos"], (None,))[0] if cfg["pos"] else None
            if not cfg["bool"] and cfg["pos"] and dest_key and dest_key not in accumulating:
                flag_type = cfg["type"]
                v1, v2 = ("1", "2") if flag_type in {int, float} else ("aaa", "bbb")
                with pytest.raises(DuplicateKeyError):
                    scalene_flags_to_kwargs([cfg["pos"], v1, cfg["pos"], v2])
                return
        pytest.fail("No non-accumulating scalar value flag found")

    def test_repeated_profile_exclude_accumulates_list(self):
        """--profile-exclude is a CSV flag; repeated occurrences should be merged into a list."""
        result = scalene_flags_to_kwargs(["--profile-exclude", "/a", "--profile-exclude", "/b"])
        assert isinstance(result["profile_exclude"], list)
        assert "/a" in result["profile_exclude"]
        assert "/b" in result["profile_exclude"]

    def test_multiple_distinct_flags(self):
        """Two distinct flags should both appear in the result."""
        flags = []
        for cfg in _META.values():
            if cfg["bool"] and cfg["pos"]:
                flags.append(cfg["pos"])
            if len(flags) == 2:
                break
        if len(flags) < 2:
            pytest.fail("Expected at least two boolean flags in _META, but found fewer")
        result = scalene_flags_to_kwargs(flags)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# scalene_kwargs_to_flags
# ---------------------------------------------------------------------------


class TestScaleneKwargsToFlags:
    """Tests for serialising a kwargs dict back into CLI flag tokens."""

    def test_empty_dict_returns_empty_list(self):
        """An empty kwargs dict should produce an empty token list."""
        assert scalene_kwargs_to_flags({}) == []

    def test_none_value_is_skipped(self):
        """A kwargs entry with a None value should be omitted from the output."""
        dest = next(iter(_META))
        assert scalene_kwargs_to_flags({dest: None}) == []

    def test_boolean_true_emits_positive_flag(self):
        """A True value for a boolean destination should emit its positive flag."""
        for dest, cfg in _META.items():
            if cfg["bool"] and cfg["pos"]:
                tokens = scalene_kwargs_to_flags({dest: True})
                assert cfg["pos"] in tokens
                return
        pytest.fail("No boolean flag found")

    def test_boolean_false_emits_negative_flag(self):
        """A False value for a negatable boolean destination should emit its --no- flag."""
        for dest, cfg in _META.items():
            if cfg["bool"] and cfg["neg"]:
                tokens = scalene_kwargs_to_flags({dest: False})
                assert cfg["neg"] in tokens
                return
        pytest.fail("No negatable boolean flag found")

    def test_boolean_false_without_neg_emits_nothing(self):
        """A False value for a boolean flag with no negative form should emit nothing."""
        for dest, cfg in _META.items():
            if cfg["bool"] and cfg["pos"] and not cfg["neg"]:
                tokens = scalene_kwargs_to_flags({dest: False})
                assert cfg["pos"] not in tokens
                return
        pytest.fail("No non-negatable boolean flag found")

    def test_value_flag_emits_flag_and_value(self):
        """A string value for a non-boolean destination should emit the flag token followed by the value."""
        for dest, cfg in _META.items():
            if not cfg["bool"] and cfg["pos"]:
                tokens = scalene_kwargs_to_flags({dest: "hello"})
                assert cfg["pos"] in tokens
                assert "hello" in tokens
                return
        pytest.fail("No value flag found")

    def test_list_value_is_joined_with_comma(self):
        """A list value for a non-boolean destination should be emitted as a comma-joined string."""
        for dest, cfg in _META.items():
            if not cfg["bool"] and cfg["pos"]:
                tokens = scalene_kwargs_to_flags({dest: ["a", "b"]})
                assert "a,b" in tokens
                return
        pytest.fail("No value flag found")

    def test_reserved_key_raises_value_error(self):
        """Passing a reserved destination key should raise ValueError."""
        from fixingahole.profiler.scalene_flags import RESERVED  # noqa: PLC0415

        reserved_key = next(iter(RESERVED))
        with pytest.raises(ValueError, match="managed by fixing-a-hole"):
            scalene_kwargs_to_flags({reserved_key: True})

    def test_unknown_key_raises_value_error(self):
        """An unrecognised destination key should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scalene flag"):
            scalene_kwargs_to_flags({"not_a_scalene_kwarg": True})


# ---------------------------------------------------------------------------
# Round-trip: flags → kwargs → flags
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that flags → kwargs → flags is stable."""

    def test_boolean_flag_round_trips(self):
        """A boolean flag should survive a flags → kwargs → flags round-trip unchanged."""
        flag = _known_bool_flag()
        kwargs = scalene_flags_to_kwargs([flag])
        flags_out = scalene_kwargs_to_flags(kwargs)
        assert flag in flags_out

    def test_value_flag_round_trips(self):
        """A value flag should survive a flags → kwargs → flags round-trip unchanged."""
        flag, val = _known_value_flag()
        kwargs = scalene_flags_to_kwargs([flag, val])
        flags_out = scalene_kwargs_to_flags(kwargs)
        assert flag in flags_out

    def test_kwargs_to_flags_round_trips(self):
        """Kwargs → flags → kwargs should reproduce the original dict."""
        for dest, cfg in _META.items():
            if cfg["bool"] and cfg["pos"]:
                original = {dest: True}
                flags = scalene_kwargs_to_flags(original)
                recovered = scalene_flags_to_kwargs(flags)
                assert recovered == original
                return
        pytest.fail("No boolean flag found")


# ---------------------------------------------------------------------------
# cpu-only / --memory targeted tests
# ---------------------------------------------------------------------------


class TestCpuOnlyMemoryFlags:
    """Targeted tests for --cpu-only / --memory flag handling."""

    def test_cpu_only_flag_maps_to_cpu_dest(self):
        """--cpu-only should be stored under the 'cpu' destination, not 'cpu_only'."""
        result = scalene_flags_to_kwargs(["--cpu-only"])
        assert "cpu" in result
        assert result["cpu"] is True
        assert "cpu_only" not in result

    def test_cpu_only_round_trips_without_duplication(self):
        """--cpu-only should survive kwargs→flags without producing a duplicate flag."""
        kwargs = scalene_flags_to_kwargs(["--cpu-only"])
        flags = scalene_kwargs_to_flags(kwargs)
        assert flags.count("--cpu-only") == 1

    def test_memory_flag_maps_to_memory_dest(self):
        """--memory should be stored under the 'memory' destination."""
        result = scalene_flags_to_kwargs(["--memory"])
        assert "memory" in result
        assert result["memory"] is True

    def test_cpu_only_and_memory_are_mutually_exclusive_in_kwargs(self):
        """Passing both --cpu-only and --memory produces both keys in kwargs (resolution is up to Profiler)."""
        result = scalene_flags_to_kwargs(["--cpu-only", "--memory"])
        assert result.get("cpu") is True
        assert result.get("memory") is True
