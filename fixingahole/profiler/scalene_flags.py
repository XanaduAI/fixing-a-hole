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
"""Convert Profiler kwargs / CLI tokens into ``scalene run`` flags and back."""

import argparse
import subprocess
import sys
import threading

import click
from colours import Colour
from rich.console import Console
from rich.markdown import Markdown
from scalene.scalene_arguments import ScaleneArguments
from scalene.scalene_parseargs import ScaleneParseArgs

from fixingahole import Config

RESERVED = {
    "outfile",
    "program_path",
    "config_file",
    "unused_args",
    # Meta-flags: not valid for `scalene run` profiling
    "help_advanced",
    "on",
    "off",
}


class _ScaleneError(ValueError, click.exceptions.Exit):
    """Base class combining :class:`ValueError` and :class:`click.exceptions.Exit`.

    Subclasses gain both programmatic (``ValueError``) and Click CLI
    (``click.exceptions.Exit``) exception behaviour without each needing
    their own ``__init__``.
    """

    _default_message: str = "An error occurred."

    def __init__(self, *args: object, code: int = 0) -> None:
        ValueError.__init__(self, *(args or (self._default_message,)))
        click.exceptions.Exit.__init__(self, code)


class DuplicateKeyError(_ScaleneError):
    """Key was already given."""

    _default_message = "A flag or option was provided more than once."


class ReservedKeyError(_ScaleneError):
    """Key is reserved for use by fixing-a-hole."""

    _default_message = "That flag is managed internally by fixing-a-hole and cannot be set directly."


class MissingValueError(_ScaleneError):
    """Value was expected for a given key."""

    _default_message = "A value was expected after the flag but none was found."


class InvalidValueError(_ScaleneError):
    """Value was invalid or unrecognised for a given key."""

    _default_message = "The value or flag provided was invalid or unrecognised."


# Protects the temporary sys.argv mutation in _run_parser against concurrent
# access (e.g. when tests are executed with a thread-based parallelism plugin).
_SYS_ARGV_LOCK = threading.Lock()


def _run_parser(advanced: bool) -> argparse.ArgumentParser:
    with _SYS_ARGV_LOCK:
        saved = sys.argv
        sys.argv = ["scalene", "run"] + (["--help-advanced"] if advanced else [])
        try:
            p = argparse.ArgumentParser(prog="scalene run", add_help=False, allow_abbrev=False)
            ScaleneParseArgs._add_run_arguments(p, ScaleneArguments())  # noqa: SLF001
        finally:
            sys.argv = saved
    return p


def _metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for adv in (False, True):
        for a in _run_parser(adv)._actions:  # noqa: SLF001
            longs = [o for o in a.option_strings if o.startswith("--")]
            if not longs or a.dest in RESERVED:
                continue
            is_bool = type(a).__name__ in {"_StoreTrueAction", "_StoreFalseAction", "_StoreConstAction"}
            e = meta.setdefault(a.dest, {"pos": None, "neg": None, "bool": is_bool, "type": bool if is_bool else None})
            for o in longs:
                (e.__setitem__("neg", o) if o.startswith("--no-") else e.__setitem__("pos", o))
            if not is_bool:
                e["bool"] = False
                e["type"] = a.type
    return meta


def _reserved_token_map() -> dict[str, tuple[str, bool]]:
    """Map every CLI token (long and short) for reserved dests to (dest, is_bool)."""
    mapping: dict[str, tuple[str, bool]] = {}
    for adv in (False, True):
        for a in _run_parser(adv)._actions:  # noqa: SLF001
            if a.dest in RESERVED:
                is_bool = type(a).__name__ in {"_StoreTrueAction", "_StoreFalseAction", "_StoreConstAction"}
                for opt in a.option_strings:
                    mapping[opt] = (a.dest, is_bool)
    return mapping


_META = _metadata()
_RESERVED_TOKEN_MAP: dict[str, tuple[str, bool]] = _reserved_token_map()
_ALIASES = {d: d for d in _META}
_ALIASES |= {e["pos"].lstrip("-").replace("-", "_"): d for d, e in _META.items() if e["pos"]}
_TOKEN_MAP: dict[str, tuple[str, dict]] = {
    flag: (key, cfg) for key, cfg in _META.items() for flag in (cfg["pos"], cfg["neg"]) if flag
}


def get_scalene_help(cmd: list[str] | None = None, *, append: str = "") -> str:
    """Render Scalene help commands."""
    cmd = cmd if cmd is not None else ["run", "--help"]
    res = subprocess.run(
        [f"{sys.executable}", "-m", "scalene", *cmd],
        check=False,
        capture_output=True,
        text=True,
        env=Config.env(),
    )
    scalene_help = ""
    if res.returncode != 0:
        Colour.warning(
            "Warning: `scalene %s` exited with code %d; help text will be empty.\n%s",
            " ".join(cmd),
            res.returncode,
            res.stderr.strip(),
        )
    else:
        lines = res.stdout.splitlines()
        rm: list[int] = []
        for i, line in enumerate(lines):
            if line.strip().startswith(("%", "examples", "usage")):
                rm.append(i)
            elif all(c == " " for c in line[:4]):
                rm.append(i)
                lines[i - 1] += "\t" + line.strip()
        for i in sorted(rm, reverse=True):
            del lines[i]

        lines = [line for line in lines if "--help" not in line]
        scalene_help = "\n\n".join(lines)
        console = Console(highlight=False)
        with console.capture() as capture:
            console.print(Markdown(scalene_help))
        scalene_help = capture.get()
    return scalene_help + append


def scalene_kwargs_to_flags(kwargs: dict) -> list[str]:
    """Convert a dict of Scalene kwargs into CLI flags."""
    tokens, unknown, reserved = [], [], []
    for key, val in kwargs.items():
        if val is None:
            continue
        norm = key.lstrip("-").replace("-", "_")
        if norm in RESERVED:
            reserved.append(key)
            continue
        dest = _ALIASES.get(norm)
        if dest is None:
            unknown.append(key)
            continue
        e = _META[dest]
        if e["bool"]:
            if val:
                tokens.append(e["pos"])
            elif e["neg"]:
                tokens.append(e["neg"])
        else:
            tokens += [e["pos"], ",".join(map(str, val)) if isinstance(val, list) else str(val)]
    if reserved:
        msg = f"These scalene flags are managed by fixing-a-hole: {reserved}"
        raise ReservedKeyError(msg, code=1)
    if unknown:
        msg = f"Unknown scalene flag(s): {unknown}. Valid: {sorted(_ALIASES)}"
        raise InvalidValueError(msg, code=1)
    return tokens


def scalene_flags_to_kwargs(ctx_args: list[str]) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Convert a Click.Context.args (list[str]) into Scalene kwargs."""
    result: dict = {}
    unknown: list[str] = []
    reserved_used: list[str] = []
    args = [part for a in ctx_args for part in (a.split("=", 1) if a.startswith("--") and "=" in a else [a])]
    i = 0
    while i < len(args):
        token = args[i]
        if token in _RESERVED_TOKEN_MAP:
            _, is_bool = _RESERVED_TOKEN_MAP[token]
            reserved_used.append(token)
            has_value = not is_bool and i + 1 < len(args) and not args[i + 1].startswith("--")
            i += 2 if has_value else 1
        elif token in _TOKEN_MAP:
            key, cfg = _TOKEN_MAP[token]
            if cfg["bool"]:
                if key not in result:
                    result[key] = not token.startswith("--no-")
                    i += 1
                else:
                    Colour.error("DuplicateKeyError: The %s flag (%s/%s) was already provided.", key, cfg["pos"], cfg["neg"])
                    raise DuplicateKeyError(code=1)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                val = args[i + 1]
                cast = _META[key]["type"]
                if cast is not None:
                    try:
                        val = cast(val)
                    except (ValueError, TypeError) as err:
                        Colour.error(
                            "Invalid value for %s: %s is not a valid %s.",
                            Colour.green(cfg["pos"]),
                            Colour.orange(repr(val)),
                            cast.__name__,
                        )
                        raise InvalidValueError(code=1) from err
                val = [v.strip() for v in val.split(",")] if isinstance(val, str) and "," in val else val
                if key not in result:
                    result[key] = val
                elif key in {"profile_exclude", "profile_only"}:
                    # CSV-accumulating flags; merge repeated occurrences.
                    result[key] = [
                        *(result[key] if isinstance(result[key], list) else (result[key],)),
                        *(val if isinstance(val, list) else (val,)),
                    ]
                else:
                    Colour.error(
                        "DuplicateKeyError: The %s option was already provided. Also found %s",
                        Colour.green(f"{token} {result[key]}"),
                        Colour.orange(f"{token} {val}"),
                    )
                    raise DuplicateKeyError(code=1)
                i += 2
            else:
                Colour.error(
                    "Missing value for %s: expected a value after the flag.",
                    Colour.green(token),
                )
                raise MissingValueError(code=3)
        else:
            unknown.append(token)
            i += 1
    if reserved_used:
        Colour.error("These scalene flags are managed by fixing-a-hole: %s", [Colour.orange(u) for u in reserved_used])
        raise ReservedKeyError(code=1)
    if unknown:
        s = "s" if len(unknown) > 1 else ""
        msg = f"Unknown flag{s}: {[Colour.orange(u) for u in unknown]}. Was a script argument placed before the `---`?"
        valid_flags = "\n ".join([Colour.green(f) for f in sorted(_TOKEN_MAP)])
        msg += f"\n{Colour.GREEN('Valid Scalene flags:')}\n {valid_flags}"
        Colour.error(msg)
        raise InvalidValueError(code=1)
    return result
