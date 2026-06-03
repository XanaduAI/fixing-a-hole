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
"""Convert Profiler kwargs / typer.Context into `scalene run` tokens."""

import argparse
import os
import subprocess
import sys

import click
from colours import Colour
from rich.console import Console
from rich.markdown import Markdown
from scalene.scalene_arguments import ScaleneArguments
from scalene.scalene_parseargs import ScaleneParseArgs

RESERVED = {
    "outfile",
    "profile_interval",
    "allocation_sampling_window",
    "program_path",
    "config_file",
    "unused_args",
}


class DuplicateKeyError(click.exceptions.Exit):
    """Key was already given."""


def _run_parser(advanced: bool) -> argparse.ArgumentParser:
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
            e = meta.setdefault(a.dest, {"pos": None, "neg": None, "bool": is_bool})
            for o in longs:
                (e.__setitem__("neg", o) if o.startswith("--no-") else e.__setitem__("pos", o))
            if not is_bool:
                e["bool"] = False
    return meta


_META = _metadata()
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
        env=os.environ | {"LINES": "320", "COLUMNS": "160"},
    )
    scalene_help = ""
    if res.returncode == 0:
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
        raise ValueError(msg)
    if unknown:
        msg = f"Unknown scalene flag(s): {unknown}. Valid: {sorted(_ALIASES)}"
        raise ValueError(msg)
    return tokens


def scalene_flags_to_kwargs(ctx_args: list[str]) -> dict:
    """Convert a Click.Context.args (list[str]) into Scalene kwargs."""
    result: dict = {}
    args = [part for a in ctx_args for part in (a.split("=") if "=" in a else [a])]
    i = 0
    while i < len(args):
        token = args[i]
        if token in _TOKEN_MAP:
            key, cfg = _TOKEN_MAP[token]
            if cfg["bool"]:
                if key not in result:
                    result[key] = not token.startswith("--no-")
                    i += 1
                else:
                    Colour.error("DuplicateKeyError: The %s flag (%s/%s) was already provided.", key, cfg["pos"], cfg["neg"])
                    raise DuplicateKeyError(code=1)
            elif i + 1 < len(args):
                val = args[i + 1]
                for cast in (int, float):
                    try:
                        val = cast(val)
                        break
                    except ValueError:
                        continue
                val = [v.strip() for v in val.split(",")] if isinstance(val, str) and "," in val else val
                if key not in result:
                    result[key] = val
                else:
                    result[key] = [
                        *(result[key] if isinstance(result[key], list) else (result[key],)),
                        *(val if isinstance(val, list) else (val,)),
                    ]
                i += 2
            else:
                i += 1
        else:
            i += 1
    return result
