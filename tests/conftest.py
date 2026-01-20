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
"""Helper code for testing the FlamingPy CLI tools."""

import contextlib
from pathlib import Path

import pytest
from click.testing import Result


def basic_name(suffix: str = "") -> str:
    """Return a basic name for testing."""
    return f"profiler_testfile{suffix}"


@pytest.fixture
def mock_file(tmp_path: Path) -> Path:
    """Create a basic test file for profiler testing."""
    basic_script = Path(__file__).parent / "scripts" / "basic.py"
    test_file = tmp_path / basic_name(".py")
    test_file.write_text(basic_script.read_text())
    return test_file


@pytest.fixture
def mock_dir(tmp_path: Path) -> Path:
    """Create a basic test file for profiler testing."""
    test_dir = tmp_path / basic_name()
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(name="root_dir", autouse=True)
def fixture_root_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary repo root directory for tests."""
    mock_root_dir = tmp_path
    mock_output_dir = mock_root_dir / "performance"
    mock_output_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).parents[1] / "fixingahole"
    for path in package_root.rglob("*.py"):
        part = ".".join(path.relative_to(package_root).parts)[:-3]
        for folder in ["ROOT_DIR", "OUTPUT_DIR"]:
            with contextlib.suppress(AttributeError):
                monkeypatch.setattr(f"fixingahole.{part}.{folder}", mock_root_dir)
    return mock_root_dir


def print_error(res: Result) -> None:
    """Trace errors thrown by the CLI."""
    print(res)
    print(res.stdout)
    exc: BaseException | None = res.exception
    if exc is not None and (tb := exc.__traceback__) is not None:
        print(f"{tb=}")
        while tb.tb_next:
            print(tb.tb_frame)
            tb = tb.tb_next
        print(tb.tb_frame)
        return
