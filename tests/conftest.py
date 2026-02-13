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
"""Helper code for running tests."""

import contextlib
from pathlib import Path

import pytest
from click.testing import Result

from fixingahole.config import Duration


@pytest.fixture(autouse=True)
def set_duration() -> None:
    """Make sure that the Duration singleton is always reset."""
    Duration("relative")
    Duration.update("relative")


def basic_name(suffix: str = "") -> str:
    """Return a basic name for testing."""
    return f"profiler_testfile{suffix}"


@pytest.fixture
def non_local_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory outside of the repo root_dir for tests."""
    return tmp_path_factory.mktemp("not_within_root_dir")


@pytest.fixture(name="root_dir", autouse=True)
def fixture_root_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary repo root directory for tests."""
    mock_root_dir = tmp_path
    mock_output_dir = mock_root_dir / "performance"
    mock_output_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).parents[1] / "fixingahole"
    for path in package_root.rglob("*.py"):
        part = ".".join(path.relative_to(package_root).parts)[:-3]
        for folder, mock_dir in [("ROOT_DIR", mock_root_dir), ("OUTPUT_DIR", mock_output_dir)]:
            with contextlib.suppress(AttributeError):
                monkeypatch.setattr(f"fixingahole.{part}.{folder}", mock_dir)
    monkeypatch.setattr("fixingahole.ROOT_DIR", mock_root_dir)
    monkeypatch.setattr("fixingahole.OUTPUT_DIR", mock_output_dir)
    return mock_root_dir


@pytest.fixture
def mock_file(root_dir: Path) -> Path:
    """Create a basic test file for profiler testing."""
    basic_script = Path(__file__).parent / "scripts" / "basic.py"
    test_file = root_dir / basic_name(".py")
    test_file.write_text(basic_script.read_text())
    return test_file


@pytest.fixture
def mock_file_with_argparse(root_dir: Path) -> Path:
    """Create a basic test file for profiler testing."""
    basic_script = Path(__file__).parent / "scripts" / "with_argparse.py"
    test_file = root_dir / basic_name(".py")
    test_file.write_text(basic_script.read_text())
    return test_file


@pytest.fixture
def mock_dir(root_dir: Path) -> Path:
    """Create a basic test file for profiler testing."""
    test_dir = root_dir / basic_name()
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def example_json(root_dir: Path) -> Path:
    """Return path to the advanced profile results JSON file."""
    example_json_file: Path = Path(__file__).parent / "scripts" / "data" / "advanced_profile_results.json"
    file_path: Path = root_dir / "example.json"
    file_path.write_bytes(example_json_file.read_bytes())
    return file_path


def print_error(res: Result) -> None:
    """Trace errors thrown by the CLI."""
    print(res)  # noqa: T201
    print(res.stdout)  # noqa: T201
    exc: BaseException | None = res.exception
    if exc is not None and (tb := exc.__traceback__) is not None:
        print(f"{tb=}")  # noqa: T201
        while tb.tb_next:
            print(tb.tb_frame)  # noqa: T201
            tb = tb.tb_next
        print(tb.tb_frame)  # noqa: T201
        return
