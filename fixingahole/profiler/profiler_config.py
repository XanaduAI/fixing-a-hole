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
"""Specialized configuration hook for the Profiler class."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fixingahole.profiler.profiler import Profiler


@runtime_checkable
class ProfilerConfig(Protocol):
    """A protocol for customizing Profiler initialization.

    Users can implement this configuration object to dynamically determine
    paths and settings before the profiler runs.
    """

    def setup(self, profiler_instance: "Profiler") -> None:
        """Perform any necessary setup on the profiler instance.

        Args:
            profiler_instance: The Profiler instance being initialized.

        """
