"""Robot-agnostic UI control-plane library.

Robot repositories provide backend adapters. This package owns only intent,
state, command lifecycle, and HTTP/SSE transport primitives.
"""

from robot_ui_platform.backend import UiBackend, UiBackendFactory
from robot_ui_platform.command_store import UiCommandStore
from robot_ui_platform.contracts import (
    CommandOutcome,
    CommandStatus,
    UiCapabilities,
    UiCommandCapability,
    UiIntent,
    UiSnapshot,
)
from robot_ui_platform.host import UiHost
from robot_ui_platform.state_store import UiStateStore

__all__ = [
    "CommandOutcome",
    "CommandStatus",
    "UiBackend",
    "UiBackendFactory",
    "UiCapabilities",
    "UiCommandCapability",
    "UiCommandStore",
    "UiHost",
    "UiIntent",
    "UiSnapshot",
    "UiStateStore",
]
