"""Optional ROS executor for XR Nero Providers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from teleop.nero_console.provider import NeroConsoleProvider
from teleop.ui.command_bus import UiCommandBus, UiCommandName


# The standalone Nero host defaults to a 3 s HTTP/ROS wait.  Reply before that
# deadline so a timed-out web request can never leave an executable XR intent.
DEFAULT_PROVIDER_COMMAND_TIMEOUT_SEC = 2.5


def require_rclpy() -> Any:
    """Import ROS only when the optional Nero Provider was explicitly enabled."""
    import rclpy

    return rclpy


class NeroConsoleRuntime:
    """Own a private ROS context while XR keeps ownership of robot control."""

    def __init__(
        self,
        *,
        command_bus: UiCommandBus,
        config_supplier: Callable[[str], dict[str, Any]],
        snapshot_supplier: Callable[[str], dict[str, Any]],
        command_mapper: Callable[[str, str, dict[str, Any]], tuple[UiCommandName, dict[str, Any]] | None],
        stream_supplier: Callable[[str, str], bytes | None] | None = None,
        stream_demand_sink: Callable[[str, str, int], None] | None = None,
        publish_interval_sec: float = 0.2,
    ) -> None:
        self._command_bus = command_bus
        self._config_supplier = config_supplier
        self._snapshot_supplier = snapshot_supplier
        self._command_mapper = command_mapper
        self._stream_supplier = stream_supplier
        self._stream_demand_sink = stream_demand_sink
        self._interval_sec = float(publish_interval_sec)
        self._context = None
        self._node = None
        self._executor = None
        self._thread: threading.Thread | None = None
        self._providers: list[NeroConsoleProvider] = []

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("NeroConsoleRuntime.start() called more than once")
        rclpy = require_rclpy()
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node

        context = Context()
        rclpy.init(context=context)
        node = Node("xr_nero_console_provider", context=context)
        executor = MultiThreadedExecutor(context=context, num_threads=2)
        self._context = context
        self._node = node
        self._executor = executor
        for mode, provider_id in (("collector", "xr_collector"), ("vla", "xr_vla")):
            provider = NeroConsoleProvider(
                node,
                mode=mode,
                provider_id=provider_id,
                command_bus=self._command_bus,
                config_supplier=lambda current_mode=mode: self._config_supplier(current_mode),
                snapshot_supplier=lambda current_mode=mode: self._snapshot_supplier(current_mode),
                command_mapper=lambda action, params, current_mode=mode: self._command_mapper(current_mode, action, params),
                stream_supplier=(
                    None
                    if self._stream_supplier is None
                    else lambda stream_id, current_mode=mode: self._stream_supplier(current_mode, stream_id)
                ),
                stream_demand_sink=(
                    None
                    if self._stream_demand_sink is None
                    else lambda stream_id, subscriber_count, current_provider_id=provider_id: self._stream_demand_sink(
                        current_provider_id,
                        stream_id,
                        subscriber_count,
                    )
                ),
                publish_interval_sec=self._interval_sec,
                command_timeout_sec=DEFAULT_PROVIDER_COMMAND_TIMEOUT_SEC,
            )
            provider.start()
            self._providers.append(provider)
        executor.add_node(node)
        self._thread = threading.Thread(target=executor.spin, name="nero_console_ros", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        for provider in self._providers:
            provider.stop()
        self._providers.clear()
        self._executor.shutdown()
        self._node.destroy_node()
        self._context.shutdown()
        self._thread.join()
        self._thread = None
