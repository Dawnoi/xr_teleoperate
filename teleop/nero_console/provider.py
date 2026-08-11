"""ROS transport for one XR Nero Web Console v1 provider."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from teleop.nero_console.contracts import (
    build_reply,
    build_snapshot,
    provider_namespace,
    validate_command_basics,
)
from teleop.ui.command_bus import (
    UiCommand,
    UiCommandBus,
    UiCommandCompletionStatus,
    UiCommandName,
)


JsonObject = dict[str, Any]
CommandMapper = Callable[[str, JsonObject], tuple[UiCommandName, JsonObject] | None]


class NeroConsoleProvider:
    """Publish one fixed-mode XR provider without owning robot control."""

    def __init__(
        self,
        node: Any,
        *,
        mode: str,
        provider_id: str,
        command_bus: UiCommandBus,
        config_supplier: Callable[[], JsonObject],
        snapshot_supplier: Callable[[], JsonObject],
        command_mapper: CommandMapper,
        stream_supplier: Callable[[str], bytes | None] | None = None,
        stream_demand_sink: Callable[[str, int], None] | None = None,
        publish_interval_sec: float = 0.2,
        command_timeout_sec: float = 10.0,
    ) -> None:
        if mode not in {"collector", "vla"}:
            raise ValueError("Nero provider mode must be collector or vla")
        if float(publish_interval_sec) <= 0.0:
            raise ValueError("Nero provider publish interval must be positive")
        if float(command_timeout_sec) <= 0.0:
            raise ValueError("Nero provider command timeout must be positive")
        self._node = node
        self._mode = mode
        self._provider_id = provider_id
        self._namespace = provider_namespace(provider_id)
        self._command_bus = command_bus
        self._config_supplier = config_supplier
        self._snapshot_supplier = snapshot_supplier
        self._command_mapper = command_mapper
        self._stream_supplier = stream_supplier
        self._stream_demand_sink = stream_demand_sink
        self._pending: dict[str, UiCommand] = {}
        self._stream_publishers: dict[str, Any] = {}
        self._config_pub = None
        self._snapshot_pub = None
        self._reply_pub = None
        self._command_sub = None
        self._timer = None
        self._interval_sec = float(publish_interval_sec)
        self._command_timeout_ns = int(float(command_timeout_sec) * 1_000_000_000)
        self._string_type = None
        self._compressed_image_type = None
        self._stream_qos = None

    def start(self) -> None:
        if self._timer is not None:
            raise RuntimeError(f"Nero provider already started: {self._provider_id}")
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        self._string_type = String
        self._compressed_image_type = CompressedImage
        config_qos = QoSProfile(depth=1)
        config_qos.reliability = ReliabilityPolicy.RELIABLE
        config_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        state_qos = QoSProfile(depth=10)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.VOLATILE
        stream_qos = QoSProfile(depth=1)
        stream_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        stream_qos.durability = DurabilityPolicy.VOLATILE
        self._config_pub = self._node.create_publisher(self._string_type, f"{self._namespace}/config", config_qos)
        self._snapshot_pub = self._node.create_publisher(self._string_type, f"{self._namespace}/snapshot", state_qos)
        self._reply_pub = self._node.create_publisher(self._string_type, f"{self._namespace}/reply", state_qos)
        self._command_sub = self._node.create_subscription(
            self._string_type,
            f"{self._namespace}/command",
            self._on_command,
            state_qos,
        )
        self._stream_qos = stream_qos
        self._publish()
        self._timer = self._node.create_timer(self._interval_sec, self._publish)

    def stop(self) -> None:
        if self._timer is not None:
            self._node.destroy_timer(self._timer)
            self._timer = None
        if self._command_sub is not None:
            self._node.destroy_subscription(self._command_sub)
            self._command_sub = None
        if self._stream_demand_sink is not None:
            for stream_id in self._stream_publishers:
                self._stream_demand_sink(stream_id, 0)
        for publisher in (self._config_pub, self._snapshot_pub, self._reply_pub, *self._stream_publishers.values()):
            if publisher is not None:
                self._node.destroy_publisher(publisher)
        self._config_pub = self._snapshot_pub = self._reply_pub = None
        self._stream_publishers.clear()

    def _message(self, value: Mapping[str, Any]) -> Any:
        if self._string_type is None:
            raise RuntimeError("Nero provider message type is unavailable before start")
        return self._string_type(
            data=json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"))
        )

    def _config(self) -> JsonObject:
        config = self._config_supplier()
        if not isinstance(config, dict):
            raise TypeError("Nero config supplier must return a dict")
        return config

    def _snapshot(self) -> JsonObject:
        supplied = self._snapshot_supplier()
        return build_snapshot(
            mode=self._mode,
            instance_id=self._provider_id,
            state_revision=str(supplied["state_revision"]),
            updated_monotonic_ms=int(supplied.get("updated_monotonic_ms", time.monotonic() * 1000)),
            lifecycle=dict(supplied["lifecycle"]),
            health=dict(supplied["health"]),
            data=dict(supplied["data"]),
        )

    def _publish(self) -> None:
        config = self._config()
        snapshot = self._snapshot()
        if self._config_pub is not None:
            self._config_pub.publish(self._message(config))
        if self._snapshot_pub is not None:
            self._snapshot_pub.publish(self._message(snapshot))
        self._publish_replies(snapshot["state_revision"])
        streams = config.get("streams", [])
        for item in streams:
            stream_id = str(item["stream_id"])
            self._publish_stream(stream_id)

    def _publish_replies(self, state_revision: str) -> None:
        for request_id, command in tuple(self._pending.items()):
            completion = self._command_bus.completion_for(request_id)
            if completion is None:
                if time.monotonic_ns() - command.created_monotonic_ns < self._command_timeout_ns:
                    continue
                self._reply_error(
                    request_id,
                    "provider_timeout",
                    "XR control loop did not complete the command before the provider timeout",
                    state_revision,
                )
                self._command_bus.cancel(request_id)
                del self._pending[request_id]
                self._command_bus.forget_completion(request_id)
                continue
            ok = completion.status == UiCommandCompletionStatus.SUCCEEDED
            reply = build_reply(
                request_id=request_id,
                ok=ok,
                code="" if ok else ("command_failed" if completion.status == UiCommandCompletionStatus.FAILED else "forbidden"),
                message=completion.message,
                state_revision=state_revision,
                result=completion.details or {},
            )
            if self._reply_pub is not None:
                self._reply_pub.publish(self._message(reply))
            del self._pending[request_id]
            self._command_bus.forget_completion(request_id)

    def _publish_stream(self, stream_id: str) -> None:
        if self._stream_supplier is None and self._stream_demand_sink is None:
            return
        publisher = self._stream_publishers.get(stream_id)
        if publisher is None:
            if self._compressed_image_type is None:
                raise RuntimeError("Nero provider image message type is unavailable before start")
            publisher = self._node.create_publisher(
                self._compressed_image_type,
                f"{self._namespace}/streams/{stream_id}",
                self._stream_qos,
            )
            self._stream_publishers[stream_id] = publisher
        subscriber_count = publisher.get_subscription_count()
        if isinstance(subscriber_count, bool) or not isinstance(subscriber_count, int) or subscriber_count < 0:
            raise RuntimeError(f"Nero stream publisher returned invalid subscriber count for {stream_id}")
        if self._stream_demand_sink is not None:
            self._stream_demand_sink(stream_id, subscriber_count)
        if subscriber_count == 0 or self._stream_supplier is None:
            return
        jpeg = self._stream_supplier(stream_id)
        if jpeg is None:
            return
        message = self._compressed_image_type()
        message.format = "jpeg"
        message.data = bytes(jpeg)
        publisher.publish(message)

    def _reply_error(self, request_id: str, code: str, message: str, state_revision: str) -> None:
        if self._reply_pub is not None:
            self._reply_pub.publish(
                self._message(
                    build_reply(
                        request_id=request_id,
                        ok=False,
                        code=code,
                        message=message,
                        state_revision=state_revision,
                    )
                )
            )

    def _on_command(self, message: Any) -> None:
        try:
            command_raw = json.loads(message.data)
        except json.JSONDecodeError as exc:
            snapshot = self._snapshot()
            self._reply_error(
                "__invalid_json__",
                "invalid_command",
                f"command is not valid JSON: {exc.msg}",
                str(snapshot["state_revision"]),
            )
            return
        command = validate_command_basics(command_raw, require_expected_revision=False)
        request_id = str(command["request_id"])
        snapshot = self._snapshot()
        current_revision = str(snapshot["state_revision"])
        action = str(command["action"])
        expected_revision = command.get("expected_revision")
        if action != "runtime.estop" and expected_revision is None:
            self._reply_error(
                request_id,
                "expected_revision_required",
                "expected_revision is required for this action",
                current_revision,
            )
            return
        if action != "runtime.estop" and str(expected_revision) != current_revision:
            self._reply_error(request_id, "state_revision_conflict", "state revision is stale", current_revision)
            return
        mapped = self._command_mapper(action, dict(command["params"]))
        if mapped is None:
            self._reply_error(request_id, "capability_not_supported", "action is not supported by XR", current_revision)
            return
        command_name, payload = mapped
        submitted = self._command_bus.submit(
            command_name,
            source=f"nero:{self._provider_id}",
            payload=payload,
            request_id=request_id,
        )
        self._pending[request_id] = submitted
