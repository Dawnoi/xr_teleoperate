import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.input.base import BaseCommandIntent
import core.camera
import core.control
import data_pipeline
import data_pipeline.audit
import data_pipeline.recording
import teleop.debug
import teleop.robot_control

from teleop.control_flow.base_command import apply_base_command, resolve_runtime_base_command_source
from teleop.real.args import parse_args


def _install_module(name, **attributes):
    module = ModuleType(name)
    module.__path__ = []
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    sys.modules[name] = module

    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is None:
            _install_module(parent_name)
            parent = sys.modules[parent_name]
        setattr(parent, child_name, module)


_install_module("core.control.g1d_agv_bridge", G1DAgvBridge=object)


class _MotionSwitcher:
    def Enter_Debug_Mode(self):
        return 0, None


_install_module(
    "core.control.motion_switcher",
    LocoClientWrapper=object,
    MotionSwitcher=_MotionSwitcher,
)
_install_module(
    "teleop.robot_control.robot_arm",
    G1_23_ArmController=object,
    G1_29_ArmController=object,
    H1_2_ArmController=object,
    H1_ArmController=object,
    H2_ArmController=object,
)
_install_module(
    "teleop.robot_control.robot_arm_ik",
    G1_23_ArmIK=object,
    G1_29_ArmIK=object,
    H1_2_ArmIK=object,
    H1_ArmIK=object,
    H2_ArmIK=object,
)
_install_module(
    "unitree_sdk2py.core.channel",
    ChannelFactoryInitialize=object,
    ChannelPublisher=object,
)
_install_module("unitree_sdk2py.idl.std_msgs.msg.dds_", String_=object)

from teleop.real.setup import RealTeleopComponents, setup_base


class _Log:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _args(*, base_controller, motion=True, base_motion=False):
    return SimpleNamespace(
        input_mode="controller",
        motion=motion,
        base_motion=base_motion,
        base_controller=base_controller,
        base_command_source="controller",
        base_stick_deadzone=0.0,
        base_max_vx=0.3,
        base_max_vy=0.3,
        base_max_wz=0.3,
        base_max_z=1.0,
        network_interface=None,
    )


def _tele_data():
    return SimpleNamespace(
        right_ctrl_aButton=False,
        left_ctrl_thumbstick=False,
        right_ctrl_thumbstick=False,
        left_ctrl_thumbstickValue=(0.2, 0.4),
        right_ctrl_thumbstickValue=(0.1, 0.5),
    )


def test_parser_exposes_explicit_base_controller_backends():
    default_args = parse_args([])

    assert default_args.base_controller == "none"
    assert default_args.base_motion is False
    assert parse_args(["--base-controller", "g1d_agv", "--base-motion"]).base_motion is True
    assert parse_args(["--base-controller", "loco"]).base_controller == "loco"
    assert parse_args(["--base-controller", "g1d_agv"]).base_controller == "g1d_agv"


def test_base_motion_routes_g1d_agv_when_arm_motion_is_disabled():
    loco_wrapper = Mock()
    agv_bridge = Mock()
    agv_bridge.get_timing_snapshot.return_value = {}

    result = apply_base_command(
        args=_args(base_controller="g1d_agv", motion=False, base_motion=True),
        tele_data=_tele_data(),
        home_return_active=False,
        loco_wrapper=loco_wrapper,
        agv_bridge=agv_bridge,
    )

    agv_bridge.set_target.assert_called_once_with(0.12, 0.0, -0.06, 0.5)
    loco_wrapper.Move.assert_not_called()
    assert result.base_control_mode == "g1d_agv_async"


def test_none_backend_never_sends_a_base_command():
    loco_wrapper = Mock()
    agv_bridge = Mock()

    result = apply_base_command(
        args=_args(base_controller="none", motion=False, base_motion=True),
        tele_data=_tele_data(),
        home_return_active=False,
        loco_wrapper=loco_wrapper,
        agv_bridge=agv_bridge,
    )

    loco_wrapper.Move.assert_not_called()
    agv_bridge.set_target.assert_not_called()
    assert result.base_control_mode == "none"


def test_base_motion_flag_is_required_before_selected_backend_can_send():
    agv_bridge = Mock()
    agv_bridge.get_timing_snapshot.return_value = {}

    result = apply_base_command(
        args=_args(base_controller="g1d_agv", motion=True, base_motion=False),
        tele_data=_tele_data(),
        home_return_active=False,
        agv_bridge=agv_bridge,
    )

    agv_bridge.set_target.assert_not_called()
    assert result.base_control_mode == "none"


def test_loco_backend_still_sends_controller_velocity_command():
    loco_wrapper = Mock()

    result = apply_base_command(
        args=_args(base_controller="loco", motion=False, base_motion=True),
        tele_data=_tele_data(),
        home_return_active=False,
        loco_wrapper=loco_wrapper,
    )

    loco_wrapper.Move.assert_called_once_with(-0.12, -0.06, -0.03)
    assert result.base_control_mode == "loco"


def test_provider_source_waits_without_sending_until_provider_is_active():
    args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
    args.base_command_source = "provider"
    agv_bridge = Mock()
    events = []
    agv_bridge.stop_sync.side_effect = lambda: (events.append("stop_sync"), "OK STOP 0 0")[1]
    agv_bridge.set_target.side_effect = lambda *command: events.append("set_target")

    result = apply_base_command(
        args=args,
        tele_data=_tele_data(),
        home_return_active=False,
        agv_bridge=agv_bridge,
        base_provider_active=False,
    )

    agv_bridge.set_target.assert_not_called()
    agv_bridge.stop_sync.assert_called_once_with()
    assert events == ["stop_sync"]
    assert result.base_control_mode == "provider_inactive"


def test_provider_source_requires_base_intent_once_provider_is_active():
    args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
    args.base_command_source = "provider"
    agv_bridge = Mock()
    events = []
    agv_bridge.stop_sync.side_effect = lambda: (events.append("stop_sync"), "OK STOP 0 0")[1]
    agv_bridge.set_target.side_effect = lambda *command: events.append("set_target")

    with pytest.raises(RuntimeError, match="requires sample.base_intent"):
        apply_base_command(
            args=args,
            tele_data=_tele_data(),
            home_return_active=False,
            agv_bridge=agv_bridge,
            base_provider_active=True,
        )

    agv_bridge.stop_sync.assert_called_once_with()
    agv_bridge.set_target.assert_not_called()
    assert events == ["stop_sync"]


def test_runtime_provider_override_routes_raw_replay_base_without_changing_xr_default():
    args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
    agv_bridge = Mock()
    agv_bridge.get_timing_snapshot.return_value = {}
    replay_intent = BaseCommandIntent(vx=0.11, vy=-0.22, wz=0.23, z=-0.44, source="raw_episode")

    result = apply_base_command(
        args=args,
        tele_data=_tele_data(),
        home_return_active=False,
        agv_bridge=agv_bridge,
        base_intent=replay_intent,
        base_provider_active=True,
        base_command_source="provider",
    )

    agv_bridge.set_target.assert_called_once_with(0.11, -0.22, 0.23, -0.44)
    assert args.base_command_source == "controller"
    assert result.base_control_mode == "g1d_agv_async"


def test_provider_source_rejects_intent_over_configured_axis_limit_before_send():
    for axis, limit_name in (
        ("vx", "base_max_vx"),
        ("vy", "base_max_vy"),
        ("wz", "base_max_wz"),
        ("z", "base_max_z"),
    ):
        args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
        args.base_command_source = "provider"
        agv_bridge = Mock()
        events = []
        agv_bridge.stop_sync.side_effect = lambda: (events.append("stop_sync"), "OK STOP 0 0")[1]
        agv_bridge.set_target.side_effect = lambda *command: events.append("set_target")
        intent_values = {axis: getattr(args, limit_name) + 0.01}

        with pytest.raises(ValueError, match=f"{axis}.*{limit_name}"):
            apply_base_command(
                args=args,
                tele_data=_tele_data(),
                home_return_active=False,
                agv_bridge=agv_bridge,
                base_intent=BaseCommandIntent(**intent_values, source="test"),
                base_provider_active=True,
            )

        agv_bridge.stop_sync.assert_called_once_with()
        agv_bridge.set_target.assert_not_called()
        assert events == ["stop_sync"]


def test_provider_source_routes_base_intent_to_selected_backend():
    args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
    args.base_command_source = "provider"
    agv_bridge = Mock()
    agv_bridge.get_timing_snapshot.return_value = {}

    result = apply_base_command(
        args=args,
        tele_data=_tele_data(),
        home_return_active=False,
        agv_bridge=agv_bridge,
        base_intent=BaseCommandIntent(vx=0.1, vy=-0.2, wz=0.3, z=-0.4, source="test"),
        base_provider_active=True,
    )

    agv_bridge.set_target.assert_called_once_with(0.1, -0.2, 0.3, -0.4)
    assert result.base_control_mode == "g1d_agv_async"


def test_ui_online_inference_uses_provider_base_source():
    assert resolve_runtime_base_command_source(
        active_input_provider="online_inference",
        is_ui_raw_replay=False,
        raw_replay_base_source="none",
    ) == "provider"


def test_ui_provider_stop_restores_xr_controller_source_without_mutating_args():
    args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
    agv_bridge = Mock()
    agv_bridge.get_timing_snapshot.return_value = {}
    provider_intent = BaseCommandIntent(vx=0.1, vy=-0.2, wz=0.3, z=-0.4, source="online_inference")

    provider_result = apply_base_command(
        args=args,
        tele_data=_tele_data(),
        home_return_active=False,
        agv_bridge=agv_bridge,
        base_intent=provider_intent,
        base_provider_active=True,
        base_stop_latched=True,
        base_command_source=resolve_runtime_base_command_source(
            active_input_provider="online_inference",
            is_ui_raw_replay=False,
            raw_replay_base_source="none",
        ),
    )
    assert provider_result.base_control_mode == "g1d_agv_async"
    assert args.base_command_source == "controller"

    agv_bridge.reset_mock()
    agv_bridge.get_timing_snapshot.return_value = {}
    controller_result = apply_base_command(
        args=args,
        tele_data=_tele_data(),
        home_return_active=False,
        agv_bridge=agv_bridge,
        base_stop_latched=True,
        base_command_source=resolve_runtime_base_command_source(
            active_input_provider="xr",
            is_ui_raw_replay=False,
            raw_replay_base_source="none",
        ),
    )

    agv_bridge.set_target.assert_called_once_with(0.12, 0.0, -0.06, 0.5)
    assert controller_result.base_control_mode == "g1d_agv_async"


def test_ui_raw_replay_action_keeps_provider_override_and_other_replay_sources_unchanged():
    assert resolve_runtime_base_command_source(
        active_input_provider="lerobot_offline",
        is_ui_raw_replay=True,
        raw_replay_base_source="action",
    ) == "provider"
    assert resolve_runtime_base_command_source(
        active_input_provider="lerobot_offline",
        is_ui_raw_replay=True,
        raw_replay_base_source="none",
    ) is None


def test_setup_initializes_g1d_bridge_without_initializing_loco():
    args = _args(base_controller="g1d_agv", motion=False, base_motion=True)
    components = RealTeleopComponents()
    bridge = Mock()

    with patch("teleop.real.setup.LocoClientWrapper") as loco_cls, patch(
        "teleop.real.setup.G1DAgvBridge", return_value=bridge
    ) as bridge_cls:
        setup_base(args, components, _Log())

    loco_cls.assert_not_called()
    bridge_cls.assert_called_once_with(network_interface=None, auto_build=True)
    assert components.loco_wrapper is None
    assert components.agv_bridge is bridge


def test_setup_initializes_loco_only_for_loco_backend():
    args = _args(base_controller="loco", motion=False, base_motion=True)
    components = RealTeleopComponents()
    loco = Mock()

    with patch("teleop.real.setup.LocoClientWrapper", return_value=loco) as loco_cls, patch(
        "teleop.real.setup.G1DAgvBridge"
    ) as bridge_cls:
        setup_base(args, components, _Log())

    loco_cls.assert_called_once_with()
    bridge_cls.assert_not_called()
    assert components.loco_wrapper is loco
    assert components.agv_bridge is None


def test_setup_does_not_initialize_any_base_backend_for_none():
    args = _args(base_controller="none", motion=False, base_motion=True)
    components = RealTeleopComponents()

    with patch("teleop.real.setup.LocoClientWrapper") as loco_cls, patch(
        "teleop.real.setup.G1DAgvBridge"
    ) as bridge_cls:
        setup_base(args, components, _Log())

    loco_cls.assert_not_called()
    bridge_cls.assert_not_called()
    assert components.loco_wrapper is None
    assert components.agv_bridge is None
