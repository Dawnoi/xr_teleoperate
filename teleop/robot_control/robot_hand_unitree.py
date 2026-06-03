# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array_in, right_hand_array_in, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array_in.get_lock():
                    left_hand_data  = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                with right_hand_array_in.get_lock():
                    right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = False, fps = 200.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread

        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.gripper_state_ready = False
        self.simulation_mode = simulation_mode
        self.running = True
        self._health_lock = threading.Lock()
        self._last_gripper_cmd_timestamp = None
        self._last_control_loop_timestamp = None
        self._last_error_message = ""
        self._last_warn_times = {}
        self._startup_time = time.time()
        self._health_monitor_thread = None
        self._last_dual_gripper_action = None
        self._last_dual_gripper_state = None
        self._last_dual_gripper_state_prev = None
        self._stall_start_time = None
        self._contact_latch_active = [False, False]
        self._contact_latch_position = [0.0, 0.0]
        self._contact_latch_names = ["left", "right"]
        
        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None
 
        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()
        self.last_gripper_state_timestamp = None

        # Shared Arrays for gripper states
        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        startup_wait_deadline = time.time() + 3.0
        while not self.gripper_state_ready and time.time() < startup_wait_deadline:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        if self.gripper_state_ready:
            logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")
        else:
            logger_mp.warning(
                "[Dex1_1_Gripper_Controller] No valid gripper state received within 3s. "
                "Controller will continue running with health monitoring enabled."
            )

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in, self.left_gripper_state_value, self.right_gripper_state_value,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        self._health_monitor_thread = threading.Thread(target=self._health_monitor_loop)
        self._health_monitor_thread.daemon = True
        self._health_monitor_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!")

    def _set_error(self, msg: str):
        with self._health_lock:
            self._last_error_message = str(msg)

    def _warn_throttled(self, key: str, message: str, interval_sec: float = 2.0):
        now = time.time()
        with self._health_lock:
            last = float(self._last_warn_times.get(key, 0.0))
            if (now - last) < interval_sec:
                return
            self._last_warn_times[key] = now
        logger_mp.warning(message)

    def _subscribe_gripper_state(self):
        while True:
            try:
                left_gripper_msg  = self.LeftGripperState_subscriber.Read()
                right_gripper_msg  = self.RightGripperState_subscriber.Read()
                self.gripper_sub_ready = True
                if left_gripper_msg is not None and right_gripper_msg is not None:
                    self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                    self.right_gripper_state_value.value = right_gripper_msg.states[0].q
                    self.last_gripper_state_timestamp = time.time()
                    self.gripper_state_ready = True
                time.sleep(0.002)
            except Exception as e:
                self._set_error(f"gripper state subscribe exception: {e}")
                self._warn_throttled(
                    "subscribe_exception",
                    f"[Dex1_1_Gripper_Controller] state subscriber exception: {e}",
                    interval_sec=2.0,
                )
                time.sleep(0.05)

    def get_state_age(self):
        if self.last_gripper_state_timestamp is None:
            return None
        return time.time() - self.last_gripper_state_timestamp

    def get_health_snapshot(self):
        state_age = self.get_state_age()
        with self._health_lock:
            return {
                "subscriber_ready": bool(self.gripper_sub_ready),
                "state_ready": bool(self.gripper_state_ready),
                "subscribe_thread_alive": bool(
                    self.subscribe_state_thread.is_alive() if self.subscribe_state_thread is not None else False
                ),
                "control_thread_alive": bool(
                    self.gripper_control_thread.is_alive() if self.gripper_control_thread is not None else False
                ),
                "state_age_sec": state_age,
                "last_cmd_age_sec": None if self._last_gripper_cmd_timestamp is None else (time.time() - self._last_gripper_cmd_timestamp),
                "last_control_loop_age_sec": None if self._last_control_loop_timestamp is None else (time.time() - self._last_control_loop_timestamp),
                "last_error": self._last_error_message,
                "last_dual_gripper_action": None if self._last_dual_gripper_action is None else list(self._last_dual_gripper_action),
                "last_dual_gripper_state": None if self._last_dual_gripper_state is None else list(self._last_dual_gripper_state),
            }

    def _health_monitor_loop(self):
        while self.running:
            snapshot = self.get_health_snapshot()
            if not snapshot["subscribe_thread_alive"]:
                self._warn_throttled(
                    "subscribe_thread_dead",
                    "[Dex1_1_Gripper_Controller] state subscriber thread is not alive. DDS state updates are lost.",
                    interval_sec=2.0,
                )
            if not snapshot["control_thread_alive"]:
                self._warn_throttled(
                    "control_thread_dead",
                    "[Dex1_1_Gripper_Controller] control thread is not alive. Gripper command publishing has stopped.",
                    interval_sec=2.0,
                )
            if not snapshot["state_ready"] and (time.time() - self._startup_time) > 3.0:
                self._warn_throttled(
                    "no_initial_state",
                    "[Dex1_1_Gripper_Controller] still no valid gripper state after startup. Check rt/dex1/*/state DDS topics, power, and wiring.",
                    interval_sec=2.0,
                )
            state_age = snapshot["state_age_sec"]
            if state_age is not None and state_age > 0.5:
                self._warn_throttled(
                    "stale_state",
                    f"[Dex1_1_Gripper_Controller] gripper state stale for {state_age:.2f}s. "
                    "Likely DDS topic stalled or hardware feedback is offline.",
                    interval_sec=1.0,
                )
            action = snapshot["last_dual_gripper_action"]
            state = snapshot["last_dual_gripper_state"]
            if (
                state_age is not None
                and state_age <= 0.2
                and action is not None
                and state is not None
                and len(action) == 2
                and len(state) == 2
            ):
                action_arr = np.asarray(action, dtype=float)
                state_arr = np.asarray(state, dtype=float)
                err = np.abs(action_arr - state_arr)
                state_prev = None if self._last_dual_gripper_state_prev is None else np.asarray(self._last_dual_gripper_state_prev, dtype=float)
                motion = 0.0 if state_prev is None else float(np.max(np.abs(state_arr - state_prev)))
                large_error = float(np.max(err)) > 0.35
                almost_still = motion < 0.01
                if large_error and almost_still:
                    if self._stall_start_time is None:
                        self._stall_start_time = time.time()
                    elif (time.time() - self._stall_start_time) > 0.4:
                        self._warn_throttled(
                            "possible_stall_or_protection",
                            "[Dex1_1_Gripper_Controller] gripper state is still updating, but commanded closing/opening "
                            "remains far from measured position while motion is nearly stalled. "
                            "This looks like contact stall, motor current limit, or driver protection rather than pure DDS disconnect.",
                            interval_sec=1.0,
                        )
                else:
                    self._stall_start_time = None
                self._last_dual_gripper_state_prev = state_arr.copy()
            time.sleep(0.2)
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        with self._health_lock:
            self._last_gripper_cmd_timestamp = time.time()
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, left_gripper_state_value, right_gripper_state_value, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True
        DELTA_GRIPPER_CMD = 0.80    # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd
        while self.running:
            try:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value
                # get current dual gripper motor state
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])
                
                if left_gripper_value != 0.0 or right_gripper_value != 0.0: # if input data has been initialized.
                    # Linear mapping from [0, THUMB_INDEX_DISTANCE_MAX] to gripper action range
                    left_target_action  = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])
                raw_target_action = np.array([left_target_action, right_target_action], dtype=float)
                # clip dual gripper action to avoid overflow
                if not self.simulation_mode:
                    left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                    right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)
                else:
                    left_actual_action  = left_target_action
                    right_actual_action = right_target_action
                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if not self.simulation_mode:
                    stall_err_thresh = 0.35
                    stall_motion_thresh = 0.01
                    close_intent_margin = 0.08
                    release_open_margin = 0.12
                    if self._last_dual_gripper_state_prev is None:
                        motion_vec = np.zeros(2, dtype=float)
                    else:
                        motion_vec = np.abs(
                            dual_gripper_state - np.asarray(self._last_dual_gripper_state_prev, dtype=float)
                        )
                    err_vec = raw_target_action - dual_gripper_state

                    for idx in range(2):
                        large_error = abs(err_vec[idx]) > stall_err_thresh
                        almost_still = motion_vec[idx] < stall_motion_thresh
                        closing_intent = raw_target_action[idx] < (dual_gripper_state[idx] - close_intent_margin)

                        if self._contact_latch_active[idx]:
                            latched_pos = self._contact_latch_position[idx]
                            reopen_requested = raw_target_action[idx] > (latched_pos + release_open_margin)
                            if reopen_requested:
                                self._contact_latch_active[idx] = False
                                logger_mp.info(
                                    f"[Dex1_1_Gripper_Controller] {self._contact_latch_names[idx]} gripper contact latch released. "
                                    "Operator requested reopening away from the contact point."
                                )
                            else:
                                dual_gripper_action[idx] = dual_gripper_state[idx]
                        elif closing_intent and large_error and almost_still:
                            self._contact_latch_active[idx] = True
                            self._contact_latch_position[idx] = float(dual_gripper_state[idx])
                            dual_gripper_action[idx] = dual_gripper_state[idx]
                            logger_mp.warning(
                                f"[Dex1_1_Gripper_Controller] {self._contact_latch_names[idx]} gripper contact latch engaged at "
                                f"q={dual_gripper_state[idx]:.3f}. Freezing command at current position to avoid "
                                "continuous hard-contact pushing and motor protection. Open slightly to release."
                            )

                with self._health_lock:
                    self._last_dual_gripper_state = dual_gripper_state.copy()
                    self._last_dual_gripper_action = dual_gripper_action.copy()

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                with self._health_lock:
                    self._last_control_loop_timestamp = time.time()
                current_time = time.time()
                time_elapsed = current_time - start_time
                self._last_dual_gripper_state_prev = dual_gripper_state.copy()
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                self._set_error(f"gripper control loop exception: {e}")
                logger_mp.exception("[Dex1_1_Gripper_Controller] control loop exception")
                time.sleep(0.05)

class Gripper_JointIndex(IntEnum):
    kGripper = 0
