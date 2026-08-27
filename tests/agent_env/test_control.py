import numpy as np
import pytest

from libero.libero.agent_env.control import (
    MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION,
    EEFCommand,
    EEFExecution,
    NativeOSCSequenceExecutor,
    OSCControlConfig,
    _gripper_internal_target_width_m,
    _set_gripper_internal_target_width_m,
    _gripper_width_limits_m,
    limit_vector_norm,
    matrix_to_rotation_vector,
    normalized_osc_action,
    rotation_vector_to_matrix,
    validate_native_osc_sequence,
)


def test_execution_metadata_separates_motion_gripper_and_overall_completion():
    execution = EEFExecution(
        command_completed=False,
        target_reached=True,
        gripper_command_completed=False,
        termination_reason="gripper_control_step_budget_exhausted",
        position_error_m=0.001,
        orientation_error_rad=0.01,
        control_steps=12,
        motion_control_steps=2,
        gripper_control_steps=8,
        settle_control_steps=2,
        requested_gripper_width_m=0.03,
        commanded_gripper_width_m=0.02,
        actual_gripper_width_m=0.025,
    )

    assert execution.to_public_dict() == {
        "command_completed": False,
        "target_reached": True,
        "gripper_command_completed": False,
        "termination_reason": "gripper_control_step_budget_exhausted",
        "position_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "control_steps": 12,
        "motion_control_steps": 2,
        "gripper_control_steps": 8,
        "settle_control_steps": 2,
        "requested_gripper_width_m": 0.03,
        "commanded_gripper_width_m": 0.02,
        "actual_gripper_width_m": 0.025,
    }


def test_high_level_command_has_no_xyz_or_rotation_magnitude_cap():
    command = EEFCommand.create(
        delta_position_m=[0.2, -0.1, 0.3],
        delta_rotation_rotvec_rad=[1.5, 0.0, -0.5],
        delta_gripper_width_m=-0.01,
    )
    np.testing.assert_allclose(command.delta_position_m, [0.2, -0.1, 0.3])
    np.testing.assert_allclose(command.delta_rotation_rotvec_rad, [1.5, 0.0, -0.5])
    assert command.delta_gripper_width_m == -0.01


def test_gripper_delta_must_be_finite():
    with pytest.raises(ValueError, match="must be a finite number"):
        EEFCommand.create(delta_gripper_width_m=np.inf)


@pytest.mark.parametrize(
    "rotation_vector",
    (
        [0.0, 0.0, 0.0],
        [0.2, -0.1, 0.3],
        [np.pi - 1.0e-7, 0.0, 0.0],
    ),
)
def test_rotation_vector_matrix_round_trip(rotation_vector):
    rotation_vector = np.asarray(rotation_vector)
    reconstructed = matrix_to_rotation_vector(
        rotation_vector_to_matrix(rotation_vector)
    )
    np.testing.assert_allclose(
        rotation_vector_to_matrix(reconstructed),
        rotation_vector_to_matrix(rotation_vector),
        atol=1.0e-7,
    )


def test_internal_substep_maps_to_normalized_osc_without_clipping():
    config = OSCControlConfig()
    action = normalized_osc_action(
        [0.04, 0.0, 0.0], [0.0, -0.35, 0.0], 1.0, config
    )
    np.testing.assert_allclose(action, [0.8, 0.0, 0.0, 0.0, -0.7, 0.0, 1.0])


def test_internal_adapter_rejects_accidental_oversized_native_action():
    with pytest.raises(ValueError, match="exceeds normalized range"):
        normalized_osc_action(
            [0.051, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, OSCControlConfig()
        )


@pytest.mark.parametrize(
    "actions, message",
    (
        ([], "between 1 and 20"),
        ([[0.0] * 7] * 21, "between 1 and 20"),
        ([[0.0] * 6], r"shape \[N, 7\]"),
        ([[0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0]], "finite"),
        ([[0.0, 0.0, 0.0, 0.0, 0.0, 1.01, 0.0]], r"within \[-1, 1\]"),
    ),
)
def test_native_osc_sequence_validation_rejects_invalid_batches(actions, message):
    with pytest.raises(ValueError, match=message):
        validate_native_osc_sequence(actions)


class _FakeNativeEnv:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return {"native_index": len(self.actions)}, 0.0, False, {}


def test_native_osc_sequence_executes_exactly_one_policy_interval_per_vector():
    env = _FakeNativeEnv()
    observed = []
    executor = NativeOSCSequenceExecutor(env, observed.append)
    actions = np.zeros((MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION, 7))
    actions[:, 0] = np.linspace(-1.0, 1.0, len(actions))
    actions[:, -1] = 1.0

    final_observation, execution = executor.execute(actions)

    assert final_observation == {"native_index": len(actions)}
    assert len(env.actions) == len(actions)
    assert len(observed) == len(actions)
    np.testing.assert_allclose(np.asarray(env.actions), actions)
    assert execution.to_public_dict() == {
        "command_completed": True,
        "termination_reason": "sequence_executed",
        "micro_step_count": len(actions),
        "control_steps": len(actions),
    }


def test_invalid_native_sequence_does_not_advance_physics():
    env = _FakeNativeEnv()
    executor = NativeOSCSequenceExecutor(env)
    with pytest.raises(ValueError, match="within"):
        executor.execute([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
    assert env.actions == []


def test_limit_vector_norm_preserves_direction():
    limited = limit_vector_norm([3.0, 4.0, 0.0], 2.0)
    np.testing.assert_allclose(limited, [1.2, 1.6, 0.0])


class _FakeModel:
    actuator_ctrlrange = np.array([[0.0, 0.04], [-0.04, 0.0]])

    @staticmethod
    def actuator_name2id(name):
        return {"left": 0, "right": 1}[name]


class _FakeGripper:
    actuators = ["left", "right"]
    current_action = np.array([0.5, -0.5])


class _FakeData:
    qpos = np.array([0.03, -0.03])


class _FakeSim:
    model = _FakeModel()
    data = _FakeData()


class _FakeRobot:
    gripper = _FakeGripper()
    sim = _FakeSim()
    _ref_gripper_joint_pos_indexes = [0, 1]


def test_panda_gripper_width_uses_physical_actuator_targets():
    robot = _FakeRobot()
    assert _gripper_width_limits_m(robot) == (0.0, 0.08)
    assert _gripper_internal_target_width_m(robot) == pytest.approx(0.06)
    _set_gripper_internal_target_width_m(robot, 0.031)
    assert _gripper_internal_target_width_m(robot) == pytest.approx(0.031)
