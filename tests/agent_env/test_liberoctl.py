import json

import pytest

from scripts.liberoctl import (
    METRIC_OSC_STEP,
    NATIVE_OSC_SEQUENCE,
    bind_current_observation_id,
    parse_args,
    request_for_args,
)


def test_metric_client_exposes_only_metric_action_command():
    args = parse_args(
        [
            "osc-step",
            "--position",
            "0.1",
            "0",
            "-0.1",
            "--rotation",
            "0",
            "0.2",
            "0",
            "--gripper-delta-m",
            "-0.01",
        ],
        action_interface=METRIC_OSC_STEP,
    )
    assert request_for_args(args) == {
        "command": "osc_step",
        "delta_position_m": [0.1, 0.0, -0.1],
        "delta_rotation_rotvec_rad": [0.0, 0.2, 0.0],
        "delta_gripper_width_m": -0.01,
    }
    with pytest.raises(SystemExit):
        parse_args(
            ["osc-sequence", "--actions-file", "actions.json"],
            action_interface=METRIC_OSC_STEP,
        )


def test_native_client_reads_actions_locally_and_exposes_only_sequence(tmp_path):
    actions = [[0.1, 0.0, 0.0, 0.0, -0.2, 0.0, 1.0]]
    actions_file = tmp_path / "actions.json"
    actions_file.write_text(json.dumps(actions), encoding="utf-8")
    args = parse_args(
        ["osc-sequence", "--actions-file", str(actions_file)],
        action_interface=NATIVE_OSC_SEQUENCE,
    )
    assert request_for_args(args) == {
        "command": "osc_sequence",
        "actions": actions,
    }
    with pytest.raises(SystemExit):
        parse_args(
            ["osc-step", "--position", "0", "0", "0"],
            action_interface=NATIVE_OSC_SEQUENCE,
        )


def test_client_automatically_binds_nonstart_requests_to_current_observation(tmp_path):
    observation = tmp_path / "observation.json"
    observation.write_text(
        json.dumps({"observation_id": "obs_000123"}), encoding="utf-8"
    )

    bound = bind_current_observation_id(
        {"command": "finish"}, observation_file=observation
    )
    assert bound == {"command": "finish", "observation_id": "obs_000123"}
    assert bind_current_observation_id(
        {"command": "start"}, observation_file=tmp_path / "missing.json"
    ) == {"command": "start"}


def test_client_rejects_a_current_observation_without_an_id(tmp_path):
    observation = tmp_path / "observation.json"
    observation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="observation_id"):
        bind_current_observation_id(
            {"command": "osc_step"}, observation_file=observation
        )
