import pytest

from libero.libero.agent_env.runtime_contract import (
    build_server_ready_contract,
    canonical_json_sha256,
    validate_server_ready_contract,
)


def _contract(**overrides):
    arguments = {
        "suite": "libero_object",
        "task_id": 0,
        "init_state_id": 3,
        "task_instruction": "pick up the alphabet soup and place it in the basket",
        "profile": "p4",
        "seed": 17,
        "resolution": 256,
        "render_gpu_device_id": 0,
        "initial_settle_control_steps": 10,
        "max_agent_steps": 50,
        "action_interface": "native_osc_sequence",
    }
    arguments.update(overrides)
    return build_server_ready_contract(**arguments)


def test_ready_contract_normalizes_and_commits_every_runtime_boundary():
    contract = _contract()
    assert contract["schema_version"] == "libero.agent_server_ready.v1"
    assert contract["protocol"] == "libero.agent_unix_socket.v1"
    assert contract["observation_profile"] == "level4"
    assert contract["accepted_operations"] == [
        "start",
        "osc_sequence",
        "finish",
    ]
    assert contract["max_native_osc_micro_steps_per_submission"] == 20
    assert contract["observation_publication"] == (
        "atomic_replace_before_response"
    )
    assert len(contract["task_instruction_sha256"]) == 64


def test_ready_contract_hash_is_canonical_and_configuration_sensitive():
    first = _contract(profile="p4")
    alias = _contract(profile="level4")
    changed = _contract(profile="level3")
    assert canonical_json_sha256(first) == canonical_json_sha256(alias)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)


def test_ready_contract_rejects_host_server_drift_with_safe_field_names():
    expected = _contract()
    actual = _contract(max_agent_steps=49)
    with pytest.raises(RuntimeError, match="max_agent_steps"):
        validate_server_ready_contract(actual, expected)
