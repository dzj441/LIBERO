import pytest

from libero.libero.agent_env.runtime_contract import (
    build_curriculum_server_ready_contract,
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


def test_ready_contract_commits_optional_external_task_source_fingerprint():
    fingerprint = {
        "schema_version": "libero.robomemarena_source.v1",
        "source_commit": "a" * 40,
        "bddl_sha256": "b" * 64,
    }
    contract = _contract(task_source_fingerprint=fingerprint)
    assert contract["task_source_fingerprint"] == fingerprint
    changed = _contract(
        task_source_fingerprint={**fingerprint, "bddl_sha256": "c" * 64}
    )
    assert canonical_json_sha256(contract) != canonical_json_sha256(changed)


def test_curriculum_ready_contract_commits_order_without_plaintext_tasks():
    episodes = [
        {
            "suite": "libero_90",
            "task_id": 7,
            "init_state_id": 17,
            "seed": 11,
            "task_instruction": "open the top drawer of the cabinet",
            "max_agent_steps": 50,
            "icl_condition": "fixed_demo",
            "fixed_demo_master_manifest_sha256": "a" * 64,
        },
        {
            "suite": "libero_goal",
            "task_id": 3,
            "init_state_id": 22,
            "seed": 12,
            "task_instruction": "open the top drawer and put the bowl inside",
            "max_agent_steps": 100,
            "icl_condition": "none",
            "fixed_demo_master_manifest_sha256": None,
        },
    ]
    contract = build_curriculum_server_ready_contract(
        episodes=episodes,
        profile="p4",
        resolution=256,
        render_gpu_device_id=0,
        initial_settle_control_steps=10,
        max_agent_steps=50,
        action_interface="native_osc_sequence",
    )
    assert contract["schema_version"] == (
        "libero.agent_curriculum_server_ready.v1"
    )
    assert contract["episode_count"] == 2
    assert contract["next_task_disclosure"] == "start_response_only"
    assert contract["episodes"][0]["task_id"] == 7
    assert contract["episode_max_agent_steps"] == [50, 100]
    assert "task_instruction" not in contract["episodes"][0]
    assert len(contract["episodes"][0]["task_instruction_sha256"]) == 64
