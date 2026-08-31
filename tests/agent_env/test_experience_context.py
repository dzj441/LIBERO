import json
from pathlib import Path

import pytest

from libero.libero.agent_env.experience_context import (
    EXPERIENCE_CONTEXT_SPEC_SCHEMA_VERSION,
    ExperienceContextError,
    normalize_experience_context_spec,
    project_experience_context_bundle,
    validate_experience_context_bundle,
)
from test_fixed_demo import _test_master


TARGET_INSTRUCTION = "pick up the alphabet soup and place it in the basket"


def _spec(tmp_path: Path, experiences):
    return normalize_experience_context_spec(
        {
            "schema_version": EXPERIENCE_CONTEXT_SPEC_SCHEMA_VERSION,
            "context_id": "test_context",
            "experiences": experiences,
        },
        artifact_root=tmp_path,
    )


def test_projects_multiple_modalities_without_public_relation_labels(tmp_path):
    _test_master(tmp_path / "master")
    spec = _spec(
        tmp_path,
        [
            {
                "experience_id": "matched_full",
                "source_type": "expert_episode",
                "master_root": "master",
                "relation_to_target": "same_task_separate_episode",
                "modalities": ["actions", "text", "observations", "video"],
                "public_text": "Inspect the experience and decide what transfers.",
            },
            {
                "experience_id": "control_actions",
                "source_type": "expert_episode",
                "master_root": "master",
                "relation_to_target": "irrelevant_task",
                "modalities": ["actions"],
            },
        ],
    )
    bundle = tmp_path / "bundle"
    receipt = project_experience_context_bundle(
        spec=spec,
        destination=bundle,
        profile="level4",
        target_task_instruction=TARGET_INSTRUCTION,
    )
    manifest = validate_experience_context_bundle(
        bundle, expected_profile="level4", expected_experience_count=2
    )

    assert receipt["experiences"][0]["relation_to_target"] == (
        "same_task_separate_episode"
    )
    assert receipt["experiences"][1]["relation_to_target"] == "irrelevant_task"
    assert receipt["experiences"][0]["text_kind"] == "outcome_summary"
    assert receipt["visibility_contract"]["relation_labels_disclosed"] is False
    assert len(receipt["bundle_integrity"]["files"]) > 2
    assert receipt["budget"]["experience_count"] == 2
    assert receipt["budget"]["native_osc_action_count"] > 0
    assert receipt["budget"]["video_duration_s"] > 0
    assert receipt["budget"]["total_bytes"] > 0
    assert manifest["experiences"][0]["modalities"] == [
        "text",
        "video",
        "observations",
        "actions",
    ]
    assert "integrity" not in manifest
    assert "visibility_contract" not in manifest
    full = bundle / "experiences" / "matched_full"
    assert (full / "text/guidance.md").is_file()
    assert (full / "video/head_wrist_rgb.mp4").stat().st_size > 0
    assert (full / "video/contact_sheets/head_rgb.png").is_file()
    assert (full / "video/contact_sheets/wrist_rgb.png").is_file()
    assert not (full / "video/contact_sheets/head_depth.png").exists()
    assert (full / "frames/frame_000000/observation.json").is_file()
    assert (full / "trajectory/actions.jsonl").is_file()
    item_manifest = json.loads((full / "manifest.json").read_text())
    assert item_manifest["content"]["text"]["kind"] == "outcome_summary"
    assert item_manifest["content"]["video"]["duration_s"] > 0
    assert item_manifest["content"]["actions"]["semantics"][
        "native_osc_sequence_compatible"
    ] is True

    action_only = bundle / "experiences" / "control_actions"
    assert (action_only / "trajectory/actions.jsonl").is_file()
    assert not (action_only / "frames").exists()
    assert not (action_only / "video").exists()
    assert not (action_only / "text").exists()

    rendered = "\n".join(
        path.read_text(encoding="utf-8")
        for path in bundle.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
    )
    assert "relation_to_target" not in rendered
    assert "same_task_separate_episode" not in rendered
    assert "irrelevant_task" not in rendered
    assert str(tmp_path) not in rendered
    assert "dataset_path" not in rendered


@pytest.mark.parametrize("profile_number", (1, 2, 3, 4))
def test_observation_context_uses_same_profile_projection(tmp_path, profile_number):
    _test_master(tmp_path / "master")
    spec = _spec(
        tmp_path,
        [
            {
                "experience_id": "observed_episode",
                "master_root": "master",
                "relation_to_target": "same_task_separate_episode",
                "modalities": ["observations"],
            }
        ],
    )
    profile = f"level{profile_number}"
    bundle = tmp_path / "bundle"
    project_experience_context_bundle(
        spec=spec,
        destination=bundle,
        profile=profile,
        target_task_instruction=TARGET_INSTRUCTION,
    )
    validate_experience_context_bundle(bundle, expected_profile=profile)
    initial = json.loads(
        (
            bundle
            / "experiences/observed_episode/frames/frame_000000/observation.json"
        ).read_text(encoding="utf-8")
    )
    later = json.loads(
        (
            bundle
            / "experiences/observed_episode/frames/frame_000001/observation.json"
        ).read_text(encoding="utf-8")
    )
    assert ("annotations" in initial) is (profile_number >= 2)
    assert "annotations" not in later
    assert ("proprioception" in initial) is (profile_number >= 3)
    assert ("depth" in initial["cameras"]["head"]) is (profile_number >= 4)


def test_same_task_relation_rejects_a_different_source_task(tmp_path):
    master = _test_master(tmp_path / "master")
    manifest_path = master / "p4_master_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["instruction"] = "turn on the stove"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spec = _spec(
        tmp_path,
        [
            {
                "experience_id": "wrong_task",
                "master_root": "master",
                "relation_to_target": "same_task_separate_episode",
                "modalities": ["text"],
            }
        ],
    )
    with pytest.raises(ExperienceContextError, match="does not match"):
        project_experience_context_bundle(
            spec=spec,
            destination=tmp_path / "bundle",
            profile="level4",
            target_task_instruction=TARGET_INSTRUCTION,
        )


def test_public_text_cannot_be_added_without_text_modality(tmp_path):
    _test_master(tmp_path / "master")
    with pytest.raises(ExperienceContextError, match="requires the text modality"):
        _spec(
            tmp_path,
            [
                {
                    "experience_id": "invalid",
                    "master_root": "master",
                    "relation_to_target": "irrelevant_task",
                    "modalities": ["actions"],
                    "public_text": "hidden extra channel",
                }
            ],
        )
