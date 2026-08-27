import pytest

from libero.libero.agent_env.observation import infer_annotation_roles


def test_infer_pick_place_roles_from_goal_region():
    parsed = {
        "objects": {"soup": ["soup_1"], "basket": ["basket_1"]},
        "fixtures": {"floor": ["floor"]},
        "regions": {"basket_1_inside": {"target": "basket_1"}},
        "goal_state": [["in", "soup_1", "basket_1_inside"]],
        "obj_of_interest": ["soup_1", "basket_1"],
    }
    roles = infer_annotation_roles(parsed)
    assert roles.manipulated_object == "soup_1"
    assert roles.goal_fixture == "basket_1"


def test_infer_drawer_bowl_roles_from_cabinet_region():
    parsed = {
        "objects": {"akita_black_bowl": ["akita_black_bowl_1"]},
        "fixtures": {"wooden_cabinet": ["wooden_cabinet_1"]},
        "regions": {
            "wooden_cabinet_1_top_region": {"target": "wooden_cabinet_1"}
        },
        "goal_state": [
            ["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]
        ],
        "obj_of_interest": ["akita_black_bowl_1", "wooden_cabinet_1"],
    }
    roles = infer_annotation_roles(parsed)
    assert roles.manipulated_object == "akita_black_bowl_1"
    assert roles.goal_fixture == "wooden_cabinet_1"


def test_ambiguous_articulation_task_fails_closed():
    parsed = {
        "objects": {},
        "fixtures": {"cabinet": ["cabinet_1"]},
        "regions": {},
        "goal_state": [["open", "cabinet_1_top_region"]],
        "obj_of_interest": ["cabinet_1"],
    }
    with pytest.raises(ValueError, match="pass AnnotationRoles explicitly"):
        infer_annotation_roles(parsed)
