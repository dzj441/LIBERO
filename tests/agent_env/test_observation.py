import numpy as np
import pytest

from libero.libero.agent_env.observation import (
    MasterObservationCollector,
    TaskEntitySelection,
    infer_task_entities,
)
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem


def test_pick_place_selects_every_object_of_interest_without_roles():
    parsed = {
        "objects": {"soup": ["soup_1"], "basket": ["basket_1"]},
        "fixtures": {"floor": ["floor"]},
        "regions": {"basket_1_inside": {"target": "basket_1"}},
        "goal_state": [["in", "soup_1", "basket_1_inside"]],
        "obj_of_interest": ["soup_1", "basket_1"],
    }
    entities = infer_task_entities(parsed)
    assert entities.instance_names == ("soup_1", "basket_1")
    assert entities.anonymous_instances() == (
        ("entity_000", "soup_1"),
        ("entity_001", "basket_1"),
    )


def test_single_fixture_articulation_task_needs_no_oracle_subpart_role():
    parsed = {
        "objects": {},
        "fixtures": {"cabinet": ["cabinet_1"]},
        "regions": {},
        "goal_state": [["open", "cabinet_1_top_region"]],
        "obj_of_interest": ["cabinet_1"],
    }
    entities = infer_task_entities(parsed)
    assert entities.instance_names == ("cabinet_1",)
    assert entities.anonymous_instances() == (("entity_000", "cabinet_1"),)


def test_goal_region_resolves_only_to_its_host_fixture():
    parsed = {
        "objects": {"bowl": ["bowl_1"]},
        "fixtures": {"cabinet": ["cabinet_1"]},
        "regions": {"cabinet_1_top_region": {"target": "cabinet_1"}},
        "obj_of_interest": ["bowl_1", "cabinet_1_top_region"],
    }
    entities = infer_task_entities(parsed)
    assert entities.instance_names == ("bowl_1", "cabinet_1")


def test_multiple_entities_preserve_bddl_order_and_remove_duplicates():
    parsed = {
        "objects": {"item": ["item_1", "item_2"]},
        "fixtures": {"cabinet": ["cabinet_1"]},
        "obj_of_interest": ["item_2", "cabinet_1", "item_1", "item_2"],
    }
    entities = infer_task_entities(parsed)
    assert entities.instance_names == ("item_2", "cabinet_1", "item_1")


def test_missing_or_unknown_task_entities_fail_closed():
    with pytest.raises(ValueError, match="no valid obj_of_interest"):
        infer_task_entities({"objects": {}, "fixtures": {}, "obj_of_interest": []})
    with pytest.raises(ValueError, match="does not resolve to a known object"):
        infer_task_entities(
            {
                "objects": {"item": ["item_1"]},
                "fixtures": {},
                "obj_of_interest": ["private_unknown"],
            }
        )


def test_collector_keeps_a_valid_task_reference_copy():
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    collector = MasterObservationCollector(
        object(),
        camera_height=3,
        camera_width=4,
        task_entities=TaskEntitySelection(("object_1",)),
        task_reference_rgb=source,
    )

    assert collector.task_reference_rgb is not source
    np.testing.assert_array_equal(collector.task_reference_rgb, source)


@pytest.mark.parametrize(
    "reference",
    (
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 1), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
        [[[]]],
    ),
)
def test_collector_rejects_invalid_task_reference_rgb(reference):
    with pytest.raises((TypeError, ValueError)):
        MasterObservationCollector(
            object(),
            camera_height=3,
            camera_width=4,
            task_entities=TaskEntitySelection(("object_1",)),
            task_reference_rgb=reference,
        )


@pytest.mark.parametrize(
    "suite_name",
    ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"),
)
def test_every_shipped_task_has_a_supported_variable_entity_set(suite_name):
    suite = get_benchmark(suite_name)()
    counts = []
    for task_index in range(suite.get_num_tasks()):
        problem = robosuite_parse_problem(suite.get_task_bddl_file_path(task_index))
        counts.append(len(infer_task_entities(problem).instance_names))
    assert all(1 <= count <= 4 for count in counts)
