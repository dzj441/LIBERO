import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from libero.libero.utils.demonstration_replay import (
    ending_true_streak,
    load_demonstration_episode,
    maximum_true_streak,
    normalize_demo_key,
    resolve_bddl_file,
    run_action_replay,
)


class DemonstrationReplaySchemaTest(unittest.TestCase):
    def test_normalize_demo_key(self):
        self.assertEqual(normalize_demo_key(3), "demo_3")
        self.assertEqual(normalize_demo_key("3"), "demo_3")
        self.assertEqual(normalize_demo_key("demo_3"), "demo_3")
        with self.assertRaises(ValueError):
            normalize_demo_key("episode_3")

    def test_success_streaks(self):
        trace = [False, True, True, False, True, True, True]
        self.assertEqual(ending_true_streak(trace), 3)
        self.assertEqual(maximum_true_streak(trace), 3)
        self.assertEqual(ending_true_streak([True, False]), 0)

    def test_resolve_relocated_bddl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bddl_files"
            target = root / "libero_object" / "task.bddl"
            target.parent.mkdir(parents=True)
            target.write_text("(define (problem task))\n", encoding="utf-8")

            resolved = resolve_bddl_file(
                "/old/machine/libero/bddl_files/libero_object/task.bddl",
                bddl_root=root,
            )
            self.assertEqual(resolved, target.resolve())

    def test_load_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bddl = root / "task.bddl"
            bddl.write_text("(define (problem task))\n", encoding="utf-8")
            dataset = root / "demo.hdf5"
            with h5py.File(dataset, "w") as handle:
                data = handle.create_group("data")
                data.attrs["bddl_file_name"] = "/stale/task.bddl"
                data.attrs["env_name"] = "TestEnv"
                data.attrs["problem_info"] = json.dumps(
                    {
                        "problem_name": "test_problem",
                        "language_instruction": "pick and place the object",
                    }
                )
                data.attrs["env_args"] = json.dumps(
                    {
                        "env_kwargs": {
                            "robots": ["Panda"],
                            "controller_configs": {"type": "OSC_POSE"},
                            "control_freq": 20,
                        }
                    }
                )
                demo = data.create_group("demo_0")
                demo.create_dataset("actions", data=np.zeros((4, 7)))
                demo.attrs["init_state"] = np.arange(9, dtype=np.float64)

            episode = load_demonstration_episode(dataset, 0, bddl_file=bddl)
            self.assertEqual(episode.demo_key, "demo_0")
            self.assertEqual(episode.actions.shape, (4, 7))
            self.assertEqual(episode.init_state.shape, (9,))
            self.assertEqual(episode.task_instruction, "pick and place the object")
            self.assertEqual(episode.robots, ("Panda",))
            self.assertEqual(episode.bddl_file, bddl.resolve())
            self.assertEqual(episode.init_state_source, "episode_attribute")


def test_p4_callback_is_initial_after_settle_then_causal_post_action(monkeypatch):
    from libero.libero.envs import env_wrapper

    instances = []

    class FakeControlEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.steps = 0
            self.closed = False
            instances.append(self)

        def reset(self):
            return {"counter": -1}

        def set_init_state(self, _state):
            return {"counter": 0}

        def step(self, _action):
            self.steps += 1
            return {"counter": self.steps}, 0.0, False, {}

        def check_success(self):
            return True

        def close(self):
            self.closed = True

    monkeypatch.setattr(env_wrapper, "ControlEnv", FakeControlEnv)
    episode = SimpleNamespace(
        bddl_file=Path("task.bddl"),
        robots=("Panda",),
        controller="OSC_POSE",
        control_freq=20,
        actions=np.zeros((2, 7), dtype=np.float64),
        init_state=np.zeros(3),
        dataset_path=Path("demo.hdf5"),
        demo_key="demo_0",
        task_instruction="task",
        problem_name="problem",
        env_name="environment",
        init_state_source="episode_attribute",
    )
    captured = []

    def callback(_env, observation, frame_index, source_action_index):
        captured.append((frame_index, source_action_index, observation["counter"]))

    report = run_action_replay(
        episode,
        settle_steps=2,
        stable_success_steps=1,
        observation_callback=callback,
    )

    assert report["verified_success"] is True
    assert captured == [(0, None, 2), (1, 0, 3), (2, 1, 4)]
    assert instances[0].kwargs["camera_depths"] is True
    assert instances[0].kwargs["camera_segmentations"] == "instance"
    assert instances[0].kwargs["use_object_obs"] is False
    assert instances[0].closed is True


if __name__ == "__main__":
    unittest.main()
