import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from libero.libero.utils.demonstration_replay import (
    ending_true_streak,
    load_demonstration_episode,
    maximum_true_streak,
    normalize_demo_key,
    resolve_bddl_file,
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


if __name__ == "__main__":
    unittest.main()
