# Agent fixed-demonstration ICL

LIBERO exposes a verified demonstration as a static, profile-projected file
bundle. The original HDF5, flattened MuJoCo states, BDDL path, source episode
key, and checker trace remain evaluator-private.

## Data flow

```text
LIBERO HDF5 init_state + native OSC actions
                  |
                  v
physical action replay + terminal stability verification
                  |
                  v
evaluator-private P4 replay master
                  |
                  v
strict Level 1--4 file projection
                  |
                  v
workspace/benchmark_inputs/expert_demo/
```

The replay restores the HDF5 initial state once. It then executes every
recorded action through `ControlEnv.step()`; recorded intermediate MuJoCo
states are never forced. The first demonstration observation is captured only
after the configured settling steps. Thereafter every transition is causal:

```text
frame_i -> source_action_i -> actual frame_i+1
```

For the initial `libero_object` Alphabet Soup asset, this produces 149 P4
frames and 148 transitions.

## Public bundle

```text
expert_demo/
├── manifest.json
├── trajectory.jsonl
├── frames/
│   ├── frame_000000/
│   │   ├── observation.json
│   │   ├── head/
│   │   ├── wrist/
│   │   └── annotations/
│   ├── frame_000001/
│   └── ...
└── overview/
    └── contact_sheets/
        ├── head_rgb.png
        ├── wrist_rgb.png
        ├── head_depth.png
        └── wrist_depth.png
```

Every frame uses the same public observation schema as an online frame. P4
contains head and wrist RGB, public kinematic state, dynamic proprioception,
metric depth, and camera calibration. Anonymous manipulated-object and
goal-fixture bbox/mask annotations appear only on `frame_000000`.

`trajectory.jsonl` publishes the recorded HDF5 action on every transition.
It is explicitly described as a native normalized per-control-cycle
`OSC_POSE` input:

- components 0--2 are normalized translation deltas, with 1.0 corresponding
  to the controller's 0.05 m output scale;
- components 3--5 are normalized rotation-vector deltas, with 1.0
  corresponding to the controller's 0.5 rad output scale;
- component 6 is the gripper drive command, where -1 opens and +1 closes.

These actions are not declared compatible with the high-level metric
`liberoctl step` interface. An Agent may inspect, transform, imitate, or ignore
them.

Only the episode-level outcome `verified successful demonstration` is public.
Per-step checker values, first-success timing, reward, object ground-truth
poses, raw segmentation IDs, and evaluator paths are absent. All public files
are ordinary non-symlink files with SHA-256 integrity metadata.

## Capture

Use the EGL wrapper so that depth and segmentation are rendered by the
driver-matched NVIDIA stack:

```bash
scripts/launch_replay_egl.sh \
  --dataset /path/to/task_demo.hdf5 \
  --episode demo_0 \
  --output-dir outputs/replay/task_demo_0_p4 \
  --p4-master-dir outputs/replay/task_demo_0_p4_master \
  --save-video \
  --camera-height 256 \
  --camera-width 256
```

The master directory is published atomically only if physical replay passes
the configured final stable-success check.

## Agent rollout

```bash
python scripts/launch_agent_episode.py \
  --suite libero_object \
  --task-id 0 \
  --init-state-id 17 \
  --profile level4 \
  --icl fixed_demo \
  --fixed-demo-master outputs/replay/task_demo_0_p4_master
```

The launcher validates the master, projects it into the new persistent
workspace, and adds only this ICL notice to the task prompt:

```text
A verified successful demonstration from a separate episode of the same task
is available at benchmark_inputs/expert_demo/. The current scene configuration
and object or goal poses may differ.
```

With `--icl none`, no bundle and no ICL notice are provided.

## Expansion TODOs

- Add `libero_goal` task ID 3,
  `open_the_top_drawer_and_put_the_bowl_inside`, as the first articulated,
  strongly ordered task. The drawer starts closed, so the Agent must open it
  before placing the bowl inside. Replay and freeze one verified P4 master,
  validate the bowl/cabinet annotation roles, and accept the integration only
  after a real P4 fixed-demo Codex rollout.
