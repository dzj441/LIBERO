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

These per-control-cycle action vectors are not direct inputs to the high-level
metric `liberoctl osc-step` interface. The latter accepts a Cartesian target
delta and realizes it through the same LIBERO OSC_POSE controller. An Agent may
inspect, transform, imitate, or ignore the source actions.

In the mutually exclusive native-action A/B condition, each source action
vector is directly compatible with one element of an `osc-sequence` JSON
array. A submission may contain at most 20 such vectors; it produces one
Agent-visible observation only after the complete sequence has executed.

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

## Validated assets

The P4 replay and fixed-demo projection pipeline has been exercised on both a
direct pick-and-place task and a sequential articulated task:

- `libero_object`, task 0: `pick up the alphabet soup and place it in the basket`;
- `libero_goal`, task 3: `open the top drawer and put the bowl inside`.

For the drawer-and-bowl task, `demo_0` replays to terminal success and remains
successful for 12 consecutive control steps. Its evaluator-private P4 master
contains 171 causal observation frames and 170 native OSC action transitions.
The initial public annotations identify the bowl as `manipulated_object` and
the complete cabinet as `goal_fixture`; private LIBERO instance names are not
published in the projected bundle.

## Contact-sensitive waypoint diagnostic

Recorded EEF states must not be treated as interchangeable with controller
commands. This was tested on the drawer-and-bowl `demo_0` using its exact
initial MuJoCo state:

- native OSC action replay opens the top drawer from `0.0` to approximately
  `-0.148 m` and completes the task;
- tracking all 170 recorded EEF poses through `BaseFrameOSCExecutor` leaves the
  drawer at approximately `+0.0016 m`, even though most EEF targets are reached
  within 1--2 mm;
- removing post-action settling, or tracking every fifth EEF pose, does not
  recover the drawer interaction;
- from the demonstrated handle pose, a pure `+0.15 m` pull does not move the
  drawer, while a force-biased `[-0.01, +0.15, -0.03] m` target opens it to
  approximately `-0.148 m` through the same high-level executor.

During the native pull, the controller repeatedly requests several centimetres
of negative X/Z displacement while the measured EEF barely moves on those
axes. That unobserved pose error supplies the contact force that seats the
finger behind the handle. The measured state trajectory records the resulting
motion, but not this controller intent. The source OSC actions therefore remain
an essential part of the demonstration contract for contact-rich tasks.

Evaluator-private reports and videos for this diagnostic are written under
`outputs/diagnostics/drawer_bowl_demo0_eef_waypoints_*`.
