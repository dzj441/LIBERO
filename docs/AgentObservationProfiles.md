# LIBERO Agent Observation and Control Contract v1

## Scope

This branch defines the online AgentEnv contract for a coding agent acting in
LIBERO. It does not implement expert replay or ICL assets. Those assets may
reuse the same public frame schema in a later iteration.

The first integration task is `libero_object` task 0:

> pick up the alphabet soup and place it in the basket

The benchmark guarantees observation-contract integrity. Reward, object world
poses, raw segmentation IDs, BDDL internals, and checker details remain
host-private. Runtime containment of a deliberately adversarial agent is a
separate evaluator concern.

## Native LIBERO control semantics

`ControlEnv.step(action)` forwards its argument directly to robosuite. With the
default `OSC_POSE` controller, the action is seven dimensional:

1. normalized XYZ delta, three values;
2. normalized rotation-vector delta, three values;
3. gripper command, one value.

Robosuite silently clips each of the first six normalized components to
`[-1, 1]`. Its default scaling maps them to:

- translation: `[-0.05, 0.05]` metre per component;
- rotation vector: `[-0.5, 0.5]` radian per component.

The rotation vector is converted to a rotation matrix and left-multiplied onto
the current EEF orientation, so its axis is expressed in the simulation world
frame. It is not an RPY/Euler vector. The gripper uses the command sign:
negative opens, positive closes, and zero holds the current internal target.

One `env.step()` is one 20 Hz policy interval. Robosuite holds the policy goal
over several lower-level MuJoCo integration steps during that interval. A raw
action is therefore not an absolute pose command, and an oversized action is
clipped rather than rejected.

## Public high-level EEF interface

The v1 implementation exposes a physical delta command:

```text
osc_step(
  delta_position_m=[dx, dy, dz],
  delta_rotation_rotvec_rad=[rx, ry, rz],
  delta_gripper_width_m=dw,
)
```

Both translation and rotation axes are expressed in the Panda robot-base
frame. Rotation is a rotation vector and is left-applied. `dw` is the change
in total two-finger jaw opening width in metres: positive opens, negative
closes, and zero preserves the existing actuator target and grip force. A
nonzero delta is relative to the measured width at the beginning of the
command. A target outside the physical Panda width range `[0, 0.08]` metre is
rejected before physics advances.

The adapter converts base-frame errors to the world-frame convention used by
OSC_POSE, executes bounded controller substeps, closes the loop on the actual
EEF pose, installs the metric gripper target, and settles before returning the
actual next observation. It never writes MuJoCo gripper qpos. Robosuite's
persistent position-actuator target is updated and all physical motion,
contact blocking, force limiting, and holding are left to MuJoCo. This small
adapter is necessary because PandaGripper's native scalar discards magnitude
and integrates only its sign on every lower-level control substep; one raw
LIBERO action cannot express an arbitrary metric width delta.

There is no public XYZ or rotation-magnitude rejection boundary. Large finite
commands use multiple native control cycles. Internal per-cycle bounds are
4 cm translation norm and 0.35 rad rotation norm, below robosuite's native
scales so robosuite never silently clips an adapter-generated action. A finite
control-cycle budget prevents an unreachable target from hanging forever;
The result never silently clips the public arm command. It returns
`command_completed`, separate motion and gripper completion flags, a safe
`termination_reason`, and final pose errors. An infeasible request therefore
ends at the internal control-step budget with the actual downstream
observation instead of being reported as executed successfully.

The rotation vector denotes the SO(3) transform obtained by its exponential
map. As usual for rotation vectors, magnitudes greater than pi are non-unique;
the closed-loop executor follows the shortest residual orientation path. This
is representation semantics, not action clipping.

An RPY-facing adapter is straightforward but is not frozen yet. If adopted, it
must declare one exact Euler convention and convert the composed rotation once
to a matrix / rotation vector. Treating RPY components as three independent
normalized OSC components would be incorrect.

## Native OSC sequence A/B design record

The metric `osc-step` interface remains the default. A controlled A/B can
select a mutually exclusive native `osc-sequence` condition so that an
Agent can use the same per-control-cycle action semantics as the source LIBERO
demonstrations without an offline conversion to Agent-specific macro actions.

One accepted `osc-sequence` submission contains between 1 and 20 normalized
7D `OSC_POSE` micro actions. Each vector has the native LIBERO component order
`[dx, dy, dz, rx, ry, rz, gripper]`; every finite component must be within
`[-1, 1]`. The server executes the vectors sequentially, with exactly one
LIBERO policy interval per micro action, and returns one actual observation
after the submitted sequence. Intermediate simulation frames remain
evaluator-private and are retained in the continuous audit video; no
intermediate public observation is materialized.

A native-sequence run permits at most 50 accepted submissions, so its total
public control budget is bounded by 1,000 native policy intervals. One run
exposes either metric `osc-step` or native `osc-sequence`, never both, keeping
the action-interface A/B identifiable. `start` and `finish` retain their
existing meanings, and only `finish` exposes official task success.

## Episode lifecycle

The stateful interface is:

```text
start_episode           -> observation 0
osc_step | osc_sequence -> execution metadata + actual next observation
finish_episode          -> final official success boolean
```

Task reward and checker state are not returned by `start_episode` or either
action operation. `finish_episode` exposes only final success and the number
of accepted high-level actions.

The initial state is settled with zero arm and hold-gripper commands before
observation 0. Each metric `osc_step` also has a short post-action settle
window. A native `osc_sequence` performs exactly the submitted policy
intervals without an added settle action. In both cases, the returned
observation is causally downstream of the accepted action.

## Observation levels

All levels include head RGB, wrist RGB, and kinematic state. Each level is a
strict public superset of the previous level.

| Level | Public contents |
| --- | --- |
| Level 1 | Head RGB, wrist RGB, 7 arm joint positions, two gripper finger joint positions, gripper width, EEF pose |
| Level 2 | Level 1 plus initial-observation-only bbox and binary mask for `manipulated_object` and `goal_fixture` in both cameras |
| Level 3 | Level 2 plus arm joint velocity, gripper velocity, controller-commanded arm torque, EEF force, EEF torque, and EEF 6D twist |
| Level 4 | Level 3 plus head/wrist metric depth, valid-depth mask, intrinsics, and dynamic extrinsics |

### Level 1 state

- `arm_joint_position_rad_7d`
- `gripper_finger_joint_position_m_2d`
- `gripper_width_m`
- `eef_pose_robot_base_xyzw_7d`

The EEF pose is the OSC-controlled grip site expressed in robot base, with
quaternion ordering XYZW.

### Level 2 annotations

Annotations exist only on `obs_000000`. Later frames contain no annotation
field. Each camera contains only the public roles:

- `manipulated_object`
- `goal_fixture`

Each role has a boolean mask, visible pixel count, visibility flag, and bbox in
`[x_min, y_min, x_max_exclusive, y_max_exclusive]` convention. Raw instance
names and segmentation IDs never enter the public frame or files.

For pick/place tasks, roles are inferred from the parsed goal and containment
region. Ambiguous tasks fail closed and require an explicit private
`AnnotationRoles` mapping. A static audit inferred roles for 116 of the 130
shipped BDDL tasks. The 14 deliberate failures are drawer, microwave, or stove
articulation tasks whose single `target + goal` annotation semantics must be
defined separately rather than guessed.

### Level 3 dynamic proprioception

- `arm_joint_velocity_rad_s_7d`
- `gripper_finger_joint_velocity_m_s_2d`
- `gripper_width_velocity_m_s`
- `commanded_arm_joint_torque_nm_7d`
- `eef_force_sensor_n_3d`
- `eef_torque_sensor_nm_3d`
- `eef_twist_robot_base_6d`

The torque is robosuite's OSC output after actuator-limit clipping and before
assignment to MuJoCo controls; it is not a PD estimate. Force and torque are
the MuJoCo measurements at the Panda `ft_frame` sensor and retain that sensor
frame. Linear and angular EEF velocities are converted to robot base.

### Level 4 geometry

Each camera adds:

- float32 metric depth in metres;
- boolean valid-depth mask;
- 3x3 intrinsic matrix;
- `matrix_T_robot_base_from_camera_opencv_4x4`.

Images use top-left-origin OpenCV row order and camera axes
`+X right, +Y down, +Z forward`. LIBERO's default OpenGL render buffer is vertically flipped once.
It is not horizontally flipped. This convention was verified by projecting the
alphabet-soup world position through the published calibration and comparing
it with the instance-mask centroid in both cameras.

## Agent-readable artifacts

`write_public_observation()` materializes an already-projected frame as:

```text
observation.json
head/rgb.png
wrist/rgb.png
head/depth_m.npy                         # Level 4 only
head/depth_visualization.png             # Level 4 only
head/depth_valid_mask.png                # Level 4 only
wrist/...                                # Level 4 only
annotations/{head,wrist}/*_mask.png      # obs 0, Level 2+ only
annotations/{head,wrist}/annotations_overlay.png
```

Dense metric depth is kept in NPY rather than expanded into JSON. The preview
is for visual inspection; geometry calculations must use `depth_m.npy`.
Level 1 output physically contains no depth or annotation files.

## Ephemeral workspace and Unix-socket runtime

The official coding-agent runtime is:

> isolated Agent workspace + background LIBERO server + Unix socket + a
> three-operation `liberoctl` + current-only disk observation.

One invocation of `scripts/launch_agent_episode.py` creates both sides:

```text
LIBERO/agent_runs/<run_id>/                 evaluator-private
  server.log
  actions.jsonl
  private_observations/obs_*/
  continuous_video.mp4
  result.json
  run_manifest.json
  server_ready.json                         verified before Codex starts
  codex_session.jsonl                       when a matching session is found
  agent_prompt.txt
  agent_workspace_contract.json
  viewed_artifacts/                         files explicitly opened by ImageView
  viewed_artifacts_manifest.json

/tmp/libero-agent-workspace-<random>/        ephemeral Codex cwd
  TASK_PROMPT.txt
  .libero/control.sock                      live episode only
  .libero/episode.json
  bin/liberoctl
  benchmark_inputs/current_observation/
  scratch/
```

`run_id` is time/random based and contains no seed. The seed and task index are
kept only in the private manifest. The normal `HOME` and `CODEX_HOME` are
preserved, so evaluator configuration, skills, plugins, and the canonical
Codex session history remain available. A matching session log is copied into
the private run directory after Codex exits for convenient audit.

The launcher starts `codex exec` with the isolated workspace as its real
`cwd`, prepends `workspace/bin` to `PATH`, and supplies the workspace-local
socket via `LIBERO_CONTROL_SOCKET`. This is one saved, auditable Codex session,
but its CLI process exits after
the Agent's final message instead of waiting at an interactive input box. Hook
trust and Codex's inner sandbox prompts are bypassed because deployment
containment is evaluator-controlled. The public client exposes `start`,
`finish`, and exactly one action operation selected for the run. The default
metric condition is:

```bash
liberoctl start
liberoctl osc-step --position DX DY DZ --rotation RX RY RZ --gripper-delta-m DG
liberoctl finish
```

The native A/B condition replaces `osc-step` with:

```bash
liberoctl osc-sequence --actions-file PATH
```

`PATH` is read by the client and only its JSON array is sent to the server; the
server never receives or reads an Agent filesystem path. There is no `observe`
operation. `start` and the selected action operation return a small JSON
receipt with the new observation ID, the relative
`benchmark_inputs/current_observation/observation.json` path, and safe execution
metadata. RGB, masks, and NPY depth are never expanded into terminal output.

Every `liberoctl` call is synchronous. The client blocks on the Unix socket
while the simulator executes, renders, and atomically publishes the complete
next observation. Only then does the server return the receipt and let the
shell command finish; that command completion is the Agent's ready
notification. No separate sleep, file polling, or asynchronous notification is
required. The client also reads the current public `observation_id` and binds it
to every action and `finish` request. The server rejects missing or stale IDs
before advancing the simulation, preventing overlapping commands from silently
acting on a newer frame.

The current directory is replaced from a fully written staging tree after
every observation. Level 2 annotations therefore disappear physically after
observation 0, and the agent receives no automatic online history. It may use
its own shell capabilities to preserve anything it chooses. The evaluator
independently snapshots every published frame and records every LIBERO policy
interval as a private head/wrist H.264 video.

Example launch:

```bash
python scripts/launch_agent_episode.py \
  --suite libero_object \
  --task-id 0 \
  --profile level4
```

Select the native-action A/B condition with:

```bash
python scripts/launch_agent_episode.py \
  --suite libero_object \
  --task-id 0 \
  --profile level4 \
  --action-interface native_osc_sequence
```

The launcher passes `HTTPS_PROXY=http://127.0.0.1:7890` to Codex by default.
The explicit URI scheme avoids unnecessary WebSocket reconnect attempts while
using the same local proxy endpoint. The
simulation server uses EGL exclusively. It detects the installed NVIDIA kernel
driver and prepends the matching
`LIBERO/runtime/nvidia/<version>/runtime-libs-full` stack, together with its EGL
vendor manifest; it does not fall back to OSMesa.

Before launching Codex, the launcher requires both the socket and an
evaluator-private `server_ready.json`. It verifies the task, init-state index,
profile, seed, resolution, timing, action interface, and action budget against
the requested run. The private run manifest records SHA-256 fingerprints for
the exact prompt, workspace operation contract, optional ICL manifest, expected
and actual ready contracts, source status, and the combined run configuration.
These audit artifacts are not copied into the Agent workspace.

### Prompt boundary

`TASK_PROMPT.txt` contains only the official task instruction and nonstrategic
interface semantics: call `start` once, use physical robot-base-frame position
and rotation-vector deltas plus total jaw-width delta, inspect each returned
current observation, and call `finish` once. It provides no object pose,
waypoint, action magnitude recommendation, or policy hint.

### Process and resume semantics

The launcher owns both the server and Codex processes. A normal or abnormal
Codex exit immediately terminates an unfinished server and records the run as
`aborted`. `finish` records official success and lets the server shut down
cleanly. The one-shot Codex process then returns after its final response. The
Unix socket is removed in both cases.

An active embodied episode is deliberately not resumable. After Codex exits,
the launcher copies the normal `$CODEX_HOME` session, prompt, public workspace
contract, and every non-observation file explicitly opened through `ImageView`
into the private run. Per-step current-observation views are reconstructed from
`private_observations/`. The inactive random workspace is left on the system
temporary disk for the operating system to reclaim; the launcher does not
delete it. A later `codex resume` must not reconnect to or continue that
simulator episode. `--keep-workspace` selects a stable named debug cwd instead.

`scripts/run_agent_env.py` remains a developer-only stdin JSONL transport for
local diagnostics. It uses the same service and current-only serializer but is
not the official Codex launch path.

## Leakage boundary

The public projector explicitly copies every permitted nested field. It does
not copy whole internal dictionaries. Tests inject fake actor pose, planner
target, contact point, raw instance ID, reward, and checker fields into the
master frame and verify that none survive projection.

LIBERO previously ignored `use_object_obs=False` and also crashed along that
path. This branch fixes the lifecycle so the live raw observation itself no
longer contains object positions or object-to-EEF poses. Instance segmentation
is still generated host-side to create Level 2 annotations, but only the two
semantic masks and bboxes are projected.

## Current validation

- Pure contract and control tests cover all four levels, initial-only
  annotations, nested allowlist leakage, rotation conversions, normalized OSC
  bounds, and artifact materialization.
- A physical Level 4 reset on `libero_object/0` produced aligned head/wrist RGB,
  depth, calibration, and both semantic annotations.
- A combined +1 cm Z and +0.08 rad base-frame rotation reached tolerance via
  the real OSC controller; later observation contained no annotations.
- A single +20 cm base-Z command was accepted, used 23 native motion cycles,
  and achieved approximately +20.1 cm rather than being silently clipped.
- A pure +0.5 rad rotation held translation within approximately 1.4 mm.
- A physical delta-width smoke changed the persistent Panda actuator target
  from 40 mm to 31.08 mm without writing qpos. The measured width moved from
  41.08 mm to 34.73 mm over three policy cycles, and a later zero-delta action
  retained the 31.08 mm target while the physical width continued converging.
- A target outside `[0, 0.08]` metre was rejected without advancing the public
  observation ID.
- Unix-socket lifecycle smokes verified both clean `finish` shutdown and Codex
  exit -> aborted server shutdown. The clean run produced current-only public
  artifacts, private observation history, action JSONL, and a 20 Hz H.264
  head/wrist video.
- A real Level 4 Codex CLI rollout on `libero_object/0` completed successfully
  in 26 accepted high-level actions. The Agent used the public wrist RGB-D and
  calibration to back-project pixels into robot-base coordinates, recovered
  from a missed grasp, maintained a contact-blocked metric gripper target while
  carrying the object, adapted after one out-of-range width request was
  rejected, and received official `success=true` from `finish`.

ICL/replay and full task-suite annotation mappings remain separate work.
