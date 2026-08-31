# Matched Stove Composition Pilot

Status: development pilot, 2026-08-29. This document records one paired
initial-state/seed configuration and must not be reported as a success-rate
estimate.

## Task family

- Support A: `libero_90` task 20, `turn on the stove`.
- Support B: `libero_90` task 19, `put the moka pot on the stove`.
- Query: `libero_10` task 2, `turn on the stove and put the moka pot on it`.
- Observation condition: Level 4.
- Action boundary: MCP `osc_sequence`, with 1--20 normalized native
  `OSC_POSE` micro-actions per Agent call.
- Runtime source: clean checkout at
  `8481902d25843211e25acb0f4bfbe6c80e414ede`.
- Query pair: `init_state_id=22`, `simulator_seed=1830315042`.
- Query budget: 100 accepted sequence calls; support budget: 50 calls each.

## Physically verified fixed demonstrations

The two downloaded LIBERO90 `demo_0` trajectories were replayed through the
real environment rather than accepted from HDF5 metadata alone.

| Task | Source actions | Public P4 frames | First success | Stable final success |
|---|---:|---:|---:|---:|
| Turn on Stove | 90 | 91 | 76 | 14 frames |
| Put Moka on Stove | 135 | 136 | 123 | 12 frames |

The masters are:

- `outputs/replay/stove_turn_on_demo_0_p4_master_v1/`
- `outputs/replay/stove_put_moka_demo_0_p4_master_v1/`
- `outputs/replay/stove_turn_on_put_moka_demo_0_p4_master_v1/` for the query
  upper bound.

Each bundle exposes the declared P4 observations and native OSC actions. It
does not expose MuJoCo state, object ground-truth poses, raw segmentation IDs,
stepwise checker results, first-success timing, or source host paths.

## Official outcomes

| Condition | Support outcomes | Query outcome | Query calls | Query micro-actions |
|---|---:|---:|---:|---:|
| Direct, no ICL | -- | false | 99 | 650 |
| Active supports, no ICL | 1/2 | false | 32 | 210 |
| Demo-assisted active supports | 2/2 | false | 89 | 1155 |
| Same-task query demo upper bound | -- | true | 27 | 299 |

Machine-readable results and videos are under
`agent_runs/stove_context_pilot_p4_v1/`.

## Behavior observations

### Direct query, no ICL

The Agent localized and visibly turned the Stove control, then attempted the
Moka transfer. Repeated vertical and handle grasps were unstable. It
recognized this and tried a tabletop-push fallback, but the Moka did not reach
the burner. The official query result was false.

### Active supports, no ICL

Support A made real knob contact and rotated it by about 0.9 rad, but the
Agent's finish call returned false. The Agent explicitly carried the failed
direction hypothesis into later context. Support B established a real Moka
grasp and completed the placement in 47 calls.

In the query, the Agent explicitly referenced both preceding episodes. It
moved the Moka to the burner in about 27 calls, substantially faster than in
Support B, and attempted the corrected knob direction. It then finished after
32 calls, but the official query result was false. The visible near-completion
is evidence of behavioral reuse, not evidence of task success.

### Demo-assisted active supports

Support A submitted the 90 expert micro-actions in five chunks and succeeded.
Support B adapted rather than copying blindly: its first 20 micro-actions
added a constant normalized XYZ correction of `[0.08, -0.09, 0.035]`; the
remaining 115 actions matched the source. The Agent described the live Moka as
about 7 mm left and 29 mm nearer than in the demonstration. Support B succeeded
in seven calls.

The query had no own demonstration. The Agent converted the two successful
support experiences into current-scene Cartesian servo segments. It turned
the knob correctly and initially grasped and lifted the Moka, but the object
slipped during lateral transfer. It detected the slip, rejected several empty
grasps using gripper-width evidence, inferred an approximately 10 cm
flange-to-fingertip offset, tried orthogonal side grasps, and attempted a
tabletop push. Recovery did not succeed; the official result was false after
89 calls.

### Same-task query demonstration upper bound

The Agent estimated live-to-demo shifts of about 6 mm for the Stove and 11 mm
for the Moka, then used short closed-loop segments rather than blindly
submitting the full trajectory. It turned the knob, established a stable
grasp, kept a safe vertical offset during transfer, centered the Moka on the
burner, released it, and received official success in 27 calls.

## Interpretation

This pilot establishes feasibility and several observable Agent behaviors:

- successful native-action ICL projection and consumption;
- pose adaptation to a separate episode;
- cross-episode reuse of both successful and failed experience;
- active failure detection and recovery attempts;
- a successful same-task ICL upper bound.

It does not yet establish that active curriculum improves success rate. That
claim requires repeated paired query states, unrelated-context controls, and
more than one Agent model.
