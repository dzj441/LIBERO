# RoboMemArena task-source adapter

## Scope

The adapter exposes RoboMemArena Tasks 1--26 through the same benchmark
contract as ordinary LIBERO tasks:

- P1--P4 public observation projection;
- current-only atomic observation publication;
- native normalized OSC sequence control over MCP or `liberoctl`;
- private continuous video and action/session audit logs;
- evaluator-private ordered-stage checking;
- final success based on the ordered task semantics rather than only the BDDL
  terminal state.

This is a compatibility task source, not a decision to redefine the benchmark
as a RoboMemArena wrapper. It provides four useful long-horizon families:
multi-object sequence, occlusion/search, counting/pouring, and transferring.
Benchmark-owned long-horizon tasks can reuse the same observation, control,
and private-evaluator boundaries.

## Self-contained compatibility core

RoboMemArena ships a modified LIBERO package under the same Python package
name. The adapter freezes only the files that differ from this checkout,
together with the 26 BDDL files and official stage predicates, under:

```text
libero/libero/agent_env/robomemarena_vendor/
```

The frozen source commit is:

```text
OpenHelix-Team/RoboMemArena
cc156e519990ae43cf3b64281a548724f428fbbd
```

At server startup, `scripts/robomemarena_bootstrap.py` creates a merged LIBERO
package on the system temporary disk. Unchanged files are symlinked from this
repository and the frozen overrides are copied on top. This gives RoboMemArena
its expected physics, assets, cameras, and task registrations without changing
ordinary LIBERO behavior and without requiring a sibling `../RoboMemArena`
checkout.

An external clean checkout can still be supplied with
`--robomemarena-root` as an explicit development comparison. The default is
the in-repository compatibility subset. The run manifest fingerprints the
source kind, upstream commit, selected BDDL, representative cabinet asset, and
stage-reference source. None of those evaluator-private values enter the Agent
workspace.

The upstream repository did not contain a repository-root license file at the
frozen commit. Redistribution terms for these compatibility inputs must be
resolved before a public release; their provenance is intentionally explicit.

## Observation integrity

The fork requires `use_object_obs=True` internally. Its raw observation can
therefore contain object poses and relative poses. That raw mapping never
crosses the server boundary. `MasterObservationCollector` builds a new
allowlisted frame and P1--P4 projection publishes only robot state, public
proprioception, RGB, depth, calibration, and initial anonymous task-entity
annotations.

Every BDDL `obj_of_interest` entry is exposed as an anonymous task entity on
the initial annotated frame. The bundle does not assign manipulated-object or
goal-fixture roles and does not expose private instance names, hidden objects,
reward, goal state, or stage progress.

## Ordered success semantics

The evaluator uses task-specific ordered stage definitions adapted from
RoboMemArena. It advances only through the next expected stage. Drawer tasks
retain their documented optional final-close stage; required success excludes
that optional stage. Counting tasks use the shared physical pour-event counter
and reject a detected third pour. For the selected V1 tasks, success also
revalidates the requested object arrangement in the current simulator state;
passing through a target region earlier in the episode is not enough. Task 4
derives its occupied target drawer from the reset state instead of assuming the
top drawer. `finish` also records the ordinary BDDL final goal privately for
comparison, but ordered-history plus terminal-state success remains
authoritative because BDDL alone cannot represent order or event counts.

The contracts are split across three layers:

- `robomemarena_vendor/bddl/*.bddl` defines scene objects, initialization, and
  ordinary terminal predicates;
- `robomemarena_vendor/stage/reference_stage.py` defines private ordered,
  counting, and terminal-state predicates;
- `robomemarena.py` runs those predicates on every simulator control cycle and
  reports their result when the Agent calls `finish`.

Three upstream ambiguities are resolved explicitly by the adapter:

- Task 4 uses `init_state_id=0/1/2` for an object initially hidden in the
  top/middle/bottom drawer. RoboMemArena's HDF5 trajectories contain all three
  variants, but its released BDDL contains only the top-drawer initialization
  and its HDF5 files do not contain MuJoCo state. Demonstration replay recovers
  the omitted variant from the recorded trajectory instruction and selects the
  matching frozen BDDL.
- Task 7 accepts the tomato-sauce bottle inside either the left or right
  bowl-drainer region, matching the disjunction in its BDDL goal.
- Task 14 interprets “another drawer” literally: after cookies are placed in
  the top drawer, chocolate may finish in either the middle or bottom drawer.
  Closing that second drawer remains optional.

Container doors do not have to be closed unless the public instruction says
so. Object order or identity is not constrained beyond the public instruction
and ordered-stage contract; positions within a valid target region are not
slot-assigned.

## Running

Run Task 4 at P4 with the long-horizon 100-submission budget:

```bash
PYTHONPATH=. ../miniconda3/envs/libero/bin/python scripts/launch_agent_episode.py \
  --suite robomemarena \
  --task-id 4 \
  --init-state-id 0 \
  --profile level4 \
  --max-agent-steps 100 \
  --action-interface native_osc_sequence \
  --control-transport mcp \
  --codex-model gpt-5.6-sol \
  --codex-effort high
```

No `--robomemarena-root` argument is required.

RoboMemArena supports either no ICL or one replay-verified fixed demo. Raw
HDF5 is never copied into an Agent workspace. It is first physically replayed
against the frozen task source, captured through the P4 allowlist, and
published only if its private ordered checker succeeds.

```bash
python scripts/replay_robomemarena_demonstration.py \
  --dataset /path/to/full_seed100_task4.hdf5 \
  --task-id 4 \
  --p4-master-dir outputs/replay/robomemarena_task4_seed100_p4_master_v1 \
  --output-dir outputs/replay/robomemarena_task4_seed100_p4_replay_v1 \
  --render-gpu-device-id 0 \
  --save-video
```

## Dataset audit and presentation catalog

The shared dataset root is:

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/robomemarena/
```

Three full trajectories were selected for each of the 26 tasks. All 78 pass
the structural audit: one aligned `data/demo_0`, finite 7D OSC actions, two
256-by-256 RGB streams, EEF state, gripper state, and joint state. Selected
trajectory lengths range from 471 to 1,878 control cycles.

Task 10 contains an upstream data quirk: its motion components are normalized,
but the close-gripper component reaches `+2`. Any future public action bundle
preserves the binary close meaning by clipping only that scalar to the
benchmark's `[-1, 1]` control contract; the replay receipt records the raw
range and whether this normalization occurred.

The reproducible catalog command is:

```bash
python scripts/catalog_robomemarena_dataset.py \
  --data-root /inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/robomemarena \
  --output-dir temp/robomemarena \
  --trajectories-per-task 3
```

It produces `dataset_audit.json`, `task_catalog.json`, `README.md`, and one
side-by-side head/wrist MP4 per task. The 26 videos are named with their public
task instructions. Both HDF5 RGB streams are already stored in top-left image
coordinates, so the video exporter preserves their rows without applying the
live-robosuite OpenGL-to-OpenCV flip a second time.

## Validation

`scripts/smoke_robomemarena_tasks.py` exercises one representative from each
family: Task 1, Task 4, Task 10, and Task 25. Each smoke performs a real EGL
reset, produces a P4 initial observation, executes one native OSC micro-action,
produces the next observation, and evaluates private stage state. The combined
report is:

```text
temp/robomemarena/environment_smoke.json
```

All four representatives pass the environment/interface smoke using only the
vendored compatibility source. `finish_success=false` is expected because a
single neutral micro-action is not intended to complete a long-horizon task.

Task 4 also has a previously replay-verified P4 master at:

```text
outputs/replay/robomemarena_task4_seed100_p4_master_v1/
```

Its 1,020 native actions passed all eight required stages, the optional final
drawer close, and the ordinary BDDL checker.
