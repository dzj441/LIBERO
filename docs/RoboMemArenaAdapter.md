# RoboMemArena task-source adapter

## Scope

The first adapter milestone exposes RoboMemArena Task 4 through the same
current benchmark contract used by ordinary LIBERO tasks:

- P1--P4 public observation projection;
- current-only atomic observation publication;
- native normalized OSC sequence control over MCP or `liberoctl`;
- private continuous video and action/session audit logs;
- evaluator-private ordered-stage checking;
- a public `finish` result containing only final success and accepted action
  count.

This is a task-source integration, not a decision to make the benchmark a
RoboMemArena wrapper. RoboMemArena supplies a useful long-horizon reference
task and stage semantics. New benchmark-owned long-horizon task families can
reuse the validated infrastructure without copying its task set.

The intended follow-up is therefore two-track:

1. keep a small, version-fingerprinted RoboMemArena compatibility subset for
   comparison with that benchmark;
2. build benchmark-owned task families that reuse the generic private
   ordered-stage evaluator boundary while using our own BDDL, assets, prompts,
   and task variations.

This avoids making RoboMemArena's task distribution or release cadence the
definition of this benchmark while preserving a concrete external
long-horizon compatibility test.

## External-source boundary

RoboMemArena is kept as a separate checkout because it ships a modified LIBERO
package and additional simulation assets under the same Python package name.
The adapter does not vendor those files. A dedicated server process loads the
external fork first, then extends its package path with this repository's
`agent_env` implementation.

The launcher and server independently verify:

- the external Git commit;
- a clean tracked working tree;
- the selected BDDL SHA-256;
- the tall-bottom cabinet asset SHA-256;
- the upstream stage-reference SHA-256.

These values are committed in the evaluator-private run manifest and
server-ready contract. Absolute checkout paths never enter the Agent
workspace.

The current GitHub checkout has no repository-root license file. Keeping it
external also avoids silently redistributing assets before their licensing is
clarified.

## Observation integrity

RoboMemArena's fork currently requires `use_object_obs=True` internally. Its
raw observation therefore contains object poses and relative poses. That raw
mapping never crosses the server boundary. `MasterObservationCollector`
constructs a new allowlisted master frame and P1--P4 projection serializes only
the selected robot state, public proprioception, RGB, depth, calibration, and
initial anonymous task-entity annotations.

Task 4 annotations follow the BDDL `obj_of_interest` list. They do not expose
the hidden drawer object, semantic roles, private instance names, goal state,
reward, or stage progress.

## Task 4 success semantics

The public instruction is:

> Open and close all drawers in order to check. Put butter into the drawer that
> already contains an object.

The private checker advances only through the next expected stage:

1. open the top drawer;
2. close the top drawer;
3. open the middle drawer;
4. close the middle drawer;
5. open the bottom drawer;
6. close the bottom drawer;
7. open the top drawer again;
8. put butter in the top drawer.

Closing the top drawer at the end is recorded as an optional ninth stage, in
line with the current reference evaluator and task wording. Final success
requires all first eight stages in order. The ordinary BDDL final-goal result
is also saved privately for audit but does not replace the ordered checker.

## Running

The external checkout currently expected on this machine is:

```text
/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMemArena
```

Run one P4 no-ICL episode with the long-horizon 100-call budget:

```bash
PYTHONPATH=. ../miniconda3/envs/libero/bin/python scripts/launch_agent_episode.py \
  --suite robomemarena \
  --task-id 4 \
  --init-state-id 0 \
  --profile level4 \
  --max-agent-steps 100 \
  --action-interface native_osc_sequence \
  --control-transport mcp \
  --robomemarena-root ../RoboMemArena \
  --codex-model gpt-5.6-sol \
  --codex-effort high
```

RoboMemArena supports either no ICL or one replay-verified fixed demonstration.
The downloaded HDF5 is never copied directly into an Agent workspace. It must
first be physically replayed against the fingerprinted task source and captured
again through the benchmark's P4 allowlist.

Task 4 seed 100 has been replayed with all 1,020 native OSC actions. It passed
all eight required ordered stages, the optional final drawer close, and the
ordinary BDDL checker. The resulting P4 master contains 1,021 causal frames and
is stored at:

```text
outputs/replay/robomemarena_task4_seed100_p4_master_v1/
```

Generate the master reproducibly with:

```bash
python scripts/replay_robomemarena_demonstration.py \
  --dataset /path/to/place_butter_into_drawer_have_object_full_seed100_task4.hdf5 \
  --robomemarena-root ../RoboMemArena \
  --task-id 4 \
  --p4-master-dir outputs/replay/robomemarena_task4_seed100_p4_master_v1 \
  --output-dir outputs/replay/robomemarena_task4_seed100_p4_replay_v1 \
  --render-gpu-device-id 0 \
  --save-video
```

The source HDF5 lacks metric depth, calibration, and a serialized MuJoCo
initial state. Its filename seed is therefore replayed using RoboMemArena's
official NumPy-plus-environment seeding convention; P4 depth and calibration
come from the replayed simulator, not from the downloaded file. Publication is
refused unless both the ordered private evaluator and BDDL checker succeed.

An audit of the projected Agent bundle found no source path, dataset or scene
seed, Git fingerprint, private object instance name, reward, goal state,
ordered-stage progress, or checker field. The public bundle contains anonymous
initial task-entity annotations, P4 observations, and normalized OSC actions.

## P4 validation pilot

The adapter was exercised end to end with a real Codex P4, no-ICL rollout of
Task 4. The run completed without a server, MCP, Codex-session, or observation
publication failure:

- Codex voluntarily called `finish_episode` after 80 accepted OSC sequence
  submissions;
- those submissions contained 886 native OSC micro-actions, all accepted by
  the simulator;
- the private ordered evaluator reported 0/8 required stages and the ordinary
  BDDL goal was also false;
- the Agent repeatedly approached the top drawer handle but did not establish
  a pull that crossed the 10 cm stage threshold.

The failed physical result is useful validation rather than an integration
success claim: P4 depth and calibration were consumed, the robot was controlled
through the public MCP boundary, continuous video and the Codex session were
recorded, and the private sequential checker observed the whole rollout.

The pilot is stored at:

```text
agent_runs/robomemarena_task4_p4_no_icl_pilot_seed_1830315042/
```

During its post-run leakage audit, the public `.libero/episode.json` was found
to contain the evaluator-private run identifier. The pilot Agent did not read
or use that value, but the field has since been removed from both single-episode
and curriculum workspace contracts. Regression tests now assert that it is
absent. The source fingerprint, run identifier, ordered-stage progress, raw
object observations, reward, and goal state remain evaluator-private.

The complete AgentEnv and demonstration-replay test suite passed with 140
tests after this correction.

## Task 4 demonstration download

Task 4 full trajectories are downloaded separately from the current
ModelScope repository into the shared dataset disk:

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/robomemarena/RoboMemArena-Multi-Object-Occlusion/
```

Only `4_drawer_butter_dataset/full_trajectory/*.hdf5` is selected. The
download process clears all HTTP, HTTPS, and ALL proxy variables and also
disables Git's HTTP/HTTPS proxy settings. Downloading an HDF5 file is not
sufficient to expose it to an Agent: the dataset commit is frozen, the selected
trajectory is replay-verified against the matching BDDL/assets, and only its
public P1--P4 projection is published.

## Version caveat

The checked-out GitHub BDDL is frozen by commit and hash, but the upstream
README says the latest Task 4/5 randomized demonstrations are currently on
ModelScope. Results from this adapter must name the exact source fingerprint;
GitHub assets and a newer dataset release must not be presented as one
undifferentiated version.
