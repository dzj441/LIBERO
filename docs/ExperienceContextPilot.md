# Experience Context v1 Drawer Pilot

Status: development pilot run on 2026-08-30--31.  This document records mechanism
validation and single-query behavior, not a publication-level success-rate
claim.  The implementation was developed from commit `8481902`; the runs retain
their exact source fingerprint and are development evidence rather than formal
benchmark results.

## Experimental question

The pilot asks whether a general-purpose Agent can select and transfer useful
information from embodied experience, rather than treating any additional
trajectory as a ground-truth shortcut.

The held-out query is identical across conditions:

- task: `libero_goal` task 3, `open the top drawer and put the bowl inside`;
- initial-state ID: 22;
- evaluator seed: 1830315042;
- observation profile: level4;
- controller: MCP `native_osc_sequence`;
- model: `gpt-5.6-sol`, effort `high`, Codex CLI 0.150.1;
- budget: at most 100 accepted Agent action submissions.

The exact no-context control already existed at the same source commit, model,
effort, CLI, initial state, seed, profile, controller, and budget.  It is used
as a historical paired control; it was not rerun in this development pilot.

## Initial results

| Condition | Context | Official | Agent calls | OSC micro-actions | Wall time | Reported total tokens | Exact-copy coverage in runs >=4 | Longest exact run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | none | fail | 100 | not re-audited | 31m 09s | 14.74M | n/a | n/a |
| legacy query demo | same-task full fixed demo | success | 15 | 183 | 5m 50s | 2.20M | 76.5% | 60 |
| matched full v1 | same-task text + RGB video + P4 observations + OSC | success | 55 | 437 | 15m 08s | 7.70M | 54.2% | 125 |
| compositional full v1 | open-drawer experience + put-bowl experience | fail | 100 | 875 | 27m 49s | 17.33M | 13.7% open / 0% bowl | 80 open / 0 bowl |
| irrelevant full v1 | alphabet-soup pick-and-place experience | fail | 89 | 950 | 26m 11s | 13.57M | 0% | 3 |
| matched actions v1 | same-task outcome text + OSC actions | fail | 100 | 1,010 | 43m 05s | 17.18M | 13.9% | 140 |
| matched observations v1 | same-task outcome text + P4 observations | infrastructure abort | n/a | n/a | n/a | n/a | n/a | n/a |

`Reported total tokens` is the cumulative Codex session counter and includes
cached input repeatedly processed across Agent turns.  It is useful for
within-run accounting but is not a count of unique prompt tokens.

## What the Agent actually did

### Matched full

The Agent inspected the current initial observation before acting and accessed
the source action trajectory plus selected source observation JSON and RGB
frames.  It did not copy the source from action zero: the shared exact initial
prefix was zero.  It subsequently reused substantial source-aligned blocks,
with 54.2% of its 437 micro-actions covered by exact runs of at least four and
a longest exact run of 125.

This was still closed-loop behavior.  The Agent publicly reported that its
first bowl grasp was off-center and generated excessive force, released it,
performed a corrected grasp, then carried and released the bowl.  The official
checker returned success.

The source and query were separate episodes but geometrically close.  Using
only fields public in the initial P4 observations, their EEF positions differed
by 3.10 cm.  Head-camera task-entity bbox centers differed by 2.0--3.5 pixels;
wrist-camera centers differed by 6.9--11.8 pixels.  This is enough to validate
adaptation instrumentation, but not a strong pose-generalization test.

### Compositional full

The Agent read both subtask manifests and sampled both source episodes.  It
copied the first 80 micro-actions of the open-drawer source exactly, then
noticed that the current cabinet was offset and stopped the replay after the
gripper passed roughly 3 cm above the handle.  It used current depth and force
feedback to recover and ultimately reported that the top drawer was open.

It did not copy any contiguous action from the put-bowl source.  The remaining
budget was spent on bowl grasp/orientation recovery, and the run exhausted all
100 action submissions before placing the bowl.  The official result was
failure.  This is evidence of subskill use and adaptation, but not successful
composition in this sample.

### Irrelevant full

The Agent recognized that the alphabet-soup experience was a different task.
It never read its action trajectory and had no exact source-aligned action run
of length four.  It instead used current P4 depth/calibration to locate the
drawer and bowl.  It made meaningful progress: it contacted the handle and
grasped/lifted the bowl, but later determined that the drawer had not remained
open and finished with official failure after 89 accepted calls.

This negative control is useful because an equally formatted extra experience
did not automatically induce source-action copying or success.

### Matched actions only

The Agent inspected the current initial observation and the complete 170-step
source OSC trajectory.  It copied the first 140 source micro-actions exactly,
which opened the drawer but missed the bowl because the query bowl was offset
by about 10 cm.  It then explicitly stopped replaying, estimated the live bowl
position from P4 depth and calibration, and generated 870 additional adaptive
or recovery actions.

The Agent repeatedly acquired a rim grasp and carried the bowl toward the open
drawer, but the bowl slipped during the long lateral motion.  It used the full
100-call budget, released the bowl near the drawer, and called finish; the bowl
landed outside and the official checker returned failure.  This condition
therefore supplied a strong motion prior but not enough state correspondence
for reliable transfer in this sample.

The exact-copy audit reports a 140-action shared prefix, equal to 82.4% of the
source episode but only 13.9% of the Agent's 1,010 executed micro-actions.  The
remaining behavior was not an unbroken expert replay.

### Matched observations only

No policy result is available yet.  Two launch attempts ended with the Codex
service error `server_overloaded` and are classified as infrastructure errors,
not benchmark failures.  The first attempt completed 37 action submissions
before interruption; it reconstructed the observed EEF trajectory, computed
rotation-vector corrections, and was still working on drawer-handle contact.
The retry was interrupted after start and before its first robot action.  This
condition must be rerun when model capacity is stable.

## Instrumentation delivered

- `experience_context` supports multiple static verified source episodes.
- Text, RGB video, P1--P4 observation trajectories, and native OSC actions are
  independently materialized at the file level.
- Relation labels, source paths, seeds, BDDL paths, MuJoCo state, raw IDs, and
  checker details remain evaluator-private.
- The optimized public root manifest is 34 lines for the real 171-frame
  matched master; the earlier 7,812-line hash inventory has moved to the
  evaluator-private projection receipt.
- Projection receipts record context budgets: files, bytes, observation
  frames, OSC actions, RGB video frames/duration, and text bytes.
- Post-hoc audit records observed context-file access, current-observation
  access before action, initial public geometry alignment, Agent messages, and
  action-copy metrics without using hidden chain of thought.
- The Viewer reports the private experimental Context ID alongside profile and
  ICL mode.

## Pilot limitations

- This is one held-out initial state.  It cannot support success-rate claims.
- Matched source/query geometry was close, so the matched result does not yet
  establish robust pose transfer.
- Full, compositional, and irrelevant bundles are not byte-, frame-, or action-
  budget matched.  The receipt now exposes these differences; subsequent
  modality claims must control or report them.
- The first three live runs were launched immediately before the public
  manifest optimization.  They saw the same permitted data but also a large
  hash inventory and negative visibility declarations.  Future formal runs
  use the reduced manifest.
- The completed actions-only retry used the optimized manifest and explicit
  `native_osc_sequence_compatible=true` action contract.  Earlier interrupted
  attempts are retained only as infrastructure evidence.
- Observations-only has no completed policy result because two attempts were
  interrupted by model-capacity errors.

## Recommended next experiment

First repeat a compact paired matrix on several query initial states selected
to have materially different public initial geometry:

1. no context;
2. matched outcome text only;
3. matched RGB video only;
4. matched P4 observations without actions;
5. matched actions without observations;
6. matched full;
7. compositional full;
8. irrelevant full.

Report success, action budget, context-access evidence, initial public geometry
distance, continuous-copy coverage, and context byte/frame/action budget for
every run.  The central analysis should separate three outcomes: no use,
shortcut replay, and observation-conditioned transfer.

## Artifacts

- runs: `agent_runs/drawer_experience_context_pilot_v1/`;
- per-run audit: `experience_context_audit.json`;
- contract: `docs/ExperienceContextBenchmarkContract.md`;
- matrix: `configs/agent_experiments/drawer_experience_context_p4_pilot.json`;
- legacy fixed-demo audit: `temp/drawer_query_demo_r0_context_audit.json`.
