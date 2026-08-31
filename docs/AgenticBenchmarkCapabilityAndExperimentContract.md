# Agentic Benchmark Capability and Experiment Contract

Status: development contract v1, frozen for the first multi-seed pilots.

This document separates three independent benchmark axes. A task is not a
"level", and a richer observation profile is not assumed to be more difficult.

## 1. Benchmark axes

### Task capability

- `grounding`: bind language roles to the correct scene entities.
- `metric_spatial_reasoning`: use geometry rather than a memorized image-space
  waypoint.
- `contact_reasoning`: infer grasp, contact, obstruction, or slip from physical
  feedback.
- `causal_composition`: complete physical prerequisites and compose multiple
  manipulation skills.
- `failure_recovery`: diagnose an unsuccessful action and change strategy.
- `cross_episode_transfer`: reuse procedural experience acquired in prior
  episodes of the same Agent session.
- `negative_transfer_control`: reject experience whose surface form is similar
  but whose procedural structure is irrelevant.

### Observation profile

- `level1`: robot state plus head and wrist RGB.
- `level2`: level1 plus first-observation task-entity bbox and mask.
- `level3`: level2 plus joint/gripper velocity, commanded torque, EEF
  force/torque, and EEF 6D velocity.
- `level4`: level3 plus metric depth and camera calibration.

The primary Agentic benchmark uses level4. Profile comparisons are diagnostic
ablations on selected tasks rather than a full Cartesian product.

### Context condition

- `direct`: execute only the held-out query episode.
- `unrelated_support`: execute equally budgeted but procedurally unrelated
  support episodes before the query.
- `passive_demo`: inspect demonstrations for support skills without executing
  support episodes.
- `active_support`: execute support episodes without expert demonstrations.
- `demo_assisted_active_support`: inspect support demonstrations and then
  execute the support episodes.
- `query_demo_upper_bound`: expose a demonstration for the query itself. This
  is an upper bound and not part of the primary transfer score.
- `context_reset_control`: execute support episodes but start the query in a
  fresh Agent context.

## 2. Support-query protocol

The unit of an experience-transfer evaluation is a task family, not an
individual rollout.

- Support episodes precede exactly one final query episode.
- `primary_metric_episode_index` identifies the final query.
- The query has no fixed demonstration except in the explicitly named upper
  bound.
- The same query `init_state_id`, simulator seed, action budget, observation
  profile, model, and reasoning effort are used across context conditions.
- Support outcomes are reported independently. A failed support episode is not
  silently treated as successful experience.
- The Agent sees future task instructions only when the corresponding episode
  starts.
- Success and intermediate task predicates remain evaluator-private until
  `finish_episode`.
- Robot control is restricted to the common MCP tools
  `start_episode`, `osc_sequence`, and `finish_episode`; general analysis tools
  remain Agent-controlled.

## 3. Replication unit

LIBERO scene geometry is primarily determined by `init_state_id`. A different
simulator seed with an unchanged initial state is not, by itself, evidence of a
new spatial configuration. Every replicate therefore records a pair:

```text
(init_state_id, simulator_seed)
```

Demonstration-source initial states are excluded from query replicates when
known. All context conditions within one replicate share the exact query pair.

## 4. Resource and scoring contract

- Action budgets are stated both as Agent sequence submissions and executed
  native OSC micro-actions.
- Multi-object queries may receive a larger sequence-submission budget than
  atomic support tasks, but that query budget is identical across conditions.
- Infrastructure failure, Agent termination, budget exhaustion, and official
  checker failure are separate result categories.
- Primary metric: final-query official success rate.
- Required secondary metrics: support success, all-subgoal completion,
  sequence submissions, native micro-actions, wall time, token use, and
  recovery attempts.
- Reported transfer quantities:
  - active transfer: `SR(active_support) - SR(direct)`;
  - context specificity: `SR(active_support) - SR(unrelated_support)`;
  - active advantage: `SR(active_support) - SR(passive_demo)`;
  - demo assistance: `SR(demo_assisted_active_support) - SR(active_support)`.

## 5. Task-design acceptance criteria

Every task family must document why its target capability is causally needed.

- A task cannot claim active perception if all required information is visible
  in the initial frame.
- A task cannot claim contact reasoning if RGB alone unambiguously reveals the
  relevant contact state.
- A task cannot claim procedural transfer if absolute demonstration waypoints
  solve the query unchanged.
- A compositional query must require all declared subskills in its official
  success predicate.
- Every query must have a physically verified demonstration or scripted/human
  reference proving feasibility, even when that asset is not Agent-visible.
- Public observations and ICL bundles contain only profile-allowed fields.

## 6. Initial task families

### Drawer causal composition

- Support A: open the top drawer.
- Support B: put the black bowl in an already-open top drawer.
- Query: open the top drawer and put the bowl inside.
- Primary capabilities: contact reasoning, causal composition,
  cross-episode transfer, and negative-transfer control.

### Two-object basket composition (calibration candidate)

- Support A: pick up the alphabet soup and place it in the basket.
- Support B: pick up the tomato sauce and place it in the basket.
- Query: put both the alphabet soup and the tomato sauce in the basket.
- Primary capabilities: repeated object grounding, multi-object state tracking,
  causal composition, and cross-episode transfer.
- The support scenes and query scene differ, so absolute support waypoints are
  not a valid query solution.

### Stove interaction composition (primary extension)

- Support A: turn on the stove.
- Support B: put the moka pot on the stove.
- Query: turn on the stove and put the moka pot on it.
- Primary capabilities: articulated contact, object manipulation, causal
  composition, and cross-episode transfer.

The matched Stove family uses the exact same Scene 3 fixtures and objects in
the two supports and query. A later abstract-transfer control may instead use
the `libero_goal` Stove and Bowl tasks as supports; that condition must be
reported separately because both scene appearance and manipulated object
change.

## 7. Evidence threshold

A single successful curriculum and a single failed direct rollout establish a
pilot hypothesis only. A benchmark claim requires multiple initial-state/seed
pairs and multiple Agent models. Development runs from a dirty worktree remain
traceable validation artifacts but are not publication-formal results.

## 8. Reproducible execution

Publication-scale comparisons are encoded as explicit
`libero.agent_experiment_matrix.v1` files. Each declared run records its
context condition and a concrete query `(init_state_id, simulator_seed)` pair;
there is no implicit seed expansion at launch time.

Run a matrix sequentially with:

```bash
python scripts/run_agent_experiment_matrix.py \
  --matrix configs/agent_experiments/<matrix>.json
```

The runner refuses to overwrite incomplete runs, skips finished runs, and
rewrites lossless JSON plus CSV and Markdown summaries after every completed
rollout. Summaries can be rebuilt without running an Agent:

```bash
python scripts/summarize_agent_experiment.py \
  --matrix configs/agent_experiments/<matrix>.json \
  --batch-root agent_runs/<matrix-name>
```

For publication runs, `--launcher-root` may point to a clean detached worktree
while `--artifact-root` points to the evaluator-private replay masters. The
runtime checkout and dirty state are still recorded independently in every
run manifest; orchestration from another checkout does not hide runtime source
provenance.
