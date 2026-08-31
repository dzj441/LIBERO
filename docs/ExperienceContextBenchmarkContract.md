# Experience Context Benchmark Contract v1

Status: development contract for controlled in-context experience ablations.

Implementation status: the current v1 exporter accepts physically verified
successful `expert_episode` masters as static sources.  `agent_episode`, failed
or partial static sources, and retrieved memory are contract-level extensions;
same-session Agent experience is already exercised by the curriculum runner.
They must not be reported as implemented static conditions yet.

The benchmark must not reduce in-context learning to a single
`fixed_demo=true|false` switch.  A context condition is defined by where an
experience came from, what public information it contains, how it relates to
the query, how reliable it is, and how it is delivered.  These axes are kept
separate from the observation profile and the robot action interface.

## 1. Benchmark question

The primary question is:

> Can a general-purpose Agent select and transfer useful invariants from
> embodied experience while adapting to the current scene, rather than merely
> replaying a trajectory or being distracted by irrelevant experience?

This supports four distinct measurements:

- `context_benefit`: matched experience versus no context;
- `modality_value`: text, video, observations, actions, and their combinations;
- `compositional_transfer`: subtask experiences versus a full query
  demonstration;
- `negative_transfer_control`: irrelevant or misleading context versus no
  context.

Reasoning effort is a separate axis.  More test-time reasoning must not be
silently equated with more embodied experience.

## 2. Controlled context axes

Every context item has evaluator-private labels for the following axes.

### Source

- `expert_episode`: replayed and physically verified dataset episode;
- `agent_episode`: a prior online Agent rollout;
- `human_guidance`: evaluator-authored language or visual guidance;
- `derived_summary`: a deterministic public-data summary of another source.

### Public modality

- `text`: task/outcome summary or a versioned guidance document;
- `video`: a head/wrist RGB video and sampled RGB contact sheets;
- `observations`: the profile-projected frame trajectory;
- `actions`: native per-control-cycle OSC_POSE commands;
- any explicit combination of the above.

`multimodal` is not a separate opaque value.  It is the declared set of
modalities, such as `["video", "observations", "actions"]`.

Text items additionally declare one of three public kinds:

- `outcome_summary`: source task plus verified episode outcome only;
- `procedural_guidance`: evaluator-authored, versioned task guidance;
- `source_agent_message`: a public message produced by the source Agent.

These kinds are separate conditions.  In particular, a success label must not
be pooled with a natural-language procedure under a generic "text ICL" name.

### Relation to the held-out query

- `same_task_separate_episode`;
- `same_task_different_variant`;
- `compositional_subtask`;
- `analogous_task`;
- `irrelevant_task`;
- `counterfactual_or_misleading`.

The relation label is evaluator-private by default.  The Agent may see the
source task instruction and the experience itself, but is not told that an
item is "matched", "irrelevant", or "misleading".  Relation disclosure can be
studied later as a separate controlled variable.

### Outcome and reliability

- `verified_success`;
- `verified_failure`;
- `partial_or_interrupted`;
- `unlabelled`.

Only claims justified by source provenance may be public.  A successful replay
may be described as verified successful.  Stepwise success, first-success
timing, checker internals, simulator state, and hidden predicates remain
private.

### Delivery

- `static_bundle`: available before the query in the Agent workspace;
- `same_session_episode`: acquired by acting in an earlier episode of the same
  Codex session;
- `retrieved_memory`: reserved for a later retrieval benchmark.

Static context and active same-session experience must not be conflated in one
condition.

### Budget

Every condition records item count, frame count, action count, bytes, video
duration, and any text-token budget.  Comparisons that claim modality effects
must either match budgets or report the mismatch explicitly.

## 3. Public workspace contract

New context bundles are published at:

```text
benchmark_inputs/experience_context/
├── manifest.json
└── experiences/
    ├── experience_000/
    │   ├── manifest.json
    │   ├── text/
    │   │   └── guidance.md
    │   ├── video/
    │   │   ├── head_wrist_rgb.mp4
    │   │   └── contact_sheets/
    │   ├── trajectory/
    │   │   └── actions.jsonl
    │   └── frames/
    └── experience_001/
```

Only requested modalities are physically present.  Omitting a modality means
its files and manifest fields do not exist, rather than merely being hidden by
a prompt.  Observation frames are projected through the same level1-level4
allowlist used for online observations.  Level 2 annotations remain
initial-observation-only within each source episode.

The `video` condition is deliberately RGB-only even under level4.  Metric
depth and calibration are observation-trajectory content and appear only when
`observations` is selected; this prevents a visual-summary ablation from
silently receiving a second geometric modality.

The public root manifest contains:

- schema version and observation profile;
- a stable anonymous experience ID;
- source task instruction;
- publicly justified outcome;
- available modality names and relative manifest path;
- aggregate item count.

It does not contain evaluator relation labels, target task IDs, target seeds,
source filesystem paths, dataset keys, BDDL paths, hidden object poses, MuJoCo
state, raw segmentation IDs, or checker details.

Per-file hashes and the visibility audit contract are evaluator-private receipt
data.  They are deliberately excluded from the public manifest: exposing a
thousand-line hash inventory adds presentation load without helping the Agent
understand or use an experience.

## 4. Legacy compatibility

The existing `--icl fixed_demo --fixed-demo-master ...` interface remains a
legacy shorthand for one context item with:

```text
source = expert_episode
relation = same_task_separate_episode
outcome = verified_success
modalities = text + video + observations + actions
```

Legacy runs may continue to expose `benchmark_inputs/expert_demo/` for exact
reproduction.  New context-axis experiments use
`benchmark_inputs/experience_context/` and record the expanded private spec.
Results from the two schemas must not be pooled without naming the difference.

## 5. Prompt contract

When a static bundle is present, the task prompt adds only:

```text
One or more embodied experiences are available at
benchmark_inputs/experience_context/. Public item manifests describe each
source task, outcome, and available modality.
```

It does not prescribe how to use the items and does not reveal their private
relation labels.  Public item manifests are the authoritative description of
source task, outcome, and available modalities.

With no context, neither the directory nor the sentence is present.

## 6. Initial Drawer matrix

Use an identical held-out Drawer query state, model, effort, P4 profile, action
budget, and CLI version across conditions:

1. `none`: no static or active context;
2. `matched_full`: full successful query-task experience;
3. `matched_text`: source task and verified outcome only;
4. `matched_video`: source task/outcome text plus RGB video only;
5. `matched_procedural_text`: versioned language instructions without visual
   or numeric trajectories;
6. `matched_observations`: profile-projected observations without actions;
7. `matched_actions`: task/outcome text plus OSC actions without observations;
8. `compositional_full`: open-drawer and put-bowl subtask experiences;
9. `irrelevant_full`: equally formatted unrelated successful experience;
10. `active_curriculum`: execute the two subtasks in the same session, with no
   static query demonstration.

The first pilot may use one query state to validate behavior and instrumentation.
Publication claims require paired repetitions over multiple initial states and
multiple Agent models.

## 7. Required audit outputs

Each run must report, without attempting to expose hidden chain of thought:

- which context files the Agent actually read or viewed;
- whether current-scene observations were inspected before the first action;
- exact and near-exact action overlap with each source trajectory;
- sequence-level action similarity and prefix length;
- the fraction covered by source-aligned contiguous action runs, so a shared
  vocabulary of common saturated OSC primitives is not mistaken for copying;
- Agent-visible messages that explicitly reference source experience;
- success, action usage, recovery attempts, wall time, and token usage.

High action similarity is not automatically leakage or failure.  It becomes
evidence of shortcutting when paired with failure to adapt to changed public
geometry, or when an irrelevant item controls behavior.  Conversely, context
benefit is strongest when the Agent reads the source, adapts its actions to the
current observation, and improves success under a held-out configuration.

## 8. Non-leakage invariant

UniVTAC/LIBERO benchmark code guarantees observation-contract integrity, not
adversarial containment.  The bundle exporter must prove that only public
profile fields and explicitly selected context modalities are supplied.  How
an evaluator configures shell, network, plugins, or sandboxing is recorded as
a runtime condition and is outside this data contract.
