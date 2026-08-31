# Agentic Task Family Roadmap

Status: candidate roadmap. Inclusion in a reported benchmark requires physical
verification and repeated Agent rollouts; task availability alone is not
evidence of an Agentic capability.

## Benchmark claim under test

The intended claim is not that a general Agent is a stronger single-step
policy than a trained VLA. The benchmark asks whether an Agent can use a
standard robot-control boundary plus public observations to acquire, combine,
adapt, and recover manipulation procedures over long interactions and across
episodes. The central comparisons therefore vary task structure and context,
not only observation richness.

## Primary families

### Drawer prerequisite composition

- Support A: `libero_90` task 7, open the top drawer.
- Support B: `libero_90` task 29, put the black bowl in an already-open top
  drawer.
- Query: `libero_goal` task 3, open the top drawer and put the bowl inside.
- Capabilities: articulated contact, prerequisite discovery, long-horizon
  composition, recovery, and cross-episode transfer.
- Status: implemented; direct, query-demo, and active-curriculum pilots exist.

### Matched Stove composition

- Support A: `libero_90` task 20, turn on the Stove.
- Support B: `libero_90` task 19, put the Moka pot on the Stove.
- Query: `libero_10` task 2, turn on the Stove and put the Moka pot on it.
- All three use Kitchen Scene 3 and the same fixtures and object classes.
- Capabilities: small articulated contact plus pick-and-place composition,
  multimodal state verification, and cross-episode transfer.
- Status: all three native demonstrations are physically replay-verified and
  frozen as P4 masters. The first matched pilot completed on 2026-08-29:
  direct query failed (99 calls); active no-ICL supports were 0/1 successful
  and the query failed (25/47/32 calls); demo-assisted supports were 2/2
  successful and the query failed after a recoverable Moka slip
  (5/7/89 calls); the same-task query-demo upper bound succeeded in 27 calls.
  These are single-pair pilot observations, not success-rate estimates.

## Replication and control families

### Frying-pan Stove replication

- Supports: `libero_90` task 20 plus task 18.
- Query: `libero_90` task 21.
- A second Scene 3 realization is available as tasks 44 and 45.
- Purpose: test whether a Stove result is specific to one Moka trajectory or
  repeats with a different manipulated object.

### Bottom-drawer close composition

- Supports: `libero_90` task 24, put the bowl in the open bottom drawer; and
  task 22, close the bottom drawer.
- Query: `libero_10` task 3, put the black bowl in the bottom drawer and close
  it.
- Purpose: reverse the Drawer interaction direction and test whether the
  Agent maintains the object goal while actuating the fixture.

### Two-object basket composition

- Supports: `libero_90` task 46 and task 49.
- Query: `libero_10` task 0, put both Alphabet Soup and Tomato Sauce in the
  basket.
- Purpose: repeated grounding, state tracking, and budget allocation.
- This is a calibration family rather than the headline result because both
  supports are structurally similar pick-and-place tasks.

### Two-target mug routing

- Supports: `libero_90` task 67 and task 68.
- Query: `libero_10` task 4, place the White Mug on the left plate and the
  Yellow-and-White Mug on the right plate.
- Purpose: entity binding, target disambiguation, and avoiding subgoal
  interference.

## Harder future families

### Stack then transport

- Candidate queries: `libero_90` tasks 63 and 64.
- Capabilities: contact-sensitive stacking, verification of a relational
  state, and transport of a compound object configuration.
- Required work: identify or author atomic support tasks without leaking the
  final stack-and-place trajectory.

### Microwave containment and closure

- Articulation supports exist as `libero_90` tasks 33 and 35.
- Query: `libero_10` task 9, put the Yellow-and-White Mug in the Microwave and
  close it.
- Required work: verify the query initial state and construct a matched
  placement support if no shipped atomic task has the same scene and target.

### Recovery-specific variants

- Perturb an object after grasp, partially obstruct a receptacle, or initialize
  an articulated fixture at an intermediate state.
- Required acceptance criterion: the perturbation must be public-observation
  recoverable and must not require hidden simulator state or checker feedback.

## Context controls required for a transfer claim

- Direct query with no prior episode.
- Matched active support episodes with no demonstrations.
- Equally budgeted unrelated support episodes.
- Matched support demonstrations without active execution.
- Demo-assisted active support.
- Context reset between support and query.
- Same-task query demonstration as a clearly labeled upper bound.

The same query `(init_state_id, simulator_seed)` pairs and action budgets must
be shared across these controls. A curriculum result is not counted as
successful support acquisition unless each support outcome is reported.
