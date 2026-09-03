# Arrange Table visual-goal task

The public instruction is `Arrange Table`. A persistent RGB task reference
shows the desired object arrangement.

## Scene contract

- Receptacles: a left plate, a right plate, and a basket.
- Manipulated objects: a porcelain mug, a yellow-and-white mug, and butter.
- Initially, neither mug is on its assigned plate and the butter is not in the
  basket.
- The task succeeds only when all three native LIBERO predicates hold:
  - `On(porcelain_mug_1, plate_1)`
  - `On(white_yellow_mug_1, plate_2)`
  - `In(butter_1, basket_1_contain_region)`

The checker is the normal LIBERO BDDL checker, so `On` requires physical
support/contact and `In` requires contact plus geometric containment. It does
not use image similarity.

## Rebuild generated assets

With the LIBERO EGL runtime configured, run:

```bash
PYTHONPATH=. python scripts/generate_arrange_table_assets.py
```

The generator creates 50 initial states, rejects any state satisfying even one
goal predicate, and renders `goal_rgb.png` only after a separately initialized
goal scene satisfies all three checker predicates.

## Human OSC teleoperation

On a headless server, use the EGL browser teleoperator:

```bash
scripts/launch_manual_osc_teleop_egl.sh \
  --host 0.0.0.0 \
  --port 8766 \
  --output-root v1temp/arrange_table_teleop
```

Open port 8766 in a browser. The page shows the desired arrangement together
with the live head and wrist cameras. It sends normalized native OSC control
cycles through the same `AgentEpisodeService` used by benchmark rollouts and
saves `actions.jsonl`, private observations, the continuous video, and the
official final checker result beneath a unique run directory. No X server or
`DISPLAY` is required. Unlike an Agent rollout, manual collection has no total
action-submission budget: it continues until the operator presses Finish or
stops the process. Each individual request remains capped at 20 controller
cycles so the browser receives timely visual feedback.

The existing LIBERO collector remains usable from a graphical X11 session:

```bash
PYTHONPATH=. python scripts/collect_demonstration.py \
  --bddl-file libero/libero/bddl_files/libero_arrange_table/arrange_table.bddl \
  --controller OSC_POSE \
  --device keyboard \
  --num-demonstration 1 \
  --directory demonstration_data/arrange_table
```

Keyboard controls are `w/a/s/d` for horizontal translation, `r/f` for vertical
translation, `z/x`, `t/g`, and `c/v` for rotation, space to toggle the gripper,
and `q` to discard/reset an attempt. The collector accepts an episode only
after the checker remains true for ten consecutive control cycles.
