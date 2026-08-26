# LIBERO Demonstration Replay

## Purpose

This branch establishes the replay half of the ICL demonstration pipeline. A
candidate demonstration is accepted only when its recorded actions can be
executed again in LIBERO and the task checker remains successful at the end.

The replay is physical action replay:

1. Construct a fresh environment from the task's current BDDL file.
2. Reset the environment and restore the episode's flattened MuJoCo
   `init_state` once.
3. Advance ten no-motion controller steps so the restored scene and OSC
   controller settle.
4. Execute every recorded 7D OSC action through `env.step(action)`.
5. Evaluate the task checker after every action and require stable final
   success.

Saved trajectory states are never forced after initialization. Consequently,
the generated images and final success are causal results of the recorded
actions rather than state-playback renderings.

## Available LIBERO data paths

Two forms of the existing dataset were inspected:

- `/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/libero`
  contains LeRobot parquet/video conversions. These contain observations and
  actions but not the complete MuJoCo initialization state or model metadata.
- `/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/libero_hdf5`
  contains the full converted LIBERO HDF5 files. Each episode includes the 7D
  action sequence and a flattened `init_state`, so this is the replay source.

For example, the selected first smoke demonstration is:

```text
suite: libero_object
task: pick up the alphabet soup and place it in the basket
episode: demo_0
actions: 148 x 7
init_state: 110 values
```

The HDF5 `model_file` attribute is retained as provenance, but many official
files contain absolute asset paths from the original collection machine. The
replay tool therefore resolves the recorded path below `bddl_files/` into the
current checkout, creates the current task environment, and restores the saved
initial state. An explicit `--bddl-file` override remains available.

## Existing collection and replay support

LIBERO already contains the building blocks, but not a standalone validation
command:

- `scripts/collect_demonstration.py` performs interactive keyboard or
  SpaceMouse collection using robosuite's `DataCollectionWrapper`.
- `scripts/create_dataset.py` re-executes recorded actions while converting raw
  collections into the published HDF5 schema.
- robosuite also ships a generic HDF5 playback utility supporting state or
  action playback.

LIBERO does not ship a task-specific scripted expert for the alphabet-soup
task. New demonstrations therefore require human teleoperation or another
policy. For the initial ICL pipeline, a physically revalidated official HDF5
episode is the simplest source.

## Replay command

Run from the repository root in the LIBERO conda environment:

```bash
python scripts/replay_demonstration.py \
  --dataset /inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/libero_hdf5/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --episode demo_0 \
  --save-video
```

By default, artifacts are written beneath:

```text
outputs/replay/<dataset-name>/<episode>/
├── replay_report.json
└── replay.mp4                 # only with --save-video
```

The video places `agentview` on the left and `robot0_eye_in_hand` on the right.
It is an evaluator-side replay artifact, not yet an Agent-visible ICL bundle.

The command exits with status 2 when stable checker verification fails. Use
`--allow-failure` only for diagnostic replay of known failed episodes.

### GPU rendering with EGL

The plain Python command defaults to OSMesa for portability. On the current
RTX 4090 host, use the EGL launcher instead:

```bash
bash scripts/launch_replay_egl.sh \
  --dataset /inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/dataset/libero_hdf5/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --episode demo_0 \
  --save-video
```

The launcher selects the data-disk NVIDIA userspace bundle whose version
matches `/proc/driver/nvidia/version`. It reuses the `runtime-libs-full`
symlink farm prepared for UniVTAC and does not copy driver libraries into the
LIBERO conda environment. `CUDA_VISIBLE_DEVICES` selects the GPU and defaults
to device 0.

This checkout uses one machine-local link to expose every prepared version:

```text
runtime/nvidia -> /inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia
```

The link is ignored by Git because its absolute target is deployment-specific.
At startup, the launcher appends the host kernel-driver version to this path;
it refuses to mix a userspace bundle with a different kernel driver. The bundle
parent currently provides versions 550.163.01, 570.124.06, 570.195.03, and
595.58.03.

The replay report records `MUJOCO_GL` together with the OpenGL vendor,
renderer, and version, making an accidental software-rendering fallback
visible during audit.

## Verification result

On 2026-08-26, `demo_0` of the alphabet-soup task was replayed in the current
checkout with seed 0 and ten settling steps. All 148 recorded actions were
executed. The checker first became true at action 120, and the completed
trajectory ended with twelve consecutive successful action steps. Both physics
only, OSMesa, and EGL replay produced the same result. The EGL run reported
`NVIDIA GeForce RTX 4090/PCIe/SSE2` with NVIDIA driver `570.124.06`.

## Boundary with the later ICL exporter

The replay report is internal and contains absolute source paths for audit. It
must not be copied directly into an Agent workspace. A later ICL exporter will:

- select only demonstrations with `verified_success: true`;
- capture the modalities required by the selected observation level;
- project them through the public observation allowlist;
- replace host paths with bundle-relative paths;
- expose the resulting static bundle to the Agent.
