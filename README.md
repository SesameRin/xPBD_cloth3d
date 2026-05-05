# XPBD Cloth on CLOTH3D

Note: This repo is the xpbd part of the cloth3d solver benchmark. For the full benchmark codebase, see https://github.com/wuwenglei/cloth3d-ipc-xpbd/.

A Taichi [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)
cloth simulator for the [CLOTH3D](https://chalearnlap.cvc.uab.es/dataset/38/description/)
dataset. Loads a sample, simulates every garment in its outfit at once
with **fabric-aware** parameters, and renders through GGUI / matplotlib
/ headless `.npy` dumps.

> **Understand the code:** Start with
> [`docs/README.md`](docs/README.md) — module map, data flow, the XPBD
> math, fabric tuning, and the C-IPC comparison recipe.

## Install

```bash
pip install taichi numpy scipy pillow plotly tqdm matplotlib imageio-ffmpeg
```

Samples must live under `cloth3d/Samples/<id>/` (same layout as shipped).
Bundled samples: `00016`, `01691`, `03543`, `06840`, `07414`.

## Run

```bash
# Multi-garment (07414 is Tshirt + Trousers, both cotton)
python3 xpbd_cloth.py --sample 07414 --garments all --body_frames 60

# Single garment, save an mp4
python3 xpbd_cloth.py --sample 00016 --garments Tshirt \
    --viewer mpl --save_video --body_frames 60 --steps 60

# Match the C-IPC cotton baseline (force every garment to cotton)
python3 xpbd_cloth.py --sample 07414 --garments all --force_fabric cotton \
    --body_frames 90 --viewer mpl --save_video

# Headless: dump cloth vertices as .npy every N steps
python3 xpbd_cloth.py --sample 07414 --garments all \
    --viewer none --steps 300 --save_every 10

# Export eval-compatible NPZ files (teammate's cloth3d_benchmark schema).
# Omit --body_frames / --steps to simulate the whole sample automatically.
python3 xpbd_cloth.py --sample 00016 --garments all \
    --arch gpu --save_npz --save_sample_npz

# Drop experiment: lift the cloth 3 m above the body and let it fall.
# Mirrors partner's IPC drop run (--garment_y_translation 3.0 in y-up).
python3 xpbd_cloth.py --sample 00007 --garments Tshirt \
    --arch gpu --garment_y_translation 3.0 \
    --save_npz --save_sample_npz \
    --npz_out xpbd_out/results_xpbd_drop

# Batch: simulate every sample under cloth3d/Samples/ into one timestamped folder.
# --save_sample_npz defaults on here (pass --no_save_sample_npz to skip).
python3 -m xpbd.batch --arch gpu
```

`python3 -m xpbd …` also works. Output goes to `xpbd_out/`; filenames
include every simulated garment, e.g. `07414_Trousers+Tshirt.mp4`.

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--sample` | `00016` | CLOTH3D sample id |
| `--garments` | `all` | comma list (`Tshirt,Trousers`) or `all` |
| `--garment` | `None` | deprecated single-garment alias |
| `--force_fabric` | off | override fabric for *all* garments (`cotton`, `silk`, `denim`, `leather`) |
| `--body_frames` | auto (full sample) | number of body frames to animate; omit or pass `-1` to use every frame `get_num_frames(sample)` returns |
| `--steps` | auto (= `--body_frames`) | simulation steps to run; omit or pass `-1` to match `--body_frames` |
| `--viewer` | `auto` | `auto` / `ggui` / `mpl` / `none` |
| `--save_video` | off | with `--viewer mpl`, render mp4 (or gif fallback) |
| `--arch` | `cpu` | Taichi backend (`cpu`, `gpu`, `vulkan`). Non-cpu backends auto-enable GPU-safe graph-colored constraint solves — see [`docs/xpbd_method.md`](docs/xpbd_method.md) §3. |
| `--dt` | `1/60` | frame step |
| `--substeps` | `10` | XPBD substeps per frame |
| `--iters` | `5` | constraint iterations per substep |
| `--dist_compliance` | per-fabric | override stretch compliance |
| `--bend_compliance` | per-fabric | override bending compliance |
| `--damping` | per-fabric | override per-substep damping |
| `--collision_radius` | `0.01` | body pushout distance, m |
| `--save_npz` | off | write per-garment `_sim.npz` files for the benchmark eval — see [`docs/eval_export.md`](docs/eval_export.md) |
| `--npz_out` | `xpbd_out/results_xpbd` | output dir for `--save_npz` |
| `--save_sample_npz` | off | also extract per-sample CLOTH3D `{sample}.npz` the eval reopens |
| `--sample_npz_dir` | `xpbd_out/cloth3d_data` | output dir for `--save_sample_npz` |
| `--garment_y_translation` | `0.0` | drop-experiment lift in metres along z (z-up); pass `3.0` to mirror partner's IPC drop. See [`docs/drop_experiment.md`](docs/drop_experiment.md) |
| `--freeze_body` / `--no_freeze_body` | auto | hold the SMPL body collider at frame 0 (mirrors partner's `--freeze_human_mesh on`). Auto-on whenever `--garment_y_translation != 0`; force off with `--no_freeze_body`. |

If you do not pass `--dist_compliance` / `--bend_compliance` /
`--damping`, each garment uses the preset for its CLOTH3D fabric.
See [`docs/fabric_presets.md`](docs/fabric_presets.md) for the table
and tuning advice.

## GGUI controls (when `--viewer ggui`)

`space` pause · `r` reset · `b` toggle body · `esc` quit ·
right-mouse drag to orbit.

## Tests

```bash
python3 -m pytest tests/ -q          # if pytest is available
python3 tests/test_smoke.py          # standalone, prints PASS/FAIL
```

Covers loading, multi-garment merge for `07414` (Tshirt + Trousers,
both cotton), edge / bending construction, `force_fabric` override,
and one CPU XPBD step.

## Where things live

```
xpbd_cloth.py        thin shim → xpbd.cli.main
xpbd/                the simulator package (fabrics / geometry / data /
                     solver / viewers / cli / export)
cloth3d/             vendored CLOTH3D toolkit (DataReader, Demo, Samples)
tests/               smoke tests
docs/                explanation files (start here for the deep dive)
xpbd_out/            rendered mp4s, headless .npy dumps, and
                     results_xpbd/ + cloth3d_data/ for benchmark eval
```

For more, read [`docs/architecture.md`](docs/architecture.md).

## Notes

- GGUI requires Vulkan. WSL setups without Vulkan should use
  `--viewer mpl` (with `--save_video` for headless) — this is also the
  fallback the `auto` viewer picks.
- CLOTH3D is z-up, meters.
- Cloth starts from the dataset's frame-0 draped pose (already
  body-fitted). For a "drop from flat" experiment, swap `_V` →
  `_V_rest` in `xpbd/data.py`.
