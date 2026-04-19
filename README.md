# XPBD Cloth on CLOTH3D

A Taichi [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)
cloth simulator for the [CLOTH3D](https://chalearnlap.cvc.uab.es/dataset/38/description/)
dataset. Loads a sample, simulates every garment in its outfit at once
with **fabric-aware** parameters, and renders through GGUI / matplotlib
/ headless `.npy` dumps.

> **Want to understand the code, not just run it?** Start with
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
python3 xpbd_cloth.py --sample 00016 --garments Tshirt --viewer mpl --save_video --body_frames 60 --steps 60

# Match the C-IPC cotton baseline (force every garment to cotton)
python3 xpbd_cloth.py --sample 07414 --garments all --force_fabric cotton \
    --body_frames 90 --viewer mpl --save_video

# Headless: dump cloth vertices as .npy every N steps
python3 xpbd_cloth.py --sample 07414 --garments all \
    --viewer none --steps 300 --save_every 10
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
| `--body_frames` | `1` | >1 animates the body collider over the first N frames |
| `--viewer` | `auto` | `auto` / `ggui` / `mpl` / `none` |
| `--save_video` | off | with `--viewer mpl`, render mp4 (or gif fallback) |
| `--arch` | `cpu` | Taichi backend (`cpu`, `gpu`, `vulkan`) |
| `--dt` | `1/60` | frame step |
| `--substeps` | `10` | XPBD substeps per frame |
| `--iters` | `5` | constraint iterations per substep |
| `--dist_compliance` | per-fabric | override stretch compliance |
| `--bend_compliance` | per-fabric | override bending compliance |
| `--damping` | per-fabric | override per-substep damping |
| `--collision_radius` | `0.01` | body pushout distance, m (also the contact thickness) |
| `--friction` | `0.6` | fraction of body's tangential motion inherited by cloth in contact (0 = frictionless, slides off; 1 = fully stuck) |
| `--friction_capture` | `0.03` | friction engages within this shell thickness (m) around the body; set to 0 to disable friction |

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
                     solver / viewers / cli)
cloth3d/             vendored CLOTH3D toolkit (DataReader, Demo, Samples)
tests/               smoke tests
docs/                explanation files (start here for the deep dive)
xpbd_out/            rendered mp4s and headless .npy frame dumps
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
- `--arch gpu` uses Taichi's CUDA backend. The solver prints per-stage
  timing at the end of each run. See
  [`docs/gpu_performance.md`](docs/gpu_performance.md) for the CPU vs
  GPU numbers on this workload and the Jacobi-with-valence-scaling
  fix that was needed to make the GPU path stable for stiff fabrics.
- For fastest iteration, **use `--viewer ggui` when you don't need a
  saved video**. GGUI renders on the GPU directly from the Taichi
  fields; matplotlib round-trips every vertex to NumPy and redraws
  polygons on the CPU, which ends up dominating wall-clock time
  (~90% on a 6k-vertex mesh).

### Keeping garments on the body

If the dress / trousers slide off the legs when the body animates, that
is the collision model being pure non-penetration — with no friction,
nothing couples cloth tangentially to the skin and gravity slowly
slips the hem down. The knobs, strongest first:

- **`--friction 0.6 → 0.9`**: more body-tangent motion transferred to
  cloth in contact. `0.9` makes cloth follow the body like it's Velcroed;
  `0.3` is "ice-skater silk."
- **`--friction_capture 0.03 → 0.06`**: wider shell where friction
  engages. Useful if the dress is separated from the skin by a few cm.
- **`--collision_radius 0.01 → 0.02`**: thicker non-penetration layer,
  so cloth sits further out and has more slack before it can clear
  the hip/shoulder.
- **`--damping 0.05`** on top: overrides per-fabric damping; heavier
  damping removes "swing" momentum that would otherwise slide cloth
  off in the first few frames.

For a sticky-dress baseline try: `--friction 0.8 --friction_capture
0.04 --collision_radius 0.015`.
