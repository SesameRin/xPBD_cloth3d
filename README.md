# XPBD Cloth on CLOTH3D

A minimal [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)
cloth simulator (Taichi) that runs on garments from the
[CLOTH3D](https://chalearnlap.cvc.uab.es/dataset/38/description/) dataset.
Loads a sample, initialises every garment in the outfit at frame 0, and
simulates them under gravity with the SMPL body as a (optionally animated)
collider.

Each garment is simulated with **fabric-aware** XPBD parameters that are
keyed to the CLOTH3D `fabric` tag (`cotton`, `silk`, `denim`, `leather`),
so a Tshirt + Trousers outfit with mixed materials behaves differently per
garment in the same run.

## Install

```bash
pip install taichi numpy scipy pillow plotly tqdm matplotlib imageio-ffmpeg
```

Samples must live under `cloth3d/Samples/<id>/` (same layout as shipped).
The bundled samples are: `00016`, `01691`, `03543`, `06840`, `07414`.

## Run

```bash
# Multi-garment outfit, all garments at once (07414 is Tshirt + Trousers, both cotton)
python3 xpbd_cloth.py --sample 07414 --garments all --body_frames 60

# Single garment (back-compat alias also works: --garment Tshirt)
python3 xpbd_cloth.py --sample 00016 --garments Tshirt --body_frames 60

# Pick a subset by name
python3 xpbd_cloth.py --sample 00016 --garments Tshirt,Trousers

# Force every garment to the same fabric (matches a C-IPC cotton-only baseline)
python3 xpbd_cloth.py --sample 07414 --garments all --force_fabric cotton --body_frames 60

# Save an mp4 (no display needed)
python3 xpbd_cloth.py --sample 07414 --garments all --viewer mpl --save_video \
    --body_frames 60 --steps 60

# Headless: dump cloth vertices as .npy every N steps
python3 xpbd_cloth.py --sample 07414 --garments all \
    --viewer none --steps 300 --save_every 10
```

Output goes to `xpbd_out/`. Filenames now include all simulated garments
(e.g. `07414_Tshirt+Trousers.mp4`).

## Common flags

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

## Solver knobs

| Flag | Default |
|---|---|
| `--dt` | `1/60` (frame step) |
| `--substeps` | `10` |
| `--iters` | `5` |
| `--dist_compliance` | per-fabric (override global) |
| `--bend_compliance` | per-fabric (override global) |
| `--damping` | per-fabric (override global) |
| `--collision_radius` | `0.01` m (body pushout distance) |

If you do not pass `--dist_compliance` / `--bend_compliance` / `--damping`,
each garment uses the preset for its CLOTH3D fabric.

## Fabric presets

These map the CLOTH3D `fabric` field to XPBD parameters. Cotton is the
anchor used for the C-IPC comparison.

| Fabric | distance compliance | bend compliance | damping | density (kg/m²) |
|---|---|---|---|---|
| `cotton`  | `5.0e-9`  | `1.0e-5`  | `0.03` | `0.30` |
| `silk`    | `2.0e-8`  | `5.0e-7`  | `0.01` | `0.10` |
| `denim`   | `1.0e-9`  | `5.0e-5`  | `0.05` | `0.45` |
| `leather` | `5.0e-10` | `2.0e-4`  | `0.08` | `0.80` |

Tuning convention: lower compliance ⇒ stiffer constraint. Cotton is moderately
stiff in stretch and somewhat stiff in bending; silk is the opposite (limp
and very low-bending); denim and leather are progressively stiffer in both.

## Comparing against C-IPC

The C-IPC pipeline in this project uses cotton as a temporary single
material across an outfit. To produce a like-for-like XPBD comparison run:

```bash
python3 xpbd_cloth.py --sample 07414 --garments all --force_fabric cotton \
    --body_frames 90 --viewer mpl --save_video
```

Both runs then produce a sequence of cloth meshes for the same body
animation that can be compared visually or via vertex-to-surface distance
(see `cloth3d/` for evaluation tooling).

## GGUI controls (when available)

`space` pause · `r` reset · `b` toggle body · `esc` quit ·
right-mouse drag to orbit.

## How it works

- **Data**: `load_sample()` calls `extract_sample_data.extract_sample_single_frame`,
  then merges every garment requested by `--garments` into a single combined
  mesh while remembering which vertices belong to which garment (so per-garment
  fabric parameters can be applied).
- **Solver** (`XPBDCloth`): per substep — predict → reset λ → iterate
  (distance / bending / collision) → update positions & velocities. Each
  edge and bending pair carries its garment's compliance, so a multi-fabric
  outfit just works without retuning.
- **Mass**: per-vertex mass is derived from per-triangle area times the
  garment's areal density.
- **Collision**: nearest-body-vertex pushout along that vertex's normal.
- **Viewer**: Taichi GGUI scene mesh, or matplotlib `Poly3DCollection` as
  fallback.

## Tests

```bash
python3 -m pytest tests/ -q          # if pytest is available
# or:
python3 tests/test_smoke.py          # standalone, prints PASS/FAIL
```

The smoke tests cover: data loading, multi-garment merge for `07414`
(Tshirt + Trousers, both cotton), edge / bending construction, and one
XPBD step on CPU.

## Notes

- GGUI requires Vulkan. On WSL without a Vulkan driver, use `--viewer mpl`
  (with `--save_video` for headless) — this is the fallback the `auto` mode
  picks automatically.
- CLOTH3D is z-up, meters.
- Cloth starts from the dataset's frame-0 draped pose (already body-fitted).
  For a "drop from flat" experiment, swap `_V` → `_V_rest` in `load_sample`.
