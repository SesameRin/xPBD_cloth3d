# XPBD Cloth on CLOTH3D

A minimal [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)
cloth simulator (Taichi) that runs on garments from the
[CLOTH3D](https://chalearnlap.cvc.uab.es/dataset/38/description/) dataset.
Loads a sample, initializes the cloth at frame 0, and simulates it under
gravity with the SMPL body as a (optionally animated) collider.

## Install

```bash
pip install taichi numpy scipy pillow plotly tqdm matplotlib imageio-ffmpeg
```

Samples must live under `cloth3d/Samples/<id>/` (same layout as shipped).
The bundled samples are: `00016`, `01691`, `03543`, `06840`, `07414`.

## Run

```bash
# Interactive viewer (auto: Taichi GGUI if Vulkan, else matplotlib window)
python3 xpbd_cloth.py --sample 00016 --garment Tshirt --body_frames 60

# Save an mp4 (no display needed)
python3 xpbd_cloth.py --sample 00016 --garment Tshirt --viewer mpl --save_video --body_frames 60 --steps 60

# Headless: dump cloth vertices as .npy every N steps
python3 xpbd_cloth.py --sample 00016 --garment Tshirt \
    --viewer none --steps 300 --save_every 10
```

Output goes to `xpbd_out/` (video or `.npy` frames).

## Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--sample` | `00016` | CLOTH3D sample id |
| `--garment` | first in outfit | e.g. `Tshirt`, `Trousers` |
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
| `--dist_compliance` | `1e-8` (stretch — lower = stiffer) |
| `--bend_compliance` | `5e-6` (bending) |
| `--damping` | `0.02` (per-substep velocity damping) |
| `--collision_radius` | `0.01` m (body pushout distance) |

## GGUI controls (when available)

`space` pause · `r` reset · `b` toggle body · `esc` quit ·
right-mouse drag to orbit.

## How it works

- Data: `load_sample()` calls `extract_sample_data.extract_sample_single_frame`
  for the initial cloth + topology, and `DataReader.read_human` for each body
  frame.
- Solver (`XPBDCloth`): per substep — predict → reset λ → iterate
  (distance / bending / collision) → update positions & velocities. Collision
  is nearest-body-vertex pushout along that vertex's normal.
- Viewer: Taichi GGUI scene mesh, or matplotlib `Poly3DCollection` as fallback.

## Notes

- GGUI requires Vulkan. On WSL without a Vulkan driver, use `--viewer mpl`
  (with `--save_video` for headless) — this is the fallback the `auto` mode
  picks automatically.
- CLOTH3D is z-up, meters.
- Cloth starts from the dataset's frame-0 draped pose (already body-fitted).
  For a "drop from flat" experiment, swap `_V` → `_V_rest` in `load_sample`.
