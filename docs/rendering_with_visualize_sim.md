# Rendering xPBD results with `visualize_sim.py`

The teammate's `cloth3d-ipc-xpbd` repo ships a standalone renderer that
turns any `*_sim.npz` result into an MP4/GIF with the same look as the
Polyscope IPC viewer (steel-blue sim cloth, teal-green GT cloth,
warm-skin body, dark background). Our xPBD export already writes the
exact NPZ schema that renderer expects, so no format conversion is
needed — only a cross-environment command.

## Why this works out of the box

`xpbd/export.py::write_result_npz` emits these keys:

- `faces`, `sim_V_seq`, `gt_V_seq`
- `human_faces`, `human_V_seq`
- `sample`, `garment_name`, `solver_family`, `ground_enabled`

`visualize_sim.py` reads exactly these keys. File naming
`<sample>_<garment>/<sample>_<garment>_sim.npz` also matches its
`**/*_sim.npz` glob, so `--source results_xpbd` works if the NPZs
live under that directory.

No libuipc code is involved. `visualize_sim.py` uses PyVista (GPU) or
matplotlib (CPU) — libuipc is only mentioned in its docstring because
the teammate's conda env happens to ship those packages.

## Environment layout

Two conda envs, one NPZ schema:

| Env | Role | Key packages |
|---|---|---|
| your xPBD env | simulate, export NPZ | `taichi`, `numpy` |
| `uipc_env` | render NPZ → MP4/GIF | `matplotlib` (CPU) or `pyvista` + `imageio[ffmpeg]` (GPU) |

We use `conda run -n uipc_env ...` to invoke the renderer without
switching shells.

## Prerequisite: install the renderer backend

`uipc_env` already has matplotlib and numpy, but neither renderer path
works until `imageio[ffmpeg]` is installed — both CPU and GPU paths use
`imageio.get_writer` to produce the video file.

### CPU renderer (matplotlib)

```bash
conda run -n uipc_env pip install "imageio[ffmpeg]"
```

Pulls in `imageio-ffmpeg`, which bundles its own ffmpeg binary. No
system ffmpeg install needed. Works headless; safe default for WSL2.

### GPU renderer (PyVista / VTK)

```bash
conda run -n uipc_env pip install pyvista "imageio[ffmpeg]"
```

PyVista wheels include VTK, adds ~150 MB. Faster (seconds per clip
instead of seconds per frame), but needs a working OpenGL context.

**WSL2 note.** VTK off-screen on WSL2 usually needs a virtual display:

```bash
sudo apt install xvfb                  # one-time
xvfb-run -a conda run -n uipc_env python .../visualize_sim.py ...
```

If you are on native Linux with a GPU driver or on WSLg with OpenGL
working, you can drop `xvfb-run`.

## Quickstart

A real export already lives at e.g.
`xpbd_out/batch_20260421_165349/results_xpbd/00016_Trousers/00016_Trousers_sim.npz`.
Render it with:

```bash
conda run -n uipc_env python \
  /home/ula/CMU/pba-proj/cloth3d-ipc-xpbd/cloth3d_benchmark/visualize_sim.py \
  --npz /home/ula/CMU/pba-proj/xPBD/xpbd_out/1_100_xpbd/results_xpbd/00016_Trousers/00016_Trousers_sim.npz \
  --renderer gpu \
  --cloth both \
  --format mp4
```

Output lands next to the NPZ as `00016_Trousers_sim_both.mp4`. Override
with `--output PATH`.

## Useful flags

| Flag | Purpose |
|---|---|
| `--cloth sim` / `gt` / `both` | Simulated only, GT only, or overlaid (default: `sim`) |
| `--renderer cpu` / `gpu` | Software matplotlib vs PyVista/VTK |
| `--format mp4` / `gif` | Container |
| `--fps 30` | Frame rate |
| `--max_frames N` | Truncate long sequences for a quick preview |
| `--show_body off` | Cloth-only clip |
| `--width 960 --height 1080` | Video resolution (GPU only) |
| `--elev 18 --azim -55` | Camera angle |

## Wrapper script

Convenience wrapper so you don't retype the path. Save as
`scripts/render_xpbd.sh` at the repo root and `chmod +x` it:

```bash
#!/usr/bin/env bash
# Usage: ./scripts/render_xpbd.sh path/to/xxx_sim.npz [extra flags]
set -e
VISUALIZE_SIM=/home/ula/CMU/pba-proj/cloth3d-ipc-xpbd/cloth3d_benchmark/visualize_sim.py
NPZ="$1"; shift
conda run -n uipc_env python "$VISUALIZE_SIM" \
    --npz "$NPZ" --renderer cpu --cloth both --format mp4 "$@"
```

Then:

```bash
./scripts/render_xpbd.sh xpbd_out/batch_.../00016_Trousers/00016_Trousers_sim.npz
./scripts/render_xpbd.sh <npz> --renderer gpu            # once PyVista is installed
./scripts/render_xpbd.sh <npz> --max_frames 60 --format gif
```

## Troubleshooting

- `ModuleNotFoundError: imageio` — install `imageio[ffmpeg]` in
  `uipc_env` (see above).
- `ModuleNotFoundError: pyvista` — only needed if you pass
  `--renderer gpu`. Install it or fall back to `--renderer cpu`.
- PyVista opens but the video is blank — you need an OpenGL context.
  Wrap the call with `xvfb-run -a` (see WSL2 note).
- `No *_sim.npz files found` when using `--source results_xpbd` —
  either pass `--npz PATH` directly, or symlink your batch output into
  `cloth3d-ipc-xpbd/cloth3d_benchmark/results_xpbd/`.
