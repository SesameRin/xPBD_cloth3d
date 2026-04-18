# Viewers

`xpbd/viewers.py` exposes three back-ends. The CLI chooses one via
`--viewer {auto, ggui, mpl, none}`. They all share the same signature
`run_X(cloth, data, args)` and never mutate the solver state.

## `run_gui` — Taichi GGUI (Vulkan)

Interactive 3D window. Best when you want to orbit the cloth, pause,
toggle the body collider, or reset to frame 0 mid-run.

| Key | Action |
|---|---|
| Space | pause / resume |
| `r` | reset cloth to `data["V0"]`, zero velocities |
| `b` | toggle body collider visibility |
| RMB drag | orbit the camera |
| Esc | quit |

Per-frame loop: optionally swap the body via `cloth.set_body`, call
`cloth.step()`, then push `cloth.x`, `cloth.face_idx`, and
`cloth.color` into a `scene.mesh`.

**Requires Vulkan.** WSL setups without a Vulkan driver should use
`mpl` or rely on `auto` to fall back automatically.

## `run_matplotlib` — `mpl`

3D animation built on `matplotlib.animation.FuncAnimation` and
`Poly3DCollection`. Two modes:

- Default: opens a window (`plt.show()`).
- `--save_video`: writes `xpbd_out/<sample>_<garments>.mp4` (or `.gif`
  if ffmpeg is unavailable). Uses `imageio_ffmpeg` to find an ffmpeg
  binary; on environments without it, the writer falls back to
  `PillowWriter` and `.gif`.

Cheap, slow, requires no GPU. The right choice for headless rendering
or when you just want a clip to share.

## `run_headless` — `none`

No display, no rendering. Per `--save_every` steps, dumps cloth vertex
positions to `xpbd_out/<sample>_<garments>_<step>.npy`. Useful for:

- Producing input for a separate evaluation pipeline (e.g.
  vertex-to-surface distance against a C-IPC baseline).
- Long batch runs on remote hardware.
- Any time you want the simulation but don't need pixels.

## Choosing a back-end

| You want… | Use |
|---|---|
| Interactive orbit, fast iteration, you have Vulkan | `ggui` |
| A clip to embed in a slide / paper / chat | `mpl --save_video` |
| Numbers for evaluation, no pictures | `none` |
| "Just show me something" without thinking | `auto` (tries ggui, falls back to mpl) |

## Output filenames

All three viewers (when they write to disk) use the same stem:
`<sample>_<garment1>+<garment2>+...`. This is built by
`viewers.video_stem(data)`, so a multi-garment run on `07414` produces
`07414_Trousers+Tshirt.mp4` (or `.npy` frames), and a single-garment
run on `00016` produces `00016_Tshirt.mp4`. This is intentional — it
makes diffing single- vs. multi-garment runs easy from the filename.
