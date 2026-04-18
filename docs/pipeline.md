# End-to-end pipeline

This walks one full run from `python3 xpbd_cloth.py …` to a saved mp4,
naming the file and function each stage lands in.

```
                    ┌────────────────────────────────────────────┐
CLI / argparse  ───►│ xpbd/cli.py            build_parser, main  │
                    └───────────────┬────────────────────────────┘
                                    │ args
                                    ▼
                    ┌────────────────────────────────────────────┐
load CLOTH3D    ───►│ xpbd/data.py           load_sample          │
                    │  ├── extract_sample_single_frame (cloth3d)  │
                    │  ├── _parse_garment_list                    │
                    │  └── concatenate V, F, C; build vert_gid    │
                    └───────────────┬────────────────────────────┘
                                    │ data dict
                                    ▼
                    ┌────────────────────────────────────────────┐
construct solver ──►│ xpbd/solver.py         XPBDCloth.__init__   │
                    │  ├── geometry.build_edges                   │
                    │  ├── geometry.build_bending_pairs           │
                    │  ├── geometry.compute_vertex_masses         │
                    │  ├── fabrics.fabric_params per garment      │
                    │  └── allocate Taichi fields, upload static  │
                    └───────────────┬────────────────────────────┘
                                    │ cloth (Taichi state)
                                    ▼
                    ┌────────────────────────────────────────────┐
viewer dispatch  ──►│ xpbd/cli.py            (auto / ggui / mpl / │
                    │                         none)               │
                    └───────────────┬────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────────────┐
              ▼                     ▼                             ▼
      run_gui (GGUI)         run_matplotlib                 run_headless
      xpbd/viewers.py        xpbd/viewers.py                xpbd/viewers.py
              │                     │                             │
              └────────►  cloth.step() per displayed frame  ◄─────┘
                                    │
                                    ▼
                       XPBDCloth.step(): see xpbd_method.md
```

## What flows between stages

### CLI → loader
- `args.sample`, `args.garments`, `args.body_frames`.
- Loader receives a string spec and returns a fully-merged numpy view of
  the cloth and the body collider.

### Loader → solver
The loader returns one dict. Keys consumed by the solver constructor:

| key | shape | role |
|---|---|---|
| `V0`         | `(N, 3) float32` | initial particle positions |
| `F`          | `(M, 3) int32`   | combined triangle list |
| `vert_gid`   | `(N,) int32`     | which garment each vertex belongs to |
| `garment_fabrics` | list of str | one fabric tag per garment |
| `body_V_seq` | `(T, 6890, 3)`   | SMPL body frames |
| `body_F`     | `(NBF, 3)`       | SMPL face list |

`C` (vertex colors) and `garment_names` are used by viewers, not by the
solver.

### Solver → viewer
Each viewer touches `cloth.x` (positions), `cloth.face_idx`,
`cloth.body_x`, `cloth.body_face_idx`, and `cloth.color`. None of them
mutate the solver state — they only call `cloth.step()` and read fields.

### Per-frame loop (any viewer)
```
if multi-frame body:
    cloth.set_body(body_V_seq[i % T])
cloth.step()                  # runs the XPBD substeps
read cloth.x.to_numpy()       # for rendering or saving
```

## File reading order, top to bottom

To trace a run by hand from a fresh checkout:

1. `xpbd_cloth.py` — confirms it is just a shim into `xpbd.cli.main`.
2. `xpbd/cli.py:main` — argparse, `ti.init`, then `load_sample`.
3. `xpbd/data.py:load_sample` — joins garments and reads SMPL frames.
4. `xpbd/solver.py:XPBDCloth.__init__` — see how compliance is broadcast.
5. `xpbd/solver.py:XPBDCloth.step` — the per-frame work.
6. `xpbd/solver.py` Taichi kernels (`predict`, `solve_distance`,
   `solve_bending`, `solve_collision`, `finalize`).
7. `xpbd/viewers.py` — pick whichever viewer matches your `--viewer`.

Cross-reference each step with `docs/xpbd_method.md` for the math.
