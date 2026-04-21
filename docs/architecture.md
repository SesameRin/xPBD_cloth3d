# Architecture

The project is split into a thin CLI shim, an `xpbd/` package with the
implementation, and a vendored copy of the CLOTH3D toolkit that serves as
the data backend.

```
xPBD/
├── xpbd_cloth.py        ← back-compat shim: `from xpbd.cli import main`
├── xpbd/                ← the simulator package (this is the new home)
│   ├── __init__.py      re-exports the public API
│   ├── __main__.py      enables `python3 -m xpbd ...`
│   ├── cli.py           argparse → load → solver → viewer
│   ├── fabrics.py       FABRIC_PRESETS, fabric_params()
│   ├── geometry.py      build_edges, build_bending_pairs, normals, mass
│   ├── data.py          load_sample(): CLOTH3D loader + multi-garment merge
│   ├── solver.py        XPBDCloth class + Taichi kernels
│   ├── viewers.py       run_gui / run_matplotlib / run_headless / run_export
│   └── export.py        eval-compatible NPZ writer (see docs/eval_export.md)
├── cloth3d/             ← vendored CLOTH3D toolkit (DataReader, Demo, …)
│   ├── DataReader/      readOBJ, readPC2, SMPL, etc.
│   └── Demo/extract_sample_data.py   used by xpbd.data
├── tests/test_smoke.py  CPU smoke tests
├── docs/                ← these explanation files
├── xpbd_out/            ← rendered mp4s and headless .npy dumps
└── README.md
```

## Module responsibilities

### `xpbd.fabrics`
A single dict, `FABRIC_PRESETS`, mapping a CLOTH3D fabric tag (`cotton`,
`silk`, `denim`, `leather`) to four numbers: distance compliance, bend
compliance, damping, and areal density. Plus `fabric_params(name)` which
falls back to cotton on unknown names. **No NumPy, no Taichi.**

### `xpbd.geometry`
Pure NumPy helpers. Run **once** at setup, never in the hot loop:

- `build_edges(F)` — unique undirected edge list from a triangle list.
- `build_bending_pairs(F)` — for each shared edge between two triangles,
  return the two opposite vertices used by the PBD bending shortcut.
- `greedy_pair_coloring(pairs, n_vertices)` — greedy graph coloring of
  edges / bending pairs so that within one color class no two
  constraints share a vertex. Only used when `XPBDCloth(gpu_safe=True)`,
  to make parallel constraint writes race-free on GPU. See
  `docs/xpbd_method.md` §3 for why.
- `per_vertex_normals(V, F)` — area-weighted vertex normals for the
  body collider.
- `compute_vertex_masses(V, F, vert_gid, fabrics)` — per-vertex mass from
  per-triangle area times the owning garment's areal density.

### `xpbd.data`
`load_sample(sample, garments_spec, n_body_frames)` is the only public
entry. It wraps `cloth3d/Demo/extract_sample_data.py`, then concatenates
every requested garment into a single `(V, F, C)` and remembers a
`vert_gid` per vertex. Returned dict has `V0`, `F`, `C`, `vert_gid`,
`garment_names`, `garment_fabrics`, `body_V_seq`, `body_F`, `sample`.

### `xpbd.solver`
`XPBDCloth` is the work-horse class. The constructor:

1. Accepts the merged mesh + `vert_gid` + `garment_fabrics`.
2. Looks up fabric params per garment, applies any global override, and
   broadcasts compliance to **per-edge** and **per-bend** Taichi fields.
3. Computes per-vertex masses from `xpbd.geometry`.
4. Allocates Taichi fields and uploads everything once.

The hot loop is `step()` → `[predict → reset_lambdas → (solve_distance,
solve_bending, solve_collision) × iterations → finalize] × substeps`.
Each subloop op is a `@ti.kernel`. See `docs/xpbd_method.md` for the math.

`XPBDCloth(gpu_safe=True)` (set automatically when `--arch gpu` /
`--arch vulkan`) switches `solve_distance` and `solve_bending` to their
colored variants: edges/bending pairs are pre-sorted by a greedy vertex
coloring, and `step()` issues one kernel launch per color class. Within
a color class all threads touch disjoint vertices, so Taichi's auto-
atomic `+=` is actually race-free even on CUDA. Across color classes
the launches are sequential, so the overall scheme is still Gauss–
Seidel. CPU (`gpu_safe=False`) keeps the original single-launch
kernels, bit-identical to the pre-GPU implementation. The coloring +
kernel pair both live in `xpbd.solver`.

### `xpbd.viewers`
Four back-ends sharing one signature `(cloth, data, args)`:

- `run_gui` — Taichi GGUI (Vulkan).
- `run_matplotlib` — `Poly3DCollection` animation; can save mp4/gif.
- `run_headless` — no display; saves cloth vertex frames as `.npy`.
- `run_export` — headless sim that writes eval-compatible NPZ files
  (per-garment `{sample}_{garment}_sim.npz`) consumable by the
  teammate's `cloth3d_benchmark/cloth3d_eval` module. Routed when
  `--save_npz` is set. See `docs/eval_export.md`.

### `xpbd.export`
Pure-NumPy NPZ writer, no Taichi or solver dependency. Provides
`write_result_npz(...)`, `slice_garment(F, vert_gid, gid)`,
`rotate_x_minus_90(V)` (CLOTH3D z-up → libuipc y-up), and
`write_sample_npz(dir, sample)` for the per-sample CLOTH3D extraction
the eval's source loader reopens. See `docs/eval_export.md` for the
schema it produces and a step-by-step recipe for running the teammate's
eval against our outputs.

### `xpbd.cli`
`build_parser()` defines all CLI flags. `main()`:
1. Parses args.
2. Initialises Taichi with the chosen arch.
3. Calls `load_sample`.
4. Constructs `XPBDCloth`, optionally with `--force_fabric` rewriting
   every garment's fabric (used for the C-IPC comparison).
5. Dispatches to the right viewer (`auto` falls back from GGUI to mpl).

### `xpbd_cloth.py` (root)
A 5-line shim: `from xpbd.cli import main; main()`. Kept so that any
existing scripts or docs that say `python3 xpbd_cloth.py …` still work.

## Why this split

- **`cli` separated from solver** — tests and notebooks can construct an
  `XPBDCloth` directly without going through argparse.
- **`fabrics` is its own file** — adding a new material is a single-file
  edit; the solver code never needs to know which materials exist.
- **`geometry` is pure NumPy** — easy to test, easy to swap, doesn't
  drag in Taichi compile time when only inspecting topology.
- **`data` isolates the CLOTH3D dependency** — the `sys.path` gymnastics
  for the vendored `cloth3d/` toolkit live in exactly one place. If the
  CLOTH3D loader ever changes, only `xpbd/data.py` cares.
- **`viewers` is a flat module** — three small functions, no inheritance.

## Public API (what `import xpbd` gives you)

```python
from xpbd import (
    FABRIC_PRESETS, DEFAULT_FABRIC, fabric_params,
    build_edges, build_bending_pairs, per_vertex_normals, compute_vertex_masses,
    load_sample,
    XPBDCloth,
)
```

The CLI entry is `xpbd.cli.main`. Tests in `tests/test_smoke.py` import
the package as `import xpbd as xc`, demonstrating the supported usage.
