# Data pipeline

How a CLOTH3D sample on disk becomes a single combined cloth + body
collider that the solver consumes.

## CLOTH3D on disk

```
cloth3d/Samples/<id>/
    info.mat            poses (72×T), shape (10,), trans (3×T), outfit, camera
    <Garment>.obj       rest mesh + UVs    (e.g. Tshirt.obj, Trousers.obj)
    <Garment>.pc16      animated vertex positions (float16)
    <Garment>.png       texture (optional, only for UV-based colors)
    <id>.mkv            rendered RGB video (not used by the solver)
```

`info["outfit"]` is the source of truth for which garments belong to
the sample and what fabric each one is. Five sample fabrics are present
in the bundled samples:

| Sample | Outfit |
|---|---|
| `00016` | Trousers (cotton), Tshirt (silk) |
| `01691` | Trousers (leather), Tshirt (silk) |
| `03543` | Dress (silk) |
| `06840` | Jumpsuit (leather) |
| `07414` | Trousers (cotton), Tshirt (cotton) |

## The vendored CLOTH3D toolkit

`cloth3d/DataReader/read.py` exposes `DataReader` with the readers we
care about:

- `read_info(sample)` → dict from `info.mat`.
- `read_garment_vertices(sample, garment, frame)` → `(N, 3)` from
  `<Garment>.pc16`.
- `read_garment_topology(sample, garment)` → triangulated faces.
- `read_human(sample, frame)` → SMPL body vertices + faces (6890 verts,
  13776 faces) by running the SMPL forward pass with the sample's pose,
  shape, and translation.

`cloth3d/Demo/extract_sample_data.py` wraps these into
`extract_sample_single_frame(sample, frame)`, which returns one flat
dict that includes both individual per-garment arrays
(`garment_<name>_V/F/C/V_rest/E/fabric`) and pre-merged versions
(`garments_merged_*`).

## What the simulator uses

`xpbd/data.py: load_sample(sample, garments_spec, n_body_frames)` is the
only consumer of the CLOTH3D toolkit in the solver package. It does
three things:

1. Calls `extract_sample_single_frame(sample, 0)` to get frame-0 cloth
   data (the cloth is already body-fitted at frame 0 — there is no
   "drop from flat" warm-up). It uses `use_uv_map=False`, so colors are
   the synthetic per-garment gradient, not the texture.
2. Resolves the `garments_spec` argument:
   - `None` or `"all"` → every garment in the outfit.
   - `"Tshirt"` → just the named garment.
   - `"Tshirt,Trousers"` → the named subset.
   - Anything not in the outfit raises `ValueError`.
3. Concatenates the requested garments into one combined `(V, F, C)`
   while remembering, in `vert_gid`, which garment each vertex belongs
   to. Faces are offset so triangle indices stay valid in the combined
   array.

Then it loads `n_body_frames` SMPL poses via `reader.read_human`,
yielding a `(T, 6890, 3)` body sequence. With `T = 1` the body is a
static collider; with `T > 1` the viewer cycles through frames each
step (`cloth.set_body(body_V_seq[i % T])`).

## Multi-garment merge: why it works without seams

CLOTH3D garments are **separate meshes** with no shared vertices and no
explicit attachment between them. We never merge geometry — we just
**concatenate** vertex and face arrays:

```
V_combined = [V_Tshirt;  V_Trousers]
F_combined = [F_Tshirt;  F_Trousers + N_Tshirt]
vert_gid   = [0,0,…,0;   1,1,…,1]
```

Edges built on `F_combined` therefore never cross garments — every
constraint stays inside the garment it came from. So when
`XPBDCloth.__init__` looks up `dist_compliance[e]` from
`vert_gid[edges[:,0]]`, it picks up the correct fabric for that edge
without any extra bookkeeping.

The two garments interact through:
- The body collider (both are pushed off the same SMPL surface).
- Implicitly through gravity and timing — they are simulated
  simultaneously each substep.

There is **no cloth-cloth collision** in this demo. A Tshirt hem
through Trousers is geometrically possible. Adding cloth-cloth pushout
would be a future extension (most likely a hash grid over all cloth
vertices).

## What `load_sample` returns

```python
{
    "V0":              (N, 3) float32,    # initial particle positions
    "F":               (M, 3) int32,      # combined triangle list
    "C":               (N, 3) float32,    # per-vertex colors in [0, 1]
    "vert_gid":        (N,)   int32,      # garment index per vertex
    "garment_names":   list[str],         # in the merge order
    "garment_fabrics": list[str],         # one per garment, e.g. "cotton"
    "body_V_seq":      (T, 6890, 3),      # SMPL frames
    "body_F":          (NBF, 3),          # SMPL faces (constant)
    "sample":          str,               # echoed back
}
```

The solver constructor consumes this dict directly. No other
preprocessing is required.

## Coordinate system

CLOTH3D is **z-up, meters** (typical body height ≈1.7–1.9 m). Gravity
in the solver is `(0, 0, -9.81)`. The viewer cameras and matplotlib
axes are oriented for z-up.

`info["zrot"]` carries a per-sequence world Z-axis rotation that the
toolkit's animated readers already apply, so cloth and body live in the
same world frame.
