# Comparing XPBD against C-IPC (cotton baseline)

The C-IPC pipeline used in this project currently drives every garment
with a single material — cotton. To produce a like-for-like XPBD run,
**force the same fabric on every garment** (so XPBD doesn't accidentally
make the silk Tshirt limp while C-IPC treats it as cotton).

## Recommended XPBD command

```bash
python3 xpbd_cloth.py --sample 07414 \
    --garments all \
    --force_fabric cotton \
    --body_frames 90 \
    --substeps 10 --iters 5 \
    --viewer mpl --save_video
```

Why these flags:

- `--garments all` — simulate every garment in the outfit, so the
  collider interaction matches what C-IPC sees.
- `--force_fabric cotton` — overrides every garment's CLOTH3D fabric
  tag with cotton, matching the C-IPC material setup. Without this,
  sample `00016` (Trousers cotton + Tshirt silk) and sample `01691`
  (Trousers leather + Tshirt silk) would diverge from the baseline at
  the per-garment level.
- `--body_frames 90` — match the length of the C-IPC clip; the body
  collider then animates over the same SMPL frames.
- `--substeps 10 --iters 5` — defaults; raise both proportionally if
  the cotton looks too floppy.
- `--viewer mpl --save_video` — produces an mp4 you can place
  side-by-side with the C-IPC clip.

## Recommended XPBD command (numerical comparison)

For evaluation rather than visual comparison, dump frames as `.npy`:

```bash
python3 xpbd_cloth.py --sample 07414 \
    --garments all --force_fabric cotton --body_frames 90 \
    --viewer none --steps 90 --save_every 1
```

This writes one `.npy` per frame to `xpbd_out/`. Each file is the cloth
vertex array `(N, 3)` corresponding to the merged garments in
`load_sample` order (see `data["garment_names"]` for the order). Pair
each XPBD frame with the matching C-IPC mesh and compute your metric
(e.g. vertex-to-surface distance — `cloth3d/` provides tooling).

## What is *not* matched

Even with `--force_fabric cotton`, XPBD and C-IPC differ in:

- **Constraint formulation** — XPBD uses spring-style distance
  constraints + the PBD opposite-vertex bending shortcut. C-IPC uses an
  IPC barrier-augmented FEM shell. Stress responses differ.
- **Self-collision** — XPBD here has *no* cloth-cloth collisions (only
  body collisions). C-IPC does cloth-cloth.
- **Collision response** — XPBD pushes each cloth vertex along the
  *nearest body vertex's* normal. C-IPC uses an IPC barrier on the
  signed distance.
- **Time stepping** — XPBD with substeps is an explicit projection
  scheme; C-IPC uses an implicit Newton solve.

These differences are the point of the comparison: not "do they
agree", but "where do they disagree, and by how much". Expect XPBD to
look slightly stretchier on tight regions and to phase through itself
near layered garments where C-IPC's self-collision saves it.

## Quick checklist

1. Same sample.
2. `--force_fabric cotton` on the XPBD side (or no force flag if
   C-IPC is also using each garment's true fabric).
3. Same number of body frames.
4. Same frame indices when comparing — `00016_*.npy` from
   `--save_every 1` aligns 1-to-1 with C-IPC frames.
5. Compare `data["garment_names"]` order: XPBD merges in CLOTH3D
   `info["outfit"]` iteration order. Make sure the C-IPC export uses
   the same order, or re-index before computing distances.
