# Fabric presets

The simulator chooses XPBD parameters per-garment from a single dict in
`xpbd/fabrics.py`:

```python
FABRIC_PRESETS = {
    "cotton":  dict(distance_compliance=5.0e-9,  bend_compliance=1.0e-5, damping=0.03, density=0.30),
    "silk":    dict(distance_compliance=2.0e-8,  bend_compliance=5.0e-7, damping=0.01, density=0.10),
    "denim":   dict(distance_compliance=1.0e-9,  bend_compliance=5.0e-5, damping=0.05, density=0.45),
    "leather": dict(distance_compliance=5.0e-10, bend_compliance=2.0e-4, damping=0.08, density=0.80),
}
```

These four materials match the CLOTH3D fabric tags. A garment's tag is
read from `info["outfit"][garment]["fabric"]` by the data pipeline.

## What each parameter does

| Parameter | Units | Effect |
|---|---|---|
| `distance_compliance` | m²/N | In-plane stretch resistance. Smaller → stiffer (less stretch). |
| `bend_compliance`     | m²/N | Resistance to bending (PBD opposite-vertex distance). Smaller → stiffer fold lines, less drape. |
| `damping`             | dimensionless, [0, 1) | Per-substep velocity damping in `predict()`. Larger → cloth feels "wet" / dead. |
| `density`             | kg/m² | Areal density. Drives per-vertex mass via `compute_vertex_masses`. |

How they enter the solver: see `docs/xpbd_method.md`. In short,
`solve_distance` and `solve_bending` use `α̃ = α / Δt²` inside the XPBD
correction, and `predict` multiplies velocity by `(1 − damping)` each
substep. Mass enters as inverse mass `wᵢ = 1 / mᵢ` in every constraint
correction.

## Why these specific numbers

The values are hand-tuned to give visually-plausible behaviour at the
default `--substeps 10 --iters 5` budget, with cotton as the anchor
(because it is the C-IPC comparison material).

Qualitatively:

- **cotton** — woven, medium thickness. Moderately stiff in stretch
  (clothes don't visibly stretch when you wear them, but they're not
  tarpaulins). Bends easily but holds creases. Density ~0.3 kg/m² is in
  the range for a t-shirt-weight cotton.
- **silk** — thin, very limp. ~10× softer in stretch *and* much softer in
  bending than cotton (silks famously drape tight to the body). Low
  damping, low density.
- **denim** — heavy, stiff in both stretch and bending. ~5× stiffer in
  stretch than cotton, much stiffer in bending so creases stand up.
  Higher density.
- **leather** — the stiffest of the four; both compliances dropped and
  density doubled. Damping is high so the cloth "settles" rather than
  oscillating.

These aren't physical measurements — XPBD compliance doesn't have a
clean correspondence to Young's modulus for arbitrary mesh resolutions
and constraint formulations. They are *consistent* values that produce
recognisable behaviour and an informative spread between materials.

If you need physically meaningful compliance from a Young's modulus E
(Pa) and rest length L₀ (m): for a single 1-D spring, stiffness
`k ≈ E · A / L₀` (A = cross-section area), and `α = 1 / k`. For a
triangulated cloth this is approximate at best — calibration is usually
empirical.

## Tuning recipe

When in doubt, start from cotton and move one knob at a time:

1. Run with the default fabric. Watch one substep.
2. If the cloth visibly stretches (looks like rubber), reduce
   `distance_compliance` by 5×. Re-run.
3. If it shimmers / oscillates, raise `damping` to 0.05 and add
   substeps.
4. If it drapes too floppy / phases through itself, reduce
   `bend_compliance`.
5. If it's stable but feels weightless, raise `density`.

Stiffer constraints need *either* lower compliance *or* more iterations.
If lowering compliance makes the cloth jitter, the iteration count
isn't keeping up.

## CLI overrides

Three flags override every garment's preset for one run:

| Flag | Effect |
|---|---|
| `--dist_compliance V` | every edge uses V regardless of fabric |
| `--bend_compliance V` | every bending pair uses V |
| `--damping V` | every vertex uses V |

`--force_fabric cotton` is the heavier hammer — it replaces every
garment's fabric *string* (so density also changes) before the solver
broadcasts presets.

## Adding a new fabric

1. Add an entry to `FABRIC_PRESETS` in `xpbd/fabrics.py`.
2. That's it. The solver looks materials up by name; nothing else
   needs to change.
3. Optionally, add a smoke test that loads a sample and forces the new
   fabric via `garment_fabrics=[...]` to the `XPBDCloth` constructor.

`fabric_params(name)` already returns cotton when the name is unknown,
so even a typo in `info.mat` won't crash the loader.
