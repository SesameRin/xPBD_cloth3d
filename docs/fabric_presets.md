# Fabric presets

The simulator chooses XPBD parameters per-garment from a single dict in
`xpbd/fabrics.py`:

```python
FABRIC_PRESETS = {
    "cotton":  dict(distance_compliance=5.0e-9,  bend_compliance=1.0e-5,  damping=0.03, density=0.30),
    "silk":    dict(distance_compliance=2.0e-8,  bend_compliance=3.0e-5,  damping=0.01, density=0.10),
    "denim":   dict(distance_compliance=2.5e-9,  bend_compliance=3.75e-6, damping=0.05, density=0.70),
    "leather": dict(distance_compliance=1.2e-9,  bend_compliance=1.9e-6,  damping=0.08, density=1.25),
}
```

These four materials match the CLOTH3D fabric tags. A garment's tag is
read from `info["outfit"][garment]["fabric"]` by the data pipeline.

The current values are an attempt to keep XPBD and the C-IPC shell
solver in `cloth3d-ipc-xpbd/cloth3d_benchmark/cloth3d_sim/materials.py`
broadly comparable. See *Alignment with C-IPC* below for what was
matched, what wasn't, and why.

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

## Alignment with C-IPC

The reference C-IPC presets use a different parameterisation entirely
(Young's modulus E, Poisson ν, volumetric density ρ_v, thickness h,
bending stiffness B). Aligning was done axis-by-axis, anchored on
cotton, with the goal: **same ordering across all four fabrics, and
relative spreads within ~1.5× of each other**. Exact agreement is not
achievable — see *Why they can't fully align* below.

### Mass — direct identity (clean)

Areal mass `ρ_a [kg/m²] = ρ_v [kg/m³] · h [m]`. xPBD's `density` was
set to exactly this product from the C-IPC table:

| Fabric | ρ_v (C-IPC) | h (C-IPC) | ρ_v · h | xPBD `density` |
|---|---|---|---|---|
| silk    | 120 | 8.0e-4 | 0.096 | 0.10 |
| cotton  | 200 | 1.5e-3 | 0.30  | 0.30 |
| denim   | 350 | 2.0e-3 | 0.70  | 0.70 |
| leather | 500 | 2.5e-3 | 1.25  | 1.25 |

This is the only parameter where the two solvers can agree analytically.

### Stretch — relative-spread match

Both solvers should rank fabrics the same way and stretch by similar
relative amounts under the same load. Using `1/α_d` for xPBD and `E·h`
for C-IPC as proxy stiffnesses, and anchoring cotton at 1.0:

| Fabric | C-IPC `E·h` ratio | xPBD target ratio | xPBD `α_d` |
|---|---|---|---|
| silk    | 0.27 | 0.27 | 2.0e-8 |
| cotton  | 1.00 | 1.00 | 5.0e-9 |
| denim   | 2.00 | 2.00 | 2.5e-9 |
| leather | 4.17 | 4.17 | 1.2e-9 |

Old preset had leather at 10× cotton — 2.5× stiffer than the C-IPC
table implies. Now compressed to match.

### Bending — sign was wrong, now corrected

The previous presets had silk's `bend_compliance` *smaller* than
cotton's — the opposite of physical intent and the opposite of C-IPC's
ordering. The docstring in this file even claimed "α much smaller for
stiff fabrics (denim) and much larger for floppy ones (silk)" while the
numbers said the reverse. That has been fixed.

Anchoring cotton at 1.0 again, using `1/α_b` for xPBD and C-IPC's
`bending_stiffness` directly:

| Fabric | C-IPC `B` ratio | xPBD target ratio | xPBD `α_b` |
|---|---|---|---|
| silk    | 0.33 | 0.33 | 3.0e-5  |
| cotton  | 1.00 | 1.00 | 1.0e-5  |
| denim   | 2.67 | 2.67 | 3.75e-6 |
| leather | 5.33 | 5.33 | 1.9e-6  |

### Damping — not aligned

C-IPC has no per-fabric damping knob (its dissipation comes from the
implicit Newton solve and the IPC barrier viscosity). xPBD's `damping`
is a per-substep velocity decay with no continuum analogue. The values
here are the original visual-tuning ones, kept only to give each fabric
a slightly different "settling" feel. **Don't try to compare damping
between solvers.**

### Poisson — not representable

C-IPC carries a per-fabric Poisson ratio ν ∈ [0.30, 0.40] that couples
stretches in different in-plane directions. xPBD's per-edge distance
constraint has no shear / cross-coupling term, so ν is silently dropped.
Expect xPBD cloth to look slightly more "rubber-bandy" under biaxial
load.

## Why they can't fully align

The two solvers solve genuinely different equations of motion. Aligning
parameter tables is at best a "match the macroscopic stiffness order
and rough magnitudes" exercise; expecting them to produce
visually-identical drape would be a mistake.

1. **Different governing physics.** xPBD is iterative position
   projection on per-edge constraints; C-IPC is implicit Newton on a
   continuous Baraff–Witkin shell with an IPC barrier. The same
   "stiffness" number maps to different force responses and converges
   to different equilibria.
2. **Compliance is mesh-dependent.** xPBD `α_d` is per-edge — putting
   more edges in parallel makes a region effectively stiffer even if α
   is unchanged. C-IPC's `E·h` is a continuum quantity, so doubling the
   mesh resolution leaves macroscopic stiffness invariant (modulo
   discretization error). Any calibration here is therefore valid only
   at one mesh resolution.
3. **Bending models are not the same shape function.** xPBD uses the
   PBD "opposite-vertex distance" shortcut — a length constraint
   between the two third-vertices of the two triangles sharing an
   interior edge. C-IPC uses a discrete shell bending energy, which is
   moment-based on the dihedral angle. Even with matched stiffness
   ratios, the *response curve* under deflection differs, especially at
   large bending angles.
4. **No Poisson coupling in xPBD.** As above, ν is dropped. Biaxial
   stretches and shear behave differently.
5. **Damping models are incompatible.** Explicit per-substep velocity
   decay vs. implicit barrier viscosity. No conversion exists.
6. **Iteration / substep dependence.** XPBD reduces (but does not
   eliminate) PBD's iteration-count-dependent stiffness for very small
   compliance values. C-IPC converges to the implicit solution. So at
   low iteration budgets xPBD's effective stiffness is somewhat lower
   than the table suggests.
7. **Collisions differ.** xPBD here does nearest-body-vertex pushout
   and *no self-collision*. C-IPC uses an IPC barrier on the signed
   distance and includes self-collision. Layered or tight regions will
   look very different — this is often misread as a stiffness mismatch
   when it's really a collision-model mismatch.
8. **Strain limiting.** C-IPC's `StrainLimitingBaraffWitkinShell`
   truncates large stretches non-linearly. xPBD has no equivalent, and
   relies entirely on stiff distance constraints. Under heavy load the
   two solvers' stretch behaviours can diverge regardless of α_d.

In short: the table above gets the *ordering* right and the *order of
magnitude* defensible. It will not get the per-vertex deformation
right.

## Possible future work

In rough order of effort vs. payoff:

1. **Empirical calibration harness.** Two canonical scenes per fabric
   — a sheet hung from one edge under gravity, and a cantilever strip
   — run both solvers at the project's default mesh resolution, then
   line-search xPBD's `α_d` and `α_b` to minimise per-vertex L2 against
   the C-IPC reference. This replaces the algebraic ratio matching
   above with a numerical fit and is the single biggest accuracy win.
2. **Re-fit per mesh resolution.** Because xPBD compliance is
   mesh-dependent, the calibration in (1) only holds for one
   resolution. Either fix the resolution project-wide or store a
   per-resolution preset table keyed on average edge length.
3. **Single source-of-truth YAML.** Move the C-IPC table and the xPBD
   table into a shared `fabrics.yaml` with `young / poisson /
   density_volumetric / thickness / bending_stiffness` as canonical
   fields. Both solvers' loaders derive their internal parameters from
   it (xPBD via the calibration map). Eliminates the silent drift that
   produced the inverted-bending bug in the first place.
4. **Per-fabric `collision_radius` from thickness.** Set
   `collision_radius ≈ thickness / 2` per garment so the cloth-body
   offset matches C-IPC's contact gap. Currently xPBD uses a single
   global value.
5. **Better bending in xPBD.** Replace the opposite-vertex-distance
   shortcut with a true dihedral-angle bending constraint (Bender et al.
   2014, §4). This makes `bend_compliance` directly comparable to
   C-IPC's moment stiffness up to a constant factor — the response
   curves still differ at large angles, but the parameter mapping
   becomes much cleaner.
6. **Add cloth–cloth collision to xPBD.** Either spatial-hash repulsion
   for cloth particles or a CCD-based correction. Not strictly a
   material-alignment item, but it removes the largest visible
   difference between the two pipelines at multi-garment scenes.

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
broadcasts presets. This is the recommended flag for like-for-like
C-IPC comparison runs (see `docs/cipc_comparison.md`).

## Adding a new fabric

1. Add an entry to `FABRIC_PRESETS` in `xpbd/fabrics.py`.
2. If the same fabric exists in the C-IPC preset table, use the
   alignment recipe above (mass identity, stretch ratio, bending
   ratio).
3. Optionally, add a smoke test that loads a sample and forces the new
   fabric via `garment_fabrics=[...]` to the `XPBDCloth` constructor.

`fabric_params(name)` already returns cotton when the name is unknown,
so even a typo in `info.mat` won't crash the loader.
