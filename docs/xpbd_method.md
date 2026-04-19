# XPBD method

This is the math behind `xpbd/solver.py`, with line-of-code anchors for
each formula.

## Why XPBD (vs PBD or implicit FEM)

Classical PBD (Müller et al. 2007) projects particle positions to satisfy
constraints. It's fast and unconditionally stable, but its effective
stiffness depends on the iteration count and time step — change either
and your cloth changes too. Stiffer materials need many iterations.

XPBD (Macklin, Müller, Chentanez 2016) fixes this by formulating each
constraint as an **elastic energy** with a compliance parameter α
(reciprocal of stiffness). The Lagrange multiplier λ for the constraint
is solved iteratively with a per-substep correction, and α is
*time-step-independent* once you divide by Δt². You get stable behaviour
that doesn't drift with iteration count, with only one extra scalar per
constraint.

## Time-step structure

`XPBDCloth.step()` runs `substeps` outer iterations of length
`sub_dt = dt / substeps`. Each substep is:

```
predict(sub_dt)             # advance velocities + tentative positions
reset_lambdas()             # λ ← 0 for every constraint
for _ in range(iterations):
    solve_distance(sub_dt)
    solve_bending(sub_dt)
    solve_collision(radius)
finalize(sub_dt)            # write back positions, derive velocities
```

Smaller `sub_dt` (more `substeps`) → better accuracy and stability,
roughly linear cost.

## 1. Predict

`xpbd/solver.py: predict()`

For each particle `i` with inverse mass `wᵢ > 0`:

```
vᵢ ← (1 − damping_i) · vᵢ + g · Δt
pᵢ ← xᵢ + vᵢ · Δt
```

`pᵢ` is the *predicted* position that the constraint solve will project
back onto the constraint manifold. Damping is per-vertex because each
garment carries its own value (cotton has more internal friction than
silk).

Pinned vertices (we don't pin any in this demo, but the field exists)
just set `pᵢ ← xᵢ` if `wᵢ == 0`.

## 2. Reset Lagrange multipliers

`xpbd/solver.py: reset_lambdas()`

XPBD's correction formula uses an **accumulated** λ across the inner
iterations of one substep, but λ is fresh each substep:

```
λ_d[e] ← 0    for every distance edge
λ_b[b] ← 0    for every bending pair
```

## 3. Distance constraints (in-plane stretch)

`xpbd/solver.py: solve_distance()`

For an edge `e = (i, j)` with rest length `L₀`, the constraint is
`C(p) = ‖pᵢ − pⱼ‖ − L₀ = 0`. Its gradient with respect to `pᵢ` is
`n̂ = (pᵢ − pⱼ) / L`, and `−n̂` for `pⱼ`. With compliance α (= the
fabric's `distance_compliance`):

```
α̃     = α / Δt²
ΔλC = ( −C − α̃ · λ ) / ( wᵢ + wⱼ + α̃ )
λ    ← λ + ΔλC
Δpᵢ = +wᵢ · ΔλC · n̂
Δpⱼ = −wⱼ · ΔλC · n̂
```

Why `α / Δt²`? In the XPBD derivation, α has units of compliance
(length²/force), and the discrete Lagrangian gives the system
`(∇C · M⁻¹ · ∇Cᵀ + α/Δt²) Δλ = −C − (α/Δt²) λ`. The shape `(α/Δt²)`
falls out of the implicit-Euler step on the elastic potential — see
Macklin et al. 2016, §3 for the derivation.

Effect of α:
- α = 0 → infinitely stiff (classical PBD constraint projection).
- larger α → softer; the constraint is allowed to be partially violated.
- in this code each *edge* reads its own α from `dist_compliance[e]`,
  set per-garment by `XPBDCloth.__init__` and `xpbd/fabrics.py`.

The kernel writes `pᵢ` and `pⱼ` with Taichi's default auto-atomic `+=`,
so no write is ever lost. The remaining racy part is the **read** of
`pᵢ` / `pⱼ` at the top of the loop body: concurrent threads computing a
constraint that shares a vertex may observe a stale position. On CPU
(≤ ~8 concurrent threads) this is rare and benign; PBD/XPBD is iterative
and the transient error washes out.

On GPU, though, every edge is in flight simultaneously, so a vertex
touched by N edges receives N corrections computed from the *same* stale
`p[i]`. For stiff materials (silk: α ≈ 2·10⁻⁸) the summed over-correction
is large enough to oscillate and diverge within one frame, sending the
cloth to NaN — that's why `--arch gpu` used to "lose" the garments.

**Fix (GPU-safe mode):** greedy **graph coloring** of distance edges and
bending pairs (`xpbd/geometry.py: greedy_pair_coloring`). Within one
color class no two constraints share a vertex, so parallel threads write
to disjoint entries of `p` — zero races. The solver launches one kernel
per color; across colors the loop is sequential, so the overall scheme
is still Gauss–Seidel (just in a specific order rather than the
arbitrary edge-list order CPU uses). This preserves the XPBD math and
convergence target: compare a CPU run vs a GPU run frame-by-frame and
you see a few mm of transient drift that doesn't diverge.

Typical cloth meshes here color into ~10–12 classes, which with
5 iterations × 10 substeps is ≈ 500 kernel launches per frame — fine on
a modern GPU. Enabled automatically by `--arch gpu` / `--arch vulkan`;
CPU (`--arch cpu`) keeps the original single-launch kernel untouched.

## 4. Bending constraints

`xpbd/solver.py: solve_bending()`

We use the classic PBD bending shortcut: for every internal mesh edge
shared by two triangles, identify the **two opposite vertices** `v₃, v₄`
(the ones that are *not* on the shared edge) and apply a distance
constraint between them with rest length equal to their initial
separation. When the dihedral angle changes, that distance changes, so
this proxies bending.

Mathematically the constraint is identical to the stretch one — same
derivation, same kernel structure — but with `α = bend_compliance`,
which is much smaller for stiff fabrics (denim) and much larger for
floppy ones (silk).

This is cheaper and easier to implement than the original cosine-of-
dihedral-angle formulation, and is well-documented in Müller et al.
2007 §4.4 and the Bender–Müller–Macklin survey ("A Survey on Position-
Based Simulation Methods", 2014).

The bending pairs themselves are precomputed once by
`xpbd/geometry.py: build_bending_pairs(F)` — for each shared edge, look
up the two adjacent triangles and pick the third vertex of each.

## 5. Collision (body pushout)

`xpbd/solver.py: solve_collision()`

Per cloth particle, find the **nearest body vertex** and push outside a
sphere of `collision_radius` centred on that body vertex, along the
body's outward normal. Concretely:

```
for each cloth particle pᵢ:
    j*   = argmin_j ‖pᵢ − bⱼ‖²
    n̂    = body_normal[j*]
    s    = (pᵢ − b[j*]) · n̂                 # signed distance to surface
    if s < radius:
        pᵢ ← pᵢ + (radius − s) · n̂
```

This is O(N · NBV) every iteration. Fine for ≈10k cloth verts × 6890
SMPL verts at demo scale; a real system would use a hash grid or BVH.

Body vertex normals are recomputed any time the body pose changes via
`set_body(body_V)` calling `geometry.per_vertex_normals`.

## 6. Finalize

`xpbd/solver.py: finalize()`

Once the inner constraint loop is done, derive the corrected velocity
from the change in position:

```
vᵢ ← (pᵢ − xᵢ) / Δt
xᵢ ← pᵢ
```

This is the standard Verlet-style velocity update used by all PBD/XPBD
schemes.

## Mass model

`xpbd/geometry.py: compute_vertex_masses(V, F, vert_gid, fabrics)`

Per-vertex mass is one third of the area of each adjacent triangle times
the *owning garment's* areal density (kg/m²). Densities live in
`FABRIC_PRESETS`. Inverse mass `wᵢ = 1 / mᵢ` is stored in
`XPBDCloth.w`. This way:

- a denser garment (denim) has more inertia per unit area;
- a fine-mesh region naturally has lower per-vertex mass than a coarse
  one with the same density (because each vertex owns less area);
- the constraint correction `wᵢ · Δp` falls out without any extra
  reasoning — heavier verts move less for a given Δλ.

## Picking parameters

For each fabric, three knobs matter most:

| What you want | Knob | Direction |
|---|---|---|
| stiffer stretch | `distance_compliance` | smaller |
| stiffer bend / less drape | `bend_compliance` | smaller |
| more "wet" / dead behaviour | `damping` | larger |

Compliance values that are too small (< 1e-11) plus too few iterations
will look correct one frame and explode the next. The defaults in
`xpbd/fabrics.py` are tuned to be safe with `--substeps 10 --iters 5`.
If you raise stiffness, raise iteration count too.

## References

- Macklin, Müller, Chentanez. *XPBD: Position-Based Simulation of
  Compliant Constrained Dynamics.* MIG 2016.
- Müller, Heidelberger, Hennix, Ratcliff. *Position based dynamics.*
  J. Vis. Commun. Image Represent. 2007.
- Bender, Müller, Macklin. *A Survey on Position-Based Simulation
  Methods in Computer Graphics.* CGF 2014.
