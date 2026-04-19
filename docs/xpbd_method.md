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
predict(sub_dt)                           # advance v + tentative p
reset_lambdas()                           # λ ← 0 for every constraint
for _ in range(iterations):
    solve_distance(sub_dt)                # in-plane stretch
    solve_bending(sub_dt)                 # dihedral bending shortcut
    solve_collision(radius)               # non-penetration pushout only
apply_contact_friction(1/substeps, μ, h)  # once per substep, not per iter
finalize(sub_dt)                          # derive v, write back positions
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

The kernel accumulates Δpᵢ and Δpⱼ into a separate per-vertex buffer
(`self.dp`) with atomic adds, and a follow-up `apply_dp` kernel flushes
`p ← p + dp` in a race-free one-thread-per-vertex pass. Each
contribution is additionally scaled by `1 / vertex_valence` so that the
sum of all parallel updates matches the magnitude a single Gauss-Seidel
step would apply. This is required for GPU correctness: naive in-place
`p[i] += …` from a parallel edge loop causes every stiff fabric to
diverge to NaN in one sub-step. See [`gpu_performance.md`](gpu_performance.md).

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

### 5a. Non-penetration — `solve_collision`

Called *inside the iteration loop* (5 iters × 10 substeps = 50× per frame).

```
for each cloth particle pᵢ:
    j*   = argmin_j ‖pᵢ − bⱼ‖²
    n̂    = body_normal[j*]
    s    = (pᵢ − b[j*]) · n̂                 # signed distance to surface
    if s < radius:
        push = min(radius − s, 2·radius)    # clamp pushout per call
        pᵢ  ← pᵢ + push · n̂
```

Pushout is clamped to `2·radius` per call so a deeply penetrating
vertex (CLOTH3D data isn't intersection-free at frame 0) crawls out
over several iterations instead of teleporting by 5–10 cm in one
shot. Teleporting propagates through the stiff distance constraints
and blows up neighbors.

### 5b. Friction — `apply_contact_friction`

Called *once per substep* (10× per frame), **not** per iteration.

```
if capture > 0 and μ > 0:
  for each cloth particle pᵢ:
      j*   = argmin_j ‖pᵢ − bⱼ‖²
      n̂    = body_normal[j*]
      s    = (pᵢ − b[j*]) · n̂
      if 0 ≤ s < capture:                   # contact AND non-penetrating
          w      = 1 − s / capture           # 1 at skin, 0 at shell edge
          b_vel  = b[j*]_now − b[j*]_prev     # body motion this frame
          tang   = b_vel − (b_vel · n̂) n̂
          pᵢ    ← pᵢ + μ · w · (1/substeps) · tang
```

Two things are doing work here.

**(1) Once per substep with `1/substeps` scaling.** If friction were
applied inside the iteration loop, it would add `μ · b_vel_tang` fifty
times per frame — the same frame-level body motion re-used at every
call — and stable cloth would drift by `50 · μ · b_vel_tang` per
frame. Running once per substep with `bv_scale = 1/substeps` makes
the per-frame sum equal exactly `μ · b_vel_tang`, which is the
physically intended amount.

**(2) Penetrator gate `s ≥ 0`.** For cloth wedged inside the body,
`argmin` can snap to a body vertex on the *far* side of a limb — its
normal faces the wrong way and its body-velocity direction is
unrelated to the correct tangent. Applying friction there drives the
vertex tangentially into nonsense. Instead we skip friction while
`s < 0` and rely on the pushout in `solve_collision` to recover the
vertex; once `s ≥ 0` on a later substep friction re-engages.

This is O(N · NBV) every call — one cloth vertex vs. every body
vertex. Fine for ≈10k cloth verts × 6890 SMPL verts at demo scale; a
real system would use a hash grid or BVH.

Body vertex normals are recomputed any time the body pose changes via
`set_body(body_V)`, which also snapshots the previous body positions
into `body_x_prev` so friction can read `b_vel` for the frame
currently being solved.

**Why friction at all?** Non-penetration only stops cloth from going
*into* the body; nothing connects it tangentially. When the body
translates or rotates, inertial cloth stays put and the body slides
out from under the garment, so the hem gradually falls off the legs.
Position-based friction couples cloth to the body's tangential motion
inside a thin "capture shell"; strength falls linearly from
`μ · b_vel_tang` at zero gap to zero at the shell edge. This is the
PBD variant of the standard contact Coulomb-friction model but
formulated on positions rather than impulses, so it fits cleanly into
the XPBD constraint-projection sweep. See `--friction` /
`--friction_capture` / `--collision_radius` in the top-level README
for tuning.

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
