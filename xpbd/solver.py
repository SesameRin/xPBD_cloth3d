"""XPBD cloth solver in Taichi.

Implements the standard XPBD step:

    predict   -> reset λ -> [solve constraints] × iterations -> finalize

Constraints used here:
    1. Distance (in-plane stretch) per mesh edge.
    2. Bending: classic PBD shortcut — distance between the two opposite
       vertices of each shared-edge triangle pair.
    3. Body collision: nearest-body-vertex pushout along that vertex's
       outward normal.

Each per-edge / per-bend pair carries its own compliance (so a multi-fabric
outfit just works), and per-vertex damping comes from the owning garment.
See `xpbd.fabrics` for the parameter table and `docs/xpbd_method.md` for
the math.

Iteration scheme: Jacobi-style. Distance and bending constraints accumulate
corrections into a separate `dp` buffer with atomic adds, then a follow-up
kernel flushes `dp` into `p`. This is required for GPU correctness — with
thousands of concurrent threads, the naive "write p in place" has a
read-before-write race that explodes for stiff materials (silk dist=2e-8).
CPU (8 threads) mostly gets away with it; CUDA does not.
"""

import time

import numpy as np
import taichi as ti

from .fabrics import DEFAULT_FABRIC, fabric_params
from .geometry import (
    build_edges,
    build_bending_pairs,
    compute_vertex_masses,
    per_vertex_normals,
)


@ti.data_oriented
class XPBDCloth:
    def __init__(
        self,
        V0,
        F,
        body_V0,
        body_F,
        vert_gid=None,
        garment_fabrics=None,
        dt=1.0 / 60.0,
        substeps=10,
        iterations=5,
        dist_compliance_override=None,
        bend_compliance_override=None,
        damping_override=None,
        gravity=(0.0, 0.0, -9.81),
        collision_radius=0.01,
    ):
        self.N = V0.shape[0]
        self.NF = F.shape[0]
        self.NBV = body_V0.shape[0]
        self.NBF = body_F.shape[0]

        if vert_gid is None:
            vert_gid = np.zeros(self.N, dtype=np.int32)
        if garment_fabrics is None or len(garment_fabrics) == 0:
            garment_fabrics = [DEFAULT_FABRIC]
        self.vert_gid = vert_gid.astype(np.int32)
        self.fabrics = list(garment_fabrics)

        E = build_edges(F)
        BP = build_bending_pairs(F)
        self.NE = E.shape[0]
        self.NB = BP.shape[0]

        # Per-vertex constraint valence — used to under-relax Jacobi updates
        # so parallel constraint solves (GPU) don't diverge on stiff materials.
        edge_val = np.bincount(E.flatten(), minlength=self.N).astype(np.float32)
        if self.NB > 0:
            bend_val = np.bincount(
                BP[:, 2:4].flatten(), minlength=self.N
            ).astype(np.float32)
        else:
            bend_val = np.zeros(self.N, dtype=np.float32)
        self.inv_edge_val = np.where(edge_val > 0, 1.0 / edge_val, 0.0).astype(
            np.float32
        )
        self.inv_bend_val = np.where(bend_val > 0, 1.0 / bend_val, 0.0).astype(
            np.float32
        )

        # Per-garment compliance / damping, with optional global overrides.
        per_g = [fabric_params(f) for f in self.fabrics]
        g_dist = np.array(
            [dist_compliance_override if dist_compliance_override is not None
             else p["distance_compliance"] for p in per_g],
            dtype=np.float32,
        )
        g_bend = np.array(
            [bend_compliance_override if bend_compliance_override is not None
             else p["bend_compliance"] for p in per_g],
            dtype=np.float32,
        )
        g_damp = np.array(
            [damping_override if damping_override is not None else p["damping"]
             for p in per_g],
            dtype=np.float32,
        )

        # Map per-edge / per-bend compliance from the garment of an endpoint.
        edge_gid = self.vert_gid[E[:, 0]]
        dist_compliance_arr = g_dist[edge_gid]
        bend_gid = (
            self.vert_gid[BP[:, 2]] if self.NB > 0
            else np.zeros((0,), dtype=np.int32)
        )
        bend_compliance_arr = (
            g_bend[bend_gid] if self.NB > 0
            else np.zeros((0,), dtype=np.float32)
        )
        vert_damping = g_damp[self.vert_gid]

        # Per-vertex mass from areal density.
        mass = compute_vertex_masses(V0, F, self.vert_gid, self.fabrics)
        w = (1.0 / mass).astype(np.float32)

        print(
            f"[solver] N={self.N} NF={self.NF} NE={self.NE} NB={self.NB} "
            f"body_V={self.NBV} body_F={self.NBF} garments={len(self.fabrics)}"
        )
        for i, f in enumerate(self.fabrics):
            print(
                f"         garment[{i}] fabric={f} "
                f"dist={g_dist[i]:.2e} bend={g_bend[i]:.2e} damp={g_damp[i]:.2f}"
            )

        self.dt = dt
        self.substeps = substeps
        self.iterations = iterations
        self.gravity = ti.Vector(list(gravity))
        self.collision_radius = collision_radius

        # particles
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.p = ti.Vector.field(3, ti.f32, self.N)
        self.dp = ti.Vector.field(3, ti.f32, self.N)  # Jacobi delta buffer
        self.w = ti.field(ti.f32, self.N)
        self.damp = ti.field(ti.f32, self.N)
        # Per-vertex Jacobi scaling. Vertex i is touched by inv_edge_val[i]^-1
        # distance edges and inv_bend_val[i]^-1 bending pairs; dividing each
        # contribution keeps parallel accumulation stable for stiff fabrics.
        self.inv_edge_val_fld = ti.field(ti.f32, self.N)
        self.inv_bend_val_fld = ti.field(ti.f32, self.N)

        # distance constraints
        self.edges = ti.Vector.field(2, ti.i32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)
        self.lambda_d = ti.field(ti.f32, self.NE)
        self.dist_compliance = ti.field(ti.f32, self.NE)

        # bending constraints (use shape max(1, NB) so taichi never sees 0)
        nb = max(1, self.NB)
        self.bend_idx = ti.Vector.field(2, ti.i32, nb)
        self.bend_rest = ti.field(ti.f32, nb)
        self.lambda_b = ti.field(ti.f32, nb)
        self.bend_compliance = ti.field(ti.f32, nb)

        # body collider
        self.body_x = ti.Vector.field(3, ti.f32, self.NBV)
        self.body_n = ti.Vector.field(3, ti.f32, self.NBV)

        # rendering
        self.face_idx = ti.field(ti.i32, self.NF * 3)
        self.body_face_idx = ti.field(ti.i32, self.NBF * 3)
        self.color = ti.Vector.field(3, ti.f32, self.N)

        self._load_static(
            V0, F, E, BP, body_V0, body_F,
            w, vert_damping, dist_compliance_arr, bend_compliance_arr,
        )
        self.set_body(body_V0)
        self.reset_timing()

    # ------------------------------------------------------------------ setup
    def _load_static(
        self, V0, F, E, BP, body_V0, body_F,
        w, vert_damping, dist_compliance_arr, bend_compliance_arr,
    ):
        V0f = V0.astype(np.float32)
        self.x.from_numpy(V0f)
        self.p.from_numpy(V0f)
        self.v.from_numpy(np.zeros_like(V0f))
        self.w.from_numpy(w.astype(np.float32))
        self.damp.from_numpy(vert_damping.astype(np.float32))
        self.inv_edge_val_fld.from_numpy(self.inv_edge_val)
        self.inv_bend_val_fld.from_numpy(self.inv_bend_val)

        self.edges.from_numpy(E.astype(np.int32))
        rl = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1).astype(np.float32)
        self.rest_len.from_numpy(rl)
        self.dist_compliance.from_numpy(dist_compliance_arr.astype(np.float32))

        if self.NB > 0:
            self.bend_idx.from_numpy(BP[:, 2:4].astype(np.int32))
            br = np.linalg.norm(
                V0[BP[:, 2]] - V0[BP[:, 3]], axis=1
            ).astype(np.float32)
            self.bend_rest.from_numpy(br)
            self.bend_compliance.from_numpy(bend_compliance_arr.astype(np.float32))

        self.body_face_idx.from_numpy(body_F.astype(np.int32).flatten())
        self.face_idx.from_numpy(F.astype(np.int32).flatten())

    def set_body(self, body_V):
        self.body_x.from_numpy(body_V.astype(np.float32))
        body_F = self.body_face_idx.to_numpy().reshape(-1, 3)
        vn = per_vertex_normals(body_V.astype(np.float32), body_F)
        self.body_n.from_numpy(vn)

    def set_color(self, C):
        self.color.from_numpy(C.astype(np.float32))

    # --------------------------------------------------------------- kernels
    @ti.kernel
    def predict(self, dt: ti.f32):
        g = self.gravity
        for i in self.x:
            if self.w[i] > 0:
                self.v[i] = self.v[i] * (1.0 - self.damp[i]) + g * dt
                self.p[i] = self.x[i] + self.v[i] * dt
            else:
                self.p[i] = self.x[i]

    @ti.kernel
    def reset_lambdas(self):
        for i in self.lambda_d:
            self.lambda_d[i] = 0.0
        for i in self.lambda_b:
            self.lambda_b[i] = 0.0

    @ti.kernel
    def clear_dp(self):
        for i in self.dp:
            self.dp[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_dp(self):
        # Flush Jacobi delta buffer into p. One thread per vertex, no race.
        for i in self.p:
            self.p[i] += self.dp[i]
            self.dp[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def solve_distance(self, dt: ti.f32):
        inv_dt2 = 1.0 / (dt * dt)
        for e in self.edges:
            alpha = self.dist_compliance[e] * inv_dt2
            i = self.edges[e][0]
            j = self.edges[e][1]
            wi = self.w[i]
            wj = self.w[j]
            wsum = wi + wj
            if wsum > 0:
                d = self.p[i] - self.p[j]
                L = d.norm(1e-8)
                n = d / L
                C = L - self.rest_len[e]
                dlambda = (-C - alpha * self.lambda_d[e]) / (wsum + alpha)
                self.lambda_d[e] += dlambda
                dpv = dlambda * n
                # Jacobi: scale per vertex by 1/valence so summed parallel
                # contributions match one-at-a-time Gauss-Seidel magnitude.
                self.dp[i] += wi * dpv * self.inv_edge_val_fld[i]
                self.dp[j] -= wj * dpv * self.inv_edge_val_fld[j]

    @ti.kernel
    def solve_bending(self, dt: ti.f32):
        inv_dt2 = 1.0 / (dt * dt)
        for b in self.bend_idx:
            alpha = self.bend_compliance[b] * inv_dt2
            i = self.bend_idx[b][0]
            j = self.bend_idx[b][1]
            wi = self.w[i]
            wj = self.w[j]
            wsum = wi + wj
            if wsum > 0:
                d = self.p[i] - self.p[j]
                L = d.norm(1e-8)
                n = d / L
                C = L - self.bend_rest[b]
                dlambda = (-C - alpha * self.lambda_b[b]) / (wsum + alpha)
                self.lambda_b[b] += dlambda
                dpv = dlambda * n
                self.dp[i] += wi * dpv * self.inv_bend_val_fld[i]
                self.dp[j] -= wj * dpv * self.inv_bend_val_fld[j]

    @ti.kernel
    def solve_collision(self, radius: ti.f32):
        # Per cloth vertex: find nearest body vertex and push outside its
        # sphere along the body normal. O(N · NBV) — fine for demo sizes.
        for i in self.p:
            pi = self.p[i]
            best = ti.i32(-1)
            best_d2 = ti.f32(1e18)
            for j in range(self.NBV):
                d2 = (pi - self.body_x[j]).norm_sqr()
                if d2 < best_d2:
                    best_d2 = d2
                    best = j
            if best >= 0:
                bx = self.body_x[best]
                bn = self.body_n[best]
                diff = pi - bx
                signed = diff.dot(bn)
                if signed < radius:
                    self.p[i] = pi + (radius - signed) * bn

    @ti.kernel
    def finalize(self, dt: ti.f32):
        inv_dt = 1.0 / dt
        for i in self.x:
            if self.w[i] > 0:
                self.v[i] = (self.p[i] - self.x[i]) * inv_dt
                self.x[i] = self.p[i]

    # ----------------------------------------------------------------- step
    def step(self):
        sub_dt = self.dt / self.substeps
        t = self.timing
        _p = time.perf_counter
        for _ in range(self.substeps):
            ti.sync()
            t0 = _p()
            self.predict(sub_dt)
            self.reset_lambdas()
            self.clear_dp()
            ti.sync()
            t1 = _p()
            for _ in range(self.iterations):
                self.solve_distance(sub_dt)
                self.apply_dp()
                if self.NB > 0:
                    self.solve_bending(sub_dt)
                    self.apply_dp()
                self.solve_collision(self.collision_radius)
            ti.sync()
            t2 = _p()
            self.finalize(sub_dt)
            ti.sync()
            t3 = _p()
            t["predict"] += t1 - t0
            t["solve"]   += t2 - t1
            t["finalize"]+= t3 - t2
        t["steps"] += 1

    # ----------------------------------------------------------------- stats
    def reset_timing(self):
        self.timing = dict(predict=0.0, solve=0.0, finalize=0.0, steps=0)

    def timing_report(self):
        t = self.timing
        n = max(1, t["steps"])
        total = t["predict"] + t["solve"] + t["finalize"]
        lines = [
            f"[timing] {n} frames   total {total:.2f}s   "
            f"{total / n * 1000:.1f} ms/frame   "
            f"{n / total if total > 0 else 0:.1f} FPS",
            f"         predict  {t['predict']:.2f}s  "
            f"({t['predict']/n*1000:.2f} ms/frame)",
            f"         solve    {t['solve']:.2f}s  "
            f"({t['solve']/n*1000:.2f} ms/frame)   "
            f"[dist+bend+coll × {self.iterations} iters × "
            f"{self.substeps} substeps]",
            f"         finalize {t['finalize']:.2f}s  "
            f"({t['finalize']/n*1000:.2f} ms/frame)",
        ]
        return "\n".join(lines)
