"""
XPBD cloth simulation on CLOTH3D garments, implemented in Taichi.

- Loads a CLOTH3D sample through the existing extract_sample_data pipeline.
- Initializes the garment at frame 0 (already fitted to the body).
- Simulates with XPBD: distance + bending constraints, gravity, and per-vertex
  body collision using the animated SMPL mesh as a moving collider.
- Renders through Taichi GGUI (falls back to headless mode with --no-gui).

Run:
    python3 xpbd_cloth.py --sample 00016 --garment Tshirt
"""

import argparse
import os
import sys
import time

import numpy as np
import taichi as ti

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "cloth3d", "DataReader"))
sys.path.insert(0, os.path.join(HERE, "cloth3d", "Demo"))

from extract_sample_data import extract_sample_single_frame, reader, get_num_frames  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sample(sample, garment_name=None, n_body_frames=1):
    """Return cloth rest + topology, plus a sequence of body meshes."""
    data0 = extract_sample_single_frame(sample, 0, use_uv_map=False, show_display=False)

    garment_names = list(data0["garment_names"])
    if garment_name is None:
        garment_name = garment_names[0]
    assert garment_name in garment_names, (
        f"garment '{garment_name}' not in sample '{sample}' "
        f"(available: {garment_names})"
    )

    key = f"garment_{garment_name}"
    V0 = np.asarray(data0[f"{key}_V"], dtype=np.float32)
    F = np.asarray(data0[f"{key}_F"], dtype=np.int32)
    C = np.asarray(data0[f"{key}_C"], dtype=np.float32) / 255.0

    total_frames = get_num_frames(sample)
    n_body_frames = max(1, min(n_body_frames, total_frames))
    body_V_seq = np.empty((n_body_frames, 6890, 3), dtype=np.float32)
    body_F = None
    for i in range(n_body_frames):
        V, F_body = reader.read_human(sample, i)
        body_V_seq[i] = V
        if body_F is None:
            body_F = np.asarray(F_body, dtype=np.int32)

    print(
        f"[data] sample={sample} garment={garment_name} "
        f"cloth_V={V0.shape[0]} cloth_F={F.shape[0]} "
        f"body_V={body_V_seq.shape[1]} body_F={body_F.shape[0]} "
        f"frames={n_body_frames}"
    )
    return dict(
        V0=V0,
        F=F,
        C=C,
        body_V_seq=body_V_seq,
        body_F=body_F,
        garment_name=garment_name,
        sample=sample,
    )


def build_edges(F):
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    return E.astype(np.int32)


def build_bending_pairs(F):
    """Return (M,4) indices (v1,v2,v3,v4) for dihedral bending.

    v1,v2 form the shared edge. v3,v4 are the opposite vertices of the
    two triangles that share it. We use the simple distance constraint
    between v3 and v4 (classic PBD bending shortcut).
    """
    edge2tri = {}
    for ti_, tri in enumerate(F):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (int(min(a, b)), int(max(a, b)))
            edge2tri.setdefault(e, []).append(ti_)
    pairs = []
    for (a, b), tris in edge2tri.items():
        if len(tris) != 2:
            continue
        opp = []
        for t in tris:
            for v in F[t]:
                if v != a and v != b:
                    opp.append(int(v))
                    break
        pairs.append([a, b, opp[0], opp[1]])
    return np.array(pairs, dtype=np.int32)


def per_vertex_normals(V, F):
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vn = np.zeros_like(V)
    np.add.at(vn, F[:, 0], fn)
    np.add.at(vn, F[:, 1], fn)
    np.add.at(vn, F[:, 2], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn.astype(np.float32)


# ---------------------------------------------------------------------------
# XPBD solver
# ---------------------------------------------------------------------------
@ti.data_oriented
class XPBDCloth:
    def __init__(
        self,
        V0,
        F,
        body_V0,
        body_F,
        dt=1.0 / 60.0,
        substeps=10,
        iterations=5,
        distance_compliance=1e-8,
        bend_compliance=5e-6,
        damping=0.02,
        gravity=(0.0, 0.0, -9.81),
        collision_radius=0.01,
    ):
        self.N = V0.shape[0]
        self.NF = F.shape[0]
        self.NBV = body_V0.shape[0]
        self.NBF = body_F.shape[0]

        E = build_edges(F)
        BP = build_bending_pairs(F)
        self.NE = E.shape[0]
        self.NB = BP.shape[0]

        print(
            f"[solver] N={self.N} NF={self.NF} NE={self.NE} NB={self.NB} "
            f"body_V={self.NBV} body_F={self.NBF}"
        )

        self.dt = dt
        self.substeps = substeps
        self.iterations = iterations
        self.distance_compliance = distance_compliance
        self.bend_compliance = bend_compliance
        self.damping = damping
        self.gravity = ti.Vector(list(gravity))
        self.collision_radius = collision_radius

        # particles
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.p = ti.Vector.field(3, ti.f32, self.N)
        self.w = ti.field(ti.f32, self.N)

        # distance constraints
        self.edges = ti.Vector.field(2, ti.i32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)
        self.lambda_d = ti.field(ti.f32, self.NE)

        # bending constraints (distance between opposite vertices)
        self.bend_idx = ti.Vector.field(2, ti.i32, self.NB)
        self.bend_rest = ti.field(ti.f32, self.NB)
        self.lambda_b = ti.field(ti.f32, self.NB)

        # body
        self.body_x = ti.Vector.field(3, ti.f32, self.NBV)
        self.body_n = ti.Vector.field(3, ti.f32, self.NBV)

        # rendering
        self.face_idx = ti.field(ti.i32, self.NF * 3)
        self.body_face_idx = ti.field(ti.i32, self.NBF * 3)
        self.color = ti.Vector.field(3, ti.f32, self.N)

        self._load_static(V0, F, E, BP, body_V0, body_F)
        self.set_body(body_V0)

    def _load_static(self, V0, F, E, BP, body_V0, body_F):
        V0f = V0.astype(np.float32)
        self.x.from_numpy(V0f)
        self.p.from_numpy(V0f)
        self.v.from_numpy(np.zeros_like(V0f))
        self.w.from_numpy(np.ones(self.N, dtype=np.float32))

        self.edges.from_numpy(E.astype(np.int32))
        rl = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1).astype(np.float32)
        self.rest_len.from_numpy(rl)

        if self.NB > 0:
            self.bend_idx.from_numpy(BP[:, 2:4].astype(np.int32))
            br = np.linalg.norm(V0[BP[:, 2]] - V0[BP[:, 3]], axis=1).astype(np.float32)
            self.bend_rest.from_numpy(br)

        self.body_face_idx.from_numpy(body_F.astype(np.int32).flatten())
        self.face_idx.from_numpy(F.astype(np.int32).flatten())

    def set_body(self, body_V):
        self.body_x.from_numpy(body_V.astype(np.float32))
        body_F = self.body_face_idx.to_numpy().reshape(-1, 3)
        vn = per_vertex_normals(body_V.astype(np.float32), body_F)
        self.body_n.from_numpy(vn)

    def set_color(self, C):
        self.color.from_numpy(C.astype(np.float32))

    # ---- Taichi kernels ----
    @ti.kernel
    def predict(self, dt: ti.f32, damping: ti.f32):
        g = self.gravity
        for i in self.x:
            if self.w[i] > 0:
                self.v[i] = self.v[i] * (1.0 - damping) + g * dt
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
    def solve_distance(self, dt: ti.f32, compliance: ti.f32):
        alpha = compliance / (dt * dt)
        for e in self.edges:
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
                dp = dlambda * n
                # Unsynchronized writes are fine at this scale (PBD is iterative).
                self.p[i] += wi * dp
                self.p[j] -= wj * dp

    @ti.kernel
    def solve_bending(self, dt: ti.f32, compliance: ti.f32):
        alpha = compliance / (dt * dt)
        for b in self.bend_idx:
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
                dp = dlambda * n
                self.p[i] += wi * dp
                self.p[j] -= wj * dp

    @ti.kernel
    def solve_collision(self, radius: ti.f32):
        # Per cloth vertex: find nearest body vertex and push outside its
        # sphere along the body normal (good enough for a demo).
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
                    # push along body normal to the collision surface
                    self.p[i] = pi + (radius - signed) * bn

    @ti.kernel
    def finalize(self, dt: ti.f32):
        inv_dt = 1.0 / dt
        for i in self.x:
            if self.w[i] > 0:
                self.v[i] = (self.p[i] - self.x[i]) * inv_dt
                self.x[i] = self.p[i]

    def step(self):
        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.predict(sub_dt, self.damping)
            self.reset_lambdas()
            for _ in range(self.iterations):
                self.solve_distance(sub_dt, self.distance_compliance)
                if self.NB > 0:
                    self.solve_bending(sub_dt, self.bend_compliance)
                self.solve_collision(self.collision_radius)
            self.finalize(sub_dt)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def run_matplotlib(cloth, data, args):
    """Matplotlib 3D animation fallback (works without Vulkan / in WSL)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection="3d")

    body_frames = data["body_V_seq"]
    n_body_frames = body_frames.shape[0]
    F_cloth = data["F"]
    F_body = data["body_F"]
    C_cloth = data["C"]

    # fix axes based on body bbox + padding
    bb_min = data["body_V_seq"][0].min(axis=0) - 0.2
    bb_max = data["body_V_seq"][0].max(axis=0) + 0.2
    ax.set_xlim(bb_min[0], bb_max[0])
    ax.set_ylim(bb_min[1], bb_max[1])
    ax.set_zlim(bb_min[2], bb_max[2])
    ax.set_box_aspect((bb_max - bb_min))
    ax.view_init(elev=10, azim=-70)
    ax.set_title(f"XPBD cloth — sample {data['sample']} / {data['garment_name']}")

    body_coll = Poly3DCollection(body_frames[0][F_body],
                                 facecolor=(0.85, 0.72, 0.60, 0.25),
                                 edgecolor="none")
    ax.add_collection3d(body_coll)

    cloth_V = cloth.x.to_numpy()
    face_colors = C_cloth[F_cloth].mean(axis=1)
    cloth_coll = Poly3DCollection(cloth_V[F_cloth],
                                  facecolors=face_colors,
                                  edgecolor=(0, 0, 0, 0.15),
                                  linewidths=0.2)
    ax.add_collection3d(cloth_coll)

    step_state = {"i": 0}

    def update(_):
        if n_body_frames > 1:
            cloth.set_body(body_frames[step_state["i"] % n_body_frames])
        cloth.step()
        step_state["i"] += 1
        cloth_coll.set_verts(cloth.x.to_numpy()[F_cloth])
        if n_body_frames > 1:
            body_coll.set_verts(body_frames[step_state["i"] % n_body_frames][F_body])
        return cloth_coll, body_coll

    anim = FuncAnimation(fig, update, frames=args.steps, interval=30, blit=False)
    if args.save_video:
        os.makedirs(args.out, exist_ok=True)
        try:
            import matplotlib as mpl
            import imageio_ffmpeg
            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=30, bitrate=2400)
            ext = ".mp4"
        except Exception as e:
            print(f"[mpl] ffmpeg unavailable ({e}); saving GIF instead.")
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=20)
            ext = ".gif"
        out = os.path.join(args.out,
                           f"{data['sample']}_{data['garment_name']}{ext}")
        print(f"[mpl] saving animation to {out}")
        anim.save(out, writer=writer, dpi=110)
    else:
        plt.tight_layout()
        plt.show()


def run_gui(cloth, data, args):
    """Interactive Taichi GGUI viewer (requires Vulkan)."""
    window = ti.ui.Window("XPBD Cloth on CLOTH3D", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.08, 0.09, 0.12))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Frame center
    center = data["V0"].mean(axis=0)
    camera.position(center[0] + 2.5, center[1] - 2.5, center[2] + 0.3)
    camera.lookat(center[0], center[1], center[2])
    camera.up(0, 0, 1)  # CLOTH3D is z-up
    camera.fov(45)

    body_frames = data["body_V_seq"]
    n_body_frames = body_frames.shape[0]

    step_idx = 0
    paused = False
    show_body = True
    last_t = time.time()
    frame_count = 0

    print("[gui] controls: space=pause  r=reset  b=toggle body  esc=quit")
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            if window.event.key == ti.ui.SPACE:
                paused = not paused
            if window.event.key == "r":
                cloth.x.from_numpy(data["V0"])
                cloth.v.from_numpy(np.zeros_like(data["V0"]))
                step_idx = 0
            if window.event.key == "b":
                show_body = not show_body

        if not paused:
            # animate body if we have multiple frames
            if n_body_frames > 1:
                bf = step_idx % n_body_frames
                cloth.set_body(body_frames[bf])
            cloth.step()
            step_idx += 1

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.35, 0.35, 0.4))
        scene.point_light(pos=(center[0] + 2, center[1] - 2, center[2] + 3),
                          color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(center[0] - 2, center[1] + 2, center[2] + 2),
                          color=(0.6, 0.6, 0.8))

        if show_body:
            scene.mesh(cloth.body_x, indices=cloth.body_face_idx,
                       color=(0.85, 0.72, 0.60), two_sided=True)
        scene.mesh(cloth.x, indices=cloth.face_idx,
                   per_vertex_color=cloth.color, two_sided=True)

        canvas.scene(scene)
        window.show()

        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - last_t + 1e-9)
            last_t = now
            window.GUI.begin("info", 0.02, 0.02, 0.2, 0.12)
            window.GUI.text(f"step {step_idx}  fps {fps:.1f}")
            window.GUI.end()


def run_headless(cloth, data, args):
    print("[headless] running without GUI; saving frames as .npy ...")
    os.makedirs(args.out, exist_ok=True)
    body_frames = data["body_V_seq"]
    n_body_frames = body_frames.shape[0]
    for i in range(args.steps):
        if n_body_frames > 1:
            cloth.set_body(body_frames[i % n_body_frames])
        cloth.step()
        if i % args.save_every == 0:
            np.save(os.path.join(args.out, f"cloth_{i:05d}.npy"),
                    cloth.x.to_numpy())
            print(f"  step {i}/{args.steps}")
    print("[headless] done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="00016")
    p.add_argument("--garment", default=None, help="garment name, e.g. Tshirt")
    p.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "vulkan"])
    p.add_argument("--body_frames", type=int, default=1,
                   help="number of body frames to animate; 1 = static")
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--dist_compliance", type=float, default=1e-8)
    p.add_argument("--bend_compliance", type=float, default=5e-6)
    p.add_argument("--damping", type=float, default=0.02)
    p.add_argument("--collision_radius", type=float, default=0.01)
    p.add_argument("--viewer", default="auto",
                   choices=["auto", "ggui", "mpl", "none"],
                   help="auto tries ggui then falls back to matplotlib")
    p.add_argument("--no_gui", action="store_true",
                   help="alias for --viewer none")
    p.add_argument("--save_video", action="store_true",
                   help="with --viewer mpl, save an mp4 instead of showing a window")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--out", default=os.path.join(HERE, "xpbd_out"))
    args = p.parse_args()
    if args.no_gui:
        args.viewer = "none"

    arch_map = {"cpu": ti.cpu, "gpu": ti.gpu, "vulkan": ti.vulkan}
    ti.init(arch=arch_map[args.arch], default_fp=ti.f32)

    data = load_sample(args.sample, args.garment, n_body_frames=args.body_frames)
    cloth = XPBDCloth(
        V0=data["V0"],
        F=data["F"],
        body_V0=data["body_V_seq"][0],
        body_F=data["body_F"],
        dt=args.dt,
        substeps=args.substeps,
        iterations=args.iters,
        distance_compliance=args.dist_compliance,
        bend_compliance=args.bend_compliance,
        damping=args.damping,
        collision_radius=args.collision_radius,
    )
    cloth.set_color(data["C"])

    if args.viewer == "none":
        run_headless(cloth, data, args)
    elif args.viewer == "mpl":
        run_matplotlib(cloth, data, args)
    elif args.viewer == "ggui":
        run_gui(cloth, data, args)
    else:  # auto
        try:
            run_gui(cloth, data, args)
        except RuntimeError as e:
            print(f"[viewer] GGUI unavailable ({e}); falling back to matplotlib.")
            run_matplotlib(cloth, data, args)


if __name__ == "__main__":
    main()
