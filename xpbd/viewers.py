"""Three viewer back-ends for the simulator:

    run_gui         - interactive Taichi GGUI window (needs Vulkan)
    run_matplotlib  - 3D matplotlib animation (window or saved mp4/gif)
    run_headless    - no display; dump cloth vertices as .npy frames

Each takes the same (cloth, data, args) tuple from `xpbd.cli`.
"""

import os
import time

import numpy as np
import taichi as ti


def video_stem(data):
    """Filename stem that lists every garment in the run."""
    tag = "+".join(data["garment_names"]) if data["garment_names"] else "cloth"
    return f"{data['sample']}_{tag}"


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

    bb_min = body_frames[0].min(axis=0) - 0.2
    bb_max = body_frames[0].max(axis=0) + 0.2
    ax.set_xlim(bb_min[0], bb_max[0])
    ax.set_ylim(bb_min[1], bb_max[1])
    ax.set_zlim(bb_min[2], bb_max[2])
    ax.set_box_aspect((bb_max - bb_min))
    ax.view_init(elev=10, azim=-70)
    title = (
        f"XPBD cloth — sample {data['sample']} / "
        f"{'+'.join(data['garment_names'])}"
    )
    ax.set_title(title)

    body_coll = Poly3DCollection(
        body_frames[0][F_body],
        facecolor=(0.85, 0.72, 0.60, 0.25),
        edgecolor="none",
    )
    ax.add_collection3d(body_coll)

    cloth_V = cloth.x.to_numpy()
    face_colors = C_cloth[F_cloth].mean(axis=1)
    cloth_coll = Poly3DCollection(
        cloth_V[F_cloth],
        facecolors=face_colors,
        edgecolor=(0, 0, 0, 0.15),
        linewidths=0.2,
    )
    ax.add_collection3d(cloth_coll)

    step_state = {"i": 0}

    def update(_):
        if n_body_frames > 1:
            cloth.set_body(body_frames[step_state["i"] % n_body_frames])
        cloth.step()
        step_state["i"] += 1
        cloth_coll.set_verts(cloth.x.to_numpy()[F_cloth])
        if n_body_frames > 1:
            body_coll.set_verts(
                body_frames[step_state["i"] % n_body_frames][F_body]
            )
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
        out = os.path.join(args.out, f"{video_stem(data)}{ext}")
        print(f"[mpl] saving animation to {out}")
        anim.save(out, writer=writer, dpi=110)
    else:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
def run_gui(cloth, data, args):
    """Interactive Taichi GGUI viewer (requires Vulkan)."""
    window = ti.ui.Window("XPBD Cloth on CLOTH3D", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.08, 0.09, 0.12))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    center = data["V0"].mean(axis=0)
    camera.position(center[0] + 2.5, center[1] - 2.5, center[2] + 0.3)
    camera.lookat(center[0], center[1], center[2])
    camera.up(0, 0, 1)
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
            if n_body_frames > 1:
                cloth.set_body(body_frames[step_idx % n_body_frames])
            cloth.step()
            step_idx += 1

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.35, 0.35, 0.4))
        scene.point_light(
            pos=(center[0] + 2, center[1] - 2, center[2] + 3),
            color=(1.0, 1.0, 1.0),
        )
        scene.point_light(
            pos=(center[0] - 2, center[1] + 2, center[2] + 2),
            color=(0.6, 0.6, 0.8),
        )

        if show_body:
            scene.mesh(
                cloth.body_x, indices=cloth.body_face_idx,
                color=(0.85, 0.72, 0.60), two_sided=True,
            )
        scene.mesh(
            cloth.x, indices=cloth.face_idx,
            per_vertex_color=cloth.color, two_sided=True,
        )

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


# ---------------------------------------------------------------------------
def run_headless(cloth, data, args):
    """Save cloth vertex positions as .npy frames; no display required."""
    print("[headless] running without GUI; saving frames as .npy ...")
    os.makedirs(args.out, exist_ok=True)
    body_frames = data["body_V_seq"]
    n_body_frames = body_frames.shape[0]
    stem = video_stem(data)
    for i in range(args.steps):
        if n_body_frames > 1:
            cloth.set_body(body_frames[i % n_body_frames])
        cloth.step()
        if i % args.save_every == 0:
            np.save(
                os.path.join(args.out, f"{stem}_{i:05d}.npy"),
                cloth.x.to_numpy(),
            )
            print(f"  step {i}/{args.steps}")
    print("[headless] done.")


# ---------------------------------------------------------------------------
def run_export(cloth, data, args):
    """Headless sim that writes eval-compatible per-garment NPZ files.

    One step = one frame. After each step we snapshot cloth.x and record
    the wall-time spent inside `cloth.step()` so the benchmark's
    cost.time_per_frame metric has meaningful data. We then slice the
    combined cloth into per-garment meshes and call `write_result_npz`
    once per garment; the writer rotates z-up CLOTH3D into y-up before
    dumping. Optionally also emits the per-sample `{sample}.npz` the
    eval's source loader reopens.
    """
    from .export import slice_garment, write_result_npz, write_sample_npz

    # Do the per-sample CLOTH3D extraction first, before running the sim.
    # This way a broken extractor path fails fast; it also means the eval's
    # source loader has its input ready immediately after our sim finishes.
    if getattr(args, "save_sample_npz", False):
        sample_npz_dir = args.sample_npz_dir
        print(f"[export] extracting per-sample CLOTH3D NPZ to {sample_npz_dir}")
        p = write_sample_npz(sample_npz_dir, data["sample"])
        print(f"[export] wrote {p}")

    body_frames = data["body_V_seq"]
    n_body_frames = body_frames.shape[0]
    n_frames = min(int(args.steps), n_body_frames)
    if n_frames < int(args.steps):
        print(
            f"[export] clamping steps from {args.steps} to {n_frames} "
            f"(bounded by body_frames={n_body_frames})"
        )

    names = data["garment_names"]
    vert_gid = data["vert_gid"]
    F_full = data["F"]
    gt_by_name = data.get("gt_V_by_garment", {})

    # Step loop — record cloth state after each frame and timings.
    V_per_frame = np.empty((n_frames, cloth.x.shape[0], 3), dtype=np.float32)
    frame_ms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        if n_body_frames > 1:
            cloth.set_body(body_frames[i % n_body_frames])
        t0 = time.perf_counter()
        cloth.step()
        ti.sync()
        frame_ms[i] = (time.perf_counter() - t0) * 1000.0
        V_per_frame[i] = cloth.x.to_numpy()
        if (i + 1) % max(1, n_frames // 10) == 0:
            print(f"  [export] step {i+1}/{n_frames}")

    # Human (SMPL) mesh matches the frames we actually simulated.
    human_V_seq = body_frames[:n_frames].astype(np.float64)
    human_faces = np.asarray(data["body_F"], dtype=np.int32)

    os.makedirs(args.npz_out, exist_ok=True)

    written: list[str] = []
    for gi, name in enumerate(names):
        vert_idx, F_local = slice_garment(F_full, vert_gid, gi)
        if vert_idx.size == 0:
            print(f"[export] skipping empty garment '{name}'")
            continue

        sim_V_seq = V_per_frame[:, vert_idx, :].astype(np.float64)
        gt_V_seq = gt_by_name.get(name)
        if gt_V_seq is None:
            gt_V_seq = np.zeros_like(sim_V_seq)
        else:
            gt_V_seq = np.asarray(gt_V_seq[:n_frames], dtype=np.float64)

        fabric = (
            data["garment_fabrics"][gi]
            if gi < len(data["garment_fabrics"])
            else ""
        )

        path = write_result_npz(
            out_dir=args.npz_out,
            sample=data["sample"],
            garment_name=name,
            fabric=fabric,
            frame_dt=float(args.dt),
            start_frame=0,
            requested_frames=n_frames,
            faces_local=F_local,
            sim_V_seq=sim_V_seq,
            gt_V_seq=gt_V_seq,
            human_faces=human_faces,
            human_V_seq=human_V_seq,
            frame_wall_time_ms=frame_ms.tolist(),
            world_valid=True,
            scene_mode="both",
            ground_enabled=False,
            # Simulator runs in CLOTH3D z-up; in the y-up eval frame the
            # equivalent gravity vector is (0, -9.81, 0).
            effective_gravity=(0.0, -9.81, 0.0),
            cloth_reference_shape="rest",
            solver_name="xPBD-Taichi-CMU",
        )
        written.append(path)
        print(f"[export] wrote {path}")

    print(f"[export] done. {len(written)} garment file(s) written.")
    return written
