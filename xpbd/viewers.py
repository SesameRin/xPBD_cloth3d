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

    cloth.reset_timing()
    wall_t0 = time.perf_counter()
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
        stem = video_stem(data)
        if getattr(args, "arch", "cpu") != "cpu":
            stem = f"{stem}_{args.arch}"
        out = os.path.join(args.out, f"{stem}{ext}")
        print(f"[mpl] saving animation to {out}")
        anim.save(out, writer=writer, dpi=110)
    else:
        plt.tight_layout()
        plt.show()
    wall = time.perf_counter() - wall_t0
    print(cloth.timing_report())
    print(f"[timing] wall-clock incl. rendering: {wall:.2f}s "
          f"({wall / max(1, args.steps) * 1000:.1f} ms/frame)")


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
    cloth.reset_timing()
    wall_t0 = time.perf_counter()
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
    wall = time.perf_counter() - wall_t0
    print("[headless] done.")
    print(cloth.timing_report())
    print(f"[timing] wall-clock: {wall:.2f}s "
          f"({wall / max(1, args.steps) * 1000:.1f} ms/frame)")
