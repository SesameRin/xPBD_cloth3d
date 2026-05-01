"""Command-line entry point. Wires argparse → data load → solver → viewer."""

import argparse
import os

import taichi as ti

from .data import get_num_frames, load_sample
from .fabrics import FABRIC_PRESETS
from .solver import XPBDCloth
from .viewers import run_export, run_gui, run_headless, run_matplotlib

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="00016")
    p.add_argument(
        "--garments", default="all",
        help="comma-separated garment names, or 'all'. "
             "Example: --garments Tshirt,Trousers",
    )
    p.add_argument(
        "--garment", default=None,
        help="[deprecated] single-garment alias for --garments",
    )
    p.add_argument(
        "--force_fabric", default=None,
        choices=sorted(FABRIC_PRESETS.keys()) + [None],
        help="override fabric for ALL garments (e.g. cotton). Useful for "
             "matching a C-IPC baseline that uses a single material.",
    )
    p.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "vulkan"])
    p.add_argument("--body_frames", type=int, default=None,
                   help="number of body frames to animate; "
                        "omit or pass -1 to use every frame in the sample")
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--dist_compliance", type=float, default=None,
                   help="override per-garment distance compliance")
    p.add_argument("--bend_compliance", type=float, default=None,
                   help="override per-garment bending compliance")
    p.add_argument("--damping", type=float, default=None,
                   help="override per-garment damping")
    p.add_argument("--collision_radius", type=float, default=0.01)
    p.add_argument("--viewer", default="auto",
                   choices=["auto", "ggui", "mpl", "none"],
                   help="auto tries ggui then falls back to matplotlib")
    p.add_argument("--no_gui", action="store_true",
                   help="alias for --viewer none")
    p.add_argument("--save_video", action="store_true",
                   help="with --viewer mpl, save an mp4 instead of showing a window")
    p.add_argument("--steps", type=int, default=None,
                   help="number of simulation steps to run; "
                        "omit or pass -1 to match the sample's frame count")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--out", default=os.path.join(_REPO, "xpbd_out"))

    # Eval-compatible NPZ export. Produces per-garment
    # {sample}_{garment}_sim.npz files that the teammate's cloth3d_eval
    # module can consume without changes.
    p.add_argument(
        "--save_npz", action="store_true",
        help="run in export mode: write per-garment sim NPZ files that "
             "match the cloth3d_benchmark eval schema.",
    )
    p.add_argument(
        "--npz_out",
        default=os.path.join(_REPO, "xpbd_out", "results_xpbd"),
        help="directory where {sample}_{garment}_sim.npz files are written.",
    )
    p.add_argument(
        "--save_sample_npz", action="store_true",
        help="also extract the per-sample CLOTH3D NPZ that the eval's "
             "source loader reopens (convert_from_cloth3d).",
    )
    p.add_argument(
        "--sample_npz_dir",
        default=os.path.join(_REPO, "xpbd_out", "cloth3d_data"),
        help="directory for --save_sample_npz output.",
    )

    # Drop-experiment offset. Mirrors the teammate's
    # `--garment_y_translation` flag in cloth3d_xpbd / cloth3d_sim: lift
    # every cloth vertex (including the per-frame GT used by the export)
    # by `+t` in the eval's y-up frame, which corresponds to `+t` along
    # CLOTH3D's z-axis. Body and the un-draped rest mesh are NOT lifted.
    # Use `3.0` to reproduce partner's drop run; leave at 0.0 for the
    # original worn-cloth experiment.
    p.add_argument(
        "--garment_y_translation", type=float, default=0.0,
        help="lift cloth (and GT trajectory) by t metres along CLOTH3D's "
             "z-axis to mirror partner's drop experiment. "
             "Use 3.0 for an apples-to-apples drop comparison.",
    )
    # Mirrors partner's --freeze_human_mesh on: pin the SMPL body
    # collider at frame 0 instead of cycling through the body sequence.
    # Uses argparse's tri-state pattern (default None) so we can
    # auto-default it to True when --garment_y_translation is set, while
    # still letting the user force it off with --no_freeze_body.
    p.add_argument(
        "--freeze_body", dest="freeze_body", action="store_true",
        default=None,
        help="hold the SMPL body at frame 0 for the whole sim "
             "(default: auto-on whenever --garment_y_translation != 0).",
    )
    p.add_argument(
        "--no_freeze_body", dest="freeze_body", action="store_false",
        help="force body animation on even during drop runs.",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.no_gui:
        args.viewer = "none"

    # back-compat: --garment takes precedence if explicitly passed
    garments_spec = args.garment if args.garment else args.garments

    # Drop runs default to a frozen body collider. The user can flip
    # this with --no_freeze_body. Worn-cloth runs leave it off so the
    # body keeps animating as before.
    if args.freeze_body is None:
        args.freeze_body = bool(args.garment_y_translation)
        if args.freeze_body:
            print("[cli] --freeze_body auto-on (drop run)")

    # Auto-default body_frames and steps to the sample's full length
    # when the user doesn't specify (or passes -1). CLOTH3D stores one
    # pose per frame, so "full sample" == get_num_frames(sample).
    if args.body_frames is None or args.body_frames <= 0:
        args.body_frames = get_num_frames(args.sample)
        print(f"[cli] --body_frames auto = {args.body_frames} (full sample)")
    if args.steps is None or args.steps <= 0:
        args.steps = args.body_frames
        print(f"[cli] --steps auto = {args.steps} (matches body_frames)")

    arch_map = {"cpu": ti.cpu, "gpu": ti.gpu, "vulkan": ti.vulkan}
    ti.init(arch=arch_map[args.arch], default_fp=ti.f32)

    data = load_sample(
        args.sample,
        garments_spec,
        n_body_frames=args.body_frames,
        need_gt_trajectory=args.save_npz,
        garment_y_translation=args.garment_y_translation,
    )

    fabrics = data["garment_fabrics"]
    if args.force_fabric:
        fabrics = [args.force_fabric] * len(fabrics)
        print(f"[solver] forcing fabric={args.force_fabric} on all garments")

    cloth = XPBDCloth(
        V0=data["V0"],
        F=data["F"],
        body_V0=data["body_V_seq"][0],
        body_F=data["body_F"],
        vert_gid=data["vert_gid"],
        garment_fabrics=fabrics,
        dt=args.dt,
        substeps=args.substeps,
        iterations=args.iters,
        dist_compliance_override=args.dist_compliance,
        bend_compliance_override=args.bend_compliance,
        damping_override=args.damping,
        collision_radius=args.collision_radius,
        gpu_safe=(args.arch != "cpu"),
    )
    cloth.set_color(data["C"])

    if args.save_npz:
        run_export(cloth, data, args)
    elif args.viewer == "none":
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
