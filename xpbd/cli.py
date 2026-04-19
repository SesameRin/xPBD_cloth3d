"""Command-line entry point. Wires argparse → data load → solver → viewer."""

import argparse
import os

import taichi as ti

from .data import load_sample
from .fabrics import FABRIC_PRESETS
from .solver import XPBDCloth
from .viewers import run_gui, run_headless, run_matplotlib

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
    p.add_argument("--body_frames", type=int, default=1,
                   help="number of body frames to animate; 1 = static")
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
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--out", default=os.path.join(_REPO, "xpbd_out"))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.no_gui:
        args.viewer = "none"

    # back-compat: --garment takes precedence if explicitly passed
    garments_spec = args.garment if args.garment else args.garments

    arch_map = {"cpu": ti.cpu, "gpu": ti.gpu, "vulkan": ti.vulkan}
    ti.init(arch=arch_map[args.arch], default_fp=ti.f32)

    data = load_sample(args.sample, garments_spec, n_body_frames=args.body_frames)

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
