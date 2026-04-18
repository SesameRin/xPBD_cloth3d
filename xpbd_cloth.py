"""Thin entry point preserved for back-compat.

The implementation now lives in the `xpbd` package:

    xpbd.fabrics    - per-material XPBD parameter presets
    xpbd.geometry   - mesh utilities (edges, bending pairs, normals, mass)
    xpbd.data       - CLOTH3D loading + multi-garment merge
    xpbd.solver     - Taichi XPBDCloth class
    xpbd.viewers    - matplotlib / GGUI / headless runners
    xpbd.cli        - argparse entry point

See `docs/architecture.md` for the full module map.

Run:
    python3 xpbd_cloth.py --sample 07414 --garments all --viewer mpl --save_video
    python3 -m xpbd                                     # equivalent
"""

from xpbd.cli import main

if __name__ == "__main__":
    main()
