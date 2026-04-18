"""XPBD cloth simulator on CLOTH3D — package entry.

Top-level re-exports for the most commonly used symbols. Individual
modules carry the implementation:

    xpbd.fabrics    - per-material XPBD parameter presets
    xpbd.geometry   - mesh utilities (edges, bending pairs, normals, mass)
    xpbd.data       - CLOTH3D loading + multi-garment merge
    xpbd.solver     - Taichi XPBDCloth class
    xpbd.viewers    - matplotlib / GGUI / headless runners
    xpbd.cli        - argparse entry point
"""

from .fabrics import FABRIC_PRESETS, DEFAULT_FABRIC, fabric_params
from .geometry import (
    build_edges,
    build_bending_pairs,
    per_vertex_normals,
    compute_vertex_masses,
)
from .data import load_sample
from .solver import XPBDCloth

__all__ = [
    "FABRIC_PRESETS",
    "DEFAULT_FABRIC",
    "fabric_params",
    "build_edges",
    "build_bending_pairs",
    "per_vertex_normals",
    "compute_vertex_masses",
    "load_sample",
    "XPBDCloth",
]
