"""Mesh-side helpers: edges, bending pairs, normals, and per-vertex mass.

These are pure NumPy functions that run once at setup; the hot loop lives
in `xpbd.solver`.
"""

import numpy as np

from .fabrics import fabric_params


def build_edges(F):
    """Unique undirected edge list (E, 2) from a triangle list (M, 3)."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    return E.astype(np.int32)


def build_bending_pairs(F):
    """Return (M, 4) indices (v1, v2, v3, v4) for dihedral bending.

    v1, v2 form the shared edge between two triangles. v3, v4 are the
    opposite vertices of those two triangles. We use the classic PBD
    bending shortcut: a distance constraint between v3 and v4.
    """
    edge2tri = {}
    for ti, tri in enumerate(F):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (int(min(a, b)), int(max(a, b)))
            edge2tri.setdefault(e, []).append(ti)
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
    """Area-weighted per-vertex normals for collision pushout."""
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vn = np.zeros_like(V)
    np.add.at(vn, F[:, 0], fn)
    np.add.at(vn, F[:, 1], fn)
    np.add.at(vn, F[:, 2], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn.astype(np.float32)


def compute_vertex_masses(V, F, vert_gid, fabrics):
    """Areal-density-based per-vertex mass.

    Each triangle contributes one third of its (area · garment-density)
    to each of its three vertices. `vert_gid[i]` indexes `fabrics`, so a
    multi-garment outfit picks up the right density per region.
    """
    tri = V[F]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    densities = np.array(
        [fabric_params(f)["density"] for f in fabrics], dtype=np.float32
    )
    tri_density = densities[vert_gid[F[:, 0]]]
    tri_mass = area * tri_density
    mass = np.zeros(V.shape[0], dtype=np.float32)
    third = tri_mass / 3.0
    np.add.at(mass, F[:, 0], third)
    np.add.at(mass, F[:, 1], third)
    np.add.at(mass, F[:, 2], third)
    mass = np.maximum(mass, 1e-6)
    return mass.astype(np.float32)
