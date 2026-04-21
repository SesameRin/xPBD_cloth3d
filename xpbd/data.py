"""CLOTH3D loading and multi-garment merge.

Wraps the CLOTH3D `extract_sample_data` pipeline so the rest of the package
sees a single combined cloth mesh plus a per-vertex garment id, regardless
of whether the user asked for one garment or the whole outfit.
"""

import os
import sys

import numpy as np

# Add CLOTH3D readers to sys.path on import (path is fixed by repo layout).
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_ROOT, "cloth3d", "DataReader"))
sys.path.insert(0, os.path.join(_ROOT, "cloth3d", "Demo"))

from extract_sample_data import (  # noqa: E402
    extract_sample_single_frame,
    reader,
    get_num_frames,
)


def _parse_garment_list(spec, available):
    """Interpret --garments flag. Return list of garment names to simulate."""
    if spec is None or str(spec).lower() in ("all", "*"):
        return list(available)
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(
            f"garments {missing} not in sample (available: {list(available)})"
        )
    return names


def load_sample(
    sample,
    garments_spec=None,
    n_body_frames=1,
    need_gt_trajectory=False,
):
    """Load one CLOTH3D sample and merge requested garments into one cloth.

    Parameters
    ----------
    sample : str
        CLOTH3D sample id, e.g. "07414".
    garments_spec : str or None
        "all" / None -> every garment in the outfit;
        "Tshirt"     -> single garment;
        "Tshirt,Trousers" -> the named subset.
    n_body_frames : int
        How many SMPL body frames to load (1 = static collider).
    need_gt_trajectory : bool
        When True, also load per-garment ground-truth CLOTH3D vertex
        positions for every frame in 0..n_body_frames-1 and per-garment
        rest shapes. Required for the eval-compatible NPZ export.

    Returns
    -------
    dict with keys:
        V0        (N, 3)   combined initial vertex positions (float32)
        F         (M, 3)   combined triangle indices (int32)
        C         (N, 3)   per-vertex colors in [0, 1] (float32)
        vert_gid  (N,)     per-vertex garment index into garment_names
        garment_names    list of garment names actually included
        garment_fabrics  list of fabric strings (one per included garment)
        body_V_seq (T, 6890, 3)  SMPL body vertices (float32)
        body_F    (NBF, 3) SMPL face list
        sample    str
        cloth_frame_dt   float, CLOTH3D source framerate (1/30 s)
        # only when need_gt_trajectory=True:
        garment_V_rest   dict[name -> (N_g, 3) float32] rest shape per garment
        gt_V_by_garment  dict[name -> (T, N_g, 3) float32] CLOTH3D GT per frame
    """
    data0 = extract_sample_single_frame(
        sample, 0, use_uv_map=False, show_display=False
    )
    available = list(data0["garment_names"])
    names = _parse_garment_list(garments_spec, available)

    V_list, F_list, C_list, gid_list, fabrics = [], [], [], [], []
    V_rest_by_garment = {}
    v_offset = 0
    for gi, name in enumerate(names):
        key = f"garment_{name}"
        V = np.asarray(data0[f"{key}_V"], dtype=np.float32)
        F = np.asarray(data0[f"{key}_F"], dtype=np.int32)
        C = np.asarray(data0[f"{key}_C"], dtype=np.float32) / 255.0
        fab = str(data0[f"{key}_fabric"]) if f"{key}_fabric" in data0 else ""

        V_list.append(V)
        F_list.append(F + v_offset)
        C_list.append(C)
        gid_list.append(np.full(V.shape[0], gi, dtype=np.int32))
        fabrics.append(fab)
        if need_gt_trajectory and f"{key}_V_rest" in data0:
            V_rest_by_garment[name] = np.asarray(
                data0[f"{key}_V_rest"], dtype=np.float32
            )
        v_offset += V.shape[0]

    V0 = np.concatenate(V_list, axis=0).astype(np.float32)
    F = np.concatenate(F_list, axis=0).astype(np.int32)
    C = np.concatenate(C_list, axis=0).astype(np.float32)
    vert_gid = np.concatenate(gid_list, axis=0).astype(np.int32)

    total_frames = get_num_frames(sample)
    n_body_frames = max(1, min(n_body_frames, total_frames))
    body_V_seq = np.empty((n_body_frames, 6890, 3), dtype=np.float32)
    body_F = None
    for i in range(n_body_frames):
        Vb, Fb = reader.read_human(sample, i)
        body_V_seq[i] = Vb
        if body_F is None:
            body_F = np.asarray(Fb, dtype=np.int32)

    out = dict(
        V0=V0,
        F=F,
        C=C,
        vert_gid=vert_gid,
        garment_names=names,
        garment_fabrics=fabrics,
        body_V_seq=body_V_seq,
        body_F=body_F,
        sample=sample,
        cloth_frame_dt=1.0 / 30.0,
    )

    if need_gt_trajectory:
        gt_V_by_garment = {}
        for gi, name in enumerate(names):
            n_g = int((vert_gid == gi).sum())
            gt_seq = np.empty((n_body_frames, n_g, 3), dtype=np.float32)
            for fi in range(n_body_frames):
                Vg = reader.read_garment_vertices(sample, name, fi)
                gt_seq[fi] = np.asarray(Vg, dtype=np.float32)
            gt_V_by_garment[name] = gt_seq
        out["garment_V_rest"] = V_rest_by_garment
        out["gt_V_by_garment"] = gt_V_by_garment
        print(
            f"[data] loaded GT trajectory: "
            f"{ {n: v.shape for n, v in gt_V_by_garment.items()} }"
        )

    print(
        f"[data] sample={sample} garments={names} fabrics={fabrics} "
        f"cloth_V={V0.shape[0]} cloth_F={F.shape[0]} "
        f"body_V={body_V_seq.shape[1]} body_F={body_F.shape[0]} "
        f"frames={n_body_frames}"
    )
    return out
