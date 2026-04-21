"""Write simulation results in the cloth3d_benchmark eval-compatible schema.

Produces per-garment `{sample}_{garment}_sim.npz` files that match the
layout written by
`cloth3d-ipc-xpbd/cloth3d_benchmark/cloth3d_xpbd/simulation.py` so they
can be consumed unchanged by
`cloth3d-ipc-xpbd/cloth3d_benchmark/eval_metrics.py`.

Coordinate frame notes:
- Our simulator runs in CLOTH3D's native z-up frame.
- The teammate's benchmark expects result positions in libuipc's y-up
  frame (the same frame `cloth3d_eval.io_source.convert_from_cloth3d`
  rotates CLOTH3D into via `rotate_x_minus_90`).
- `write_result_npz` therefore rotates every written vertex sequence
  (sim, gt, human) from z-up to y-up, and records
  `garment_y_translation = 0` so the eval source loader does not add an
  extra offset.

The companion helper `write_sample_npz` writes the per-sample extracted
CLOTH3D data in z-up (raw CLOTH3D frame), because the eval's source
loader does the rotation itself when reopening sample files.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

import numpy as np


# CLOTH3D is z-up, libuipc/eval is y-up. The teammate's converter uses
# a -90 deg rotation about X: (x, y, z) -> (x, z, -y). Reproduce the
# exact same mapping here so sim / gt / human all land in the same frame
# the eval pipeline expects.
def rotate_x_minus_90(vertices: np.ndarray) -> np.ndarray:
    """Rotate an (..., 3) vertex array by -90 deg about the x-axis.

    CLOTH3D z-up -> libuipc y-up: y' = z, z' = -y.
    """
    V = np.asarray(vertices)
    out = np.empty_like(V)
    out[..., 0] = V[..., 0]
    out[..., 1] = V[..., 2]
    out[..., 2] = -V[..., 1]
    return out


def slice_garment(
    F_full: np.ndarray,
    vert_gid: np.ndarray,
    gid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(vert_idx, F_local)` for one garment inside a merged mesh.

    `vert_idx` is an (N_g,) int32 array selecting that garment's rows out
    of the merged vertex array. `F_local` is (M_g, 3) int32 re-indexed
    into the 0..N_g-1 local space, so the returned face list is the one
    the benchmark stores for that garment.
    """
    vert_idx = np.where(vert_gid == gid)[0].astype(np.int32)
    if vert_idx.size == 0:
        return vert_idx, np.zeros((0, 3), dtype=np.int32)

    # Keep only faces whose three vertices all belong to this garment,
    # then remap indices into local 0..N_g-1 space.
    keep = np.all(np.isin(F_full, vert_idx), axis=1)
    F_sub = F_full[keep].astype(np.int32)
    remap = -np.ones(int(vert_gid.shape[0]), dtype=np.int32)
    remap[vert_idx] = np.arange(vert_idx.shape[0], dtype=np.int32)
    F_local = remap[F_sub]
    return vert_idx, F_local


def write_result_npz(
    *,
    out_dir: str,
    sample: str,
    garment_name: str,
    fabric: str,
    frame_dt: float,
    start_frame: int,
    requested_frames: int,
    faces_local: np.ndarray,
    sim_V_seq: np.ndarray,
    gt_V_seq: np.ndarray,
    human_faces: np.ndarray | None = None,
    human_V_seq: np.ndarray | None = None,
    frame_wall_time_ms: Sequence[float] | None = None,
    world_valid: bool = True,
    scene_mode: str = "both",
    ground_enabled: bool = False,
    effective_gravity: Iterable[float] = (0.0, -9.81, 0.0),
    cloth_reference_shape: str = "rest",
    solver_name: str = "xPBD-Taichi-CMU",
    extra: dict | None = None,
) -> str:
    """Write one eval-compatible `{sample}_{garment}_sim.npz` file.

    All vertex sequences (`sim_V_seq`, `gt_V_seq`, `human_V_seq`) must
    already be provided in CLOTH3D z-up; this function rotates them into
    y-up before writing. `faces_local` uses per-garment local indexing
    (0..N_g-1); that is how the teammate's eval reads it.

    Returns the absolute path of the written file.
    """
    run_dir = os.path.join(out_dir, f"{sample}_{garment_name}")
    os.makedirs(run_dir, exist_ok=True)

    sim_V_seq = np.asarray(sim_V_seq, dtype=np.float64)
    gt_V_seq = np.asarray(gt_V_seq, dtype=np.float64)
    sim_y = np.stack([rotate_x_minus_90(f) for f in sim_V_seq], axis=0)
    gt_y = np.stack([rotate_x_minus_90(f) for f in gt_V_seq], axis=0)

    simulated_frames = int(sim_y.shape[0])
    end_frame_exclusive = int(start_frame) + simulated_frames

    payload: dict[str, object] = {
        "result_format_version": 2,
        "solver_family": "xpbd",
        "solver_name": solver_name,
        "sample": str(sample),
        "garment_name": str(garment_name),
        "fabric": str(fabric),
        "frame_dt": float(frame_dt),
        "start_frame": int(start_frame),
        "end_frame_exclusive": int(end_frame_exclusive),
        "requested_frames": int(requested_frames),
        "simulated_frames": int(simulated_frames),
        "world_valid": bool(world_valid),
        "scene_mode": str(scene_mode),
        "ground_enabled": bool(ground_enabled),
        "wind_enabled": False,
        "wind_acceleration": np.zeros(3, dtype=np.float64),
        "effective_gravity": np.asarray(effective_gravity, dtype=np.float64),
        "cloth_reference_shape": str(cloth_reference_shape),
        "garment_y_translation": 0.0,
        "contact_tolerance": 0.0,
        "cloth_simulation_method": 0,
        "cloth_bending_method": 0,
        "faces": np.asarray(faces_local, dtype=np.int32),
        "sim_V_seq": sim_y,
        "gt_V_seq": gt_y,
        "frame_wall_time_ms_seq": np.asarray(
            list(frame_wall_time_ms or []), dtype=np.float64
        ),
        "time_per_frame_scope": "solver.step",
    }

    if human_faces is not None and human_V_seq is not None:
        human_V_seq = np.asarray(human_V_seq, dtype=np.float64)
        human_y = np.stack([rotate_x_minus_90(f) for f in human_V_seq], axis=0)
        payload["human_faces"] = np.asarray(human_faces, dtype=np.int32)
        payload["human_V_seq"] = human_y

    if extra:
        for k, v in extra.items():
            if k not in payload:
                payload[k] = v

    result_path = os.path.join(run_dir, f"{sample}_{garment_name}_sim.npz")
    np.savez_compressed(result_path, **payload)
    return result_path


def write_sample_npz(sample_npz_dir: str, sample: str) -> str:
    """Extract and save the per-sample CLOTH3D NPZ the eval reopens.

    The teammate's eval pipeline calls
    `cloth3d_sim.data.convert_from_cloth3d` which reads
    `{cloth3d_npz_path}/{sample}.npz`. That file has the full CLOTH3D
    schema (merged_*, human_V_seq, garment_<name>_V_seq, V_rest, E, etc.)
    and is produced by `cloth3d/Demo/extract_sample_data.py`. We call the
    same extractor here so the user can run the eval without having to
    separately re-extract in the teammate repo.
    """
    # Lazy import so the CLOTH3D reader only loads when we actually need
    # to extract — it pulls scipy/plotly/tqdm and owns a big sys.path.
    _HERE = os.path.abspath(os.path.dirname(__file__))
    _REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
    sys.path.insert(0, os.path.join(_REPO, "cloth3d", "DataReader"))
    sys.path.insert(0, os.path.join(_REPO, "cloth3d", "Demo"))
    from extract_sample_data import (  # noqa: E402
        extract_sample_all_frames,
        save_sample,
    )

    os.makedirs(sample_npz_dir, exist_ok=True)
    out = extract_sample_all_frames(sample, use_uv_map=False)
    save_sample(out, sample_npz_dir)
    return os.path.join(sample_npz_dir, f"{sample}.npz")
