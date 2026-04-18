"""Smoke tests for the XPBD CLOTH3D pipeline.

Runs as either:
    python3 -m pytest tests/ -q
    python3 tests/test_smoke.py
"""

import os
import sys

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import taichi as ti  # noqa: E402

import xpbd as xc  # noqa: E402

ti.init(arch=ti.cpu, default_fp=ti.f32, log_level=ti.WARN)


# ---------------------------------------------------------------------------
def test_fabric_presets_have_expected_keys():
    for fab in ("cotton", "silk", "denim", "leather"):
        p = xc.fabric_params(fab)
        for key in ("distance_compliance", "bend_compliance", "damping", "density"):
            assert key in p, f"{fab} missing {key}"
        assert p["distance_compliance"] > 0
        assert p["bend_compliance"] > 0
    # cotton is meaningfully stiffer than silk in stretch
    assert xc.fabric_params("cotton")["distance_compliance"] < \
           xc.fabric_params("silk")["distance_compliance"]
    # denim/leather bend more stiffly than cotton
    assert xc.fabric_params("denim")["bend_compliance"] > \
           xc.fabric_params("cotton")["bend_compliance"]


def test_unknown_fabric_falls_back_to_cotton():
    assert xc.fabric_params("unobtainium") == xc.fabric_params("cotton")
    assert xc.fabric_params(None) == xc.fabric_params("cotton")
    assert xc.fabric_params("") == xc.fabric_params("cotton")


# ---------------------------------------------------------------------------
def test_load_single_garment():
    data = xc.load_sample("00016", "Tshirt", n_body_frames=1)
    assert data["V0"].ndim == 2 and data["V0"].shape[1] == 3
    assert data["F"].ndim == 2 and data["F"].shape[1] == 3
    assert data["vert_gid"].shape[0] == data["V0"].shape[0]
    assert data["garment_names"] == ["Tshirt"]
    assert len(data["garment_fabrics"]) == 1
    assert data["body_V_seq"].shape == (1, 6890, 3)


def test_load_all_garments_07414_two_garments_one_mesh():
    data = xc.load_sample("07414", "all", n_body_frames=1)
    names = data["garment_names"]
    assert "Tshirt" in names and "Trousers" in names, f"got {names}"

    n_per = []
    for g in names:
        d = xc.load_sample("07414", g, n_body_frames=1)
        n_per.append(d["V0"].shape[0])
    assert data["V0"].shape[0] == sum(n_per), \
        "merged vertex count must equal sum of per-garment counts"

    # each vertex must be tagged to a valid garment id
    assert data["vert_gid"].min() == 0
    assert data["vert_gid"].max() == len(names) - 1

    # 07414 fabrics are all cotton in CLOTH3D
    assert all(f == "cotton" for f in data["garment_fabrics"]), \
        f"expected cotton fabrics, got {data['garment_fabrics']}"

    # Triangle indices stay in range of merged vertex array
    assert data["F"].max() < data["V0"].shape[0]
    assert data["F"].min() >= 0


def test_garment_subset_selection():
    data = xc.load_sample("00016", "Tshirt,Trousers", n_body_frames=1)
    assert sorted(data["garment_names"]) == ["Trousers", "Tshirt"]


def test_invalid_garment_raises():
    try:
        xc.load_sample("00016", "Hat", n_body_frames=1)
    except ValueError as e:
        assert "Hat" in str(e)
        return
    raise AssertionError("expected ValueError for unknown garment")


# ---------------------------------------------------------------------------
def test_edges_and_bending_pairs_are_consistent():
    data = xc.load_sample("00016", "Tshirt", n_body_frames=1)
    F = data["F"]
    E = xc.build_edges(F)
    BP = xc.build_bending_pairs(F)
    assert E.shape[1] == 2
    assert (E[:, 0] < E[:, 1]).all(), "edges should be sorted ascending"
    assert E.max() < data["V0"].shape[0]
    # bending pairs reference 4 distinct vertices each
    if BP.shape[0] > 0:
        assert BP.shape[1] == 4
        assert BP.max() < data["V0"].shape[0]


# ---------------------------------------------------------------------------
def test_one_xpbd_step_runs_and_advances_state():
    data = xc.load_sample("07414", "all", n_body_frames=1)
    cloth = xc.XPBDCloth(
        V0=data["V0"],
        F=data["F"],
        body_V0=data["body_V_seq"][0],
        body_F=data["body_F"],
        vert_gid=data["vert_gid"],
        garment_fabrics=data["garment_fabrics"],
        substeps=2,
        iterations=2,
    )
    cloth.set_color(data["C"])
    before = cloth.x.to_numpy().copy()
    cloth.step()
    after = cloth.x.to_numpy()
    assert np.isfinite(after).all(), "vertices went non-finite after one step"
    # gravity is on, so at least *some* vertices must have moved
    assert not np.allclose(before, after), "step did not change positions at all"


def test_force_fabric_rewrites_compliance():
    data = xc.load_sample("00016", "all", n_body_frames=1)
    forced = ["denim"] * len(data["garment_fabrics"])
    cloth_default = xc.XPBDCloth(
        V0=data["V0"], F=data["F"],
        body_V0=data["body_V_seq"][0], body_F=data["body_F"],
        vert_gid=data["vert_gid"],
        garment_fabrics=data["garment_fabrics"],
        substeps=1, iterations=1,
    )
    cloth_forced = xc.XPBDCloth(
        V0=data["V0"], F=data["F"],
        body_V0=data["body_V_seq"][0], body_F=data["body_F"],
        vert_gid=data["vert_gid"],
        garment_fabrics=forced,
        substeps=1, iterations=1,
    )
    a = cloth_default.dist_compliance.to_numpy()
    b = cloth_forced.dist_compliance.to_numpy()
    assert not np.allclose(a, b), "forcing fabric should change compliance values"


# ---------------------------------------------------------------------------
def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
