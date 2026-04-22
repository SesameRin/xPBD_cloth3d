"""Per-fabric XPBD parameter presets.

XPBD compliance α has units of (length² / force) and converts to a Hookean
spring stiffness via the kernel's α / dt² term: smaller α ⇒ stiffer
constraint.

These values are aligned (loosely) with the C-IPC shell presets in
`cloth3d-ipc-xpbd/cloth3d_benchmark/cloth3d_sim/materials.py` so that the
two solvers rank the four fabrics the same way and have comparable
relative spreads on stretch / bending / mass. See `docs/fabric_presets.md`
for the alignment table and its (substantial) caveats.

Fields per preset:
    distance_compliance : in-plane stretch resistance (smaller = stiffer)
    bend_compliance     : opposite-vertex bending resistance (smaller = stiffer)
    damping             : per-substep velocity damping in [0, 1]
    density             : areal density in kg/m² (drives particle mass)
"""

FABRIC_PRESETS = {
    "cotton":  dict(distance_compliance=5.0e-9,  bend_compliance=1.0e-5,  damping=0.03, density=0.30),
    "silk":    dict(distance_compliance=2.0e-8,  bend_compliance=3.0e-5,  damping=0.01, density=0.10),
    "denim":   dict(distance_compliance=2.5e-9,  bend_compliance=3.75e-6, damping=0.05, density=0.70),
    "leather": dict(distance_compliance=1.2e-9,  bend_compliance=1.9e-6,  damping=0.08, density=1.25),
}

# Used when a fabric string is missing or unrecognised.
DEFAULT_FABRIC = "cotton"


def fabric_params(fabric_name):
    """Return the preset dict for `fabric_name`, falling back to cotton."""
    key = (fabric_name or "").lower().strip()
    return FABRIC_PRESETS.get(key, FABRIC_PRESETS[DEFAULT_FABRIC])
