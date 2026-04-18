"""Per-fabric XPBD parameter presets.

XPBD compliance α has units of (length² / force) and converts to a Hookean
spring stiffness via the kernel's α / dt² term: smaller α ⇒ stiffer
constraint. The values below are hand-tuned to give visually-plausible
drape for each CLOTH3D fabric tag, anchored on cotton (the material used
in this project's C-IPC comparison runs).

Fields per preset:
    distance_compliance : in-plane stretch resistance (smaller = stiffer)
    bend_compliance     : dihedral / bending resistance (smaller = stiffer)
    damping             : per-substep velocity damping in [0, 1]
    density             : areal density hint in kg/m² (drives particle mass)
"""

FABRIC_PRESETS = {
    "cotton":  dict(distance_compliance=5.0e-9,  bend_compliance=1.0e-5, damping=0.03, density=0.30),
    "silk":    dict(distance_compliance=2.0e-8,  bend_compliance=5.0e-7, damping=0.01, density=0.10),
    "denim":   dict(distance_compliance=1.0e-9,  bend_compliance=5.0e-5, damping=0.05, density=0.45),
    "leather": dict(distance_compliance=5.0e-10, bend_compliance=2.0e-4, damping=0.08, density=0.80),
}

# Used when a fabric string is missing or unrecognised.
DEFAULT_FABRIC = "cotton"


def fabric_params(fabric_name):
    """Return the preset dict for `fabric_name`, falling back to cotton."""
    key = (fabric_name or "").lower().strip()
    return FABRIC_PRESETS.get(key, FABRIC_PRESETS[DEFAULT_FABRIC])
