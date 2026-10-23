"""Ground incidence at the SAR scene center, not the spacecraft look angle.

SICD SCPCOA.IncidenceAng is the angle at the target between the line of
sight and the Earth-tangent-plane normal (degrees). The law-of-cosines
angle at the antenna reference point is the off-nadir / look angle and
must not be used as incidence.
"""
import numpy as np


def angles_rad(arp, scp, slant_range):
    """Return (look angle at spacecraft, ground incidence at target), radians.

    The law-of-cosines angle at the target is the angle between the downward
    vertical and the direction to the satellite, which is obtuse when the
    satellite is above the horizon. Ground incidence is the supplement:
    the angle between the upward normal and the line of sight.
    """
    arp = np.asarray(arp, dtype=float)
    scp = np.asarray(scp, dtype=float)
    arp_n = np.linalg.norm(arp)
    scp_n = np.linalg.norm(scp)
    look_cos = (arp_n**2 + slant_range**2 - scp_n**2) / (2 * arp_n * slant_range)
    # Interior angle at the target, between downward vertical and LOS.
    down_cos = (scp_n**2 + slant_range**2 - arp_n**2) / (2 * scp_n * slant_range)
    look = np.arccos(np.clip(look_cos, -1.0, 1.0))
    angle_from_down = np.arccos(np.clip(down_cos, -1.0, 1.0))
    incidence = np.pi - angle_from_down
    return look, incidence


def target_incidence_rad(meta, arp, scp, slant_range):
    """Prefer SICD metadata incidence. Fall back to the angle at the target.

    Returns (incidence_rad, source_label, look_angle_rad).
    look_angle_rad is diagnostic only.
    """
    look, target = angles_rad(arp, scp, slant_range)
    scpcoa = getattr(meta, "SCPCOA", None)
    inc_deg = getattr(scpcoa, "IncidenceAng", None) if scpcoa is not None else None
    if inc_deg is not None:
        return float(np.radians(float(inc_deg))), "SICD SCPCOA.IncidenceAng", look
    return (
        float(target),
        "geometric angle at target (SICD IncidenceAng absent)",
        look,
    )


def _self_check():
    """Synthetic triangle only. Not a scene measurement."""
    class _COA:
        IncidenceAng = 32.3

    class _Meta:
        SCPCOA = _COA()

    # Spherical-Earth sketch: satellite outside, target on the sphere.
    scp = np.array([6371000.0, 0.0, 0.0])
    arp = np.array([6371000.0 + 700000.0, 500000.0, 0.0])
    slant = np.linalg.norm(arp - scp)
    look, target = angles_rad(arp, scp, slant)
    if not (np.pi / 2 > target > look > 0):
        raise SystemExit(
            "expected 90° > ground incidence > spacecraft look angle > 0, "
            f"got incidence {np.degrees(target):.2f} look {np.degrees(look):.2f}"
        )
    inc, source, look2 = target_incidence_rad(_Meta(), arp, scp, slant)
    if source != "SICD SCPCOA.IncidenceAng":
        raise SystemExit(f"metadata incidence not preferred: {source}")
    if abs(np.degrees(inc) - 32.3) > 1e-9:
        raise SystemExit("metadata incidence was not used")
    if abs(look2 - look) > 1e-12:
        raise SystemExit("look-angle diagnostic drifted")

    class _Empty:
        SCPCOA = object()

    inc_fb, source_fb, _ = target_incidence_rad(_Empty(), arp, scp, slant)
    if "target" not in source_fb:
        raise SystemExit(f"fallback source unexpected: {source_fb}")
    if abs(inc_fb - target) > 1e-12:
        raise SystemExit("fallback is not the angle at the target")
    print("incidence self-check ok")
    print(f"  synthetic look {np.degrees(look):.2f} deg")
    print(f"  synthetic target incidence {np.degrees(target):.2f} deg")
    print("  (synthetic geometry only; not a Giza or Umbra measurement)")


if __name__ == "__main__":
    _self_check()
