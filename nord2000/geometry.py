"""2-D geometry helpers in the vertical (x, z) propagation plane."""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def vec(p) -> np.ndarray:
    return np.asarray(p, dtype=float)


def dist(a, b) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def cross(o, a, b) -> float:
    """z-component of (a - o) x (b - o)."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def angle(v) -> float:
    return math.atan2(v[1], v[0])


def cw_angle(a, b) -> float:
    """Clockwise angle in [0, 2 pi) rotating direction ``a`` onto ``b``."""
    return (angle(a) - angle(b)) % TWO_PI


def unit_normal(a, b) -> np.ndarray:
    """Unit normal of the directed segment a->b pointing to its left (air side)."""
    u = vec(b) - vec(a)
    n = np.array([-u[1], u[0]])
    return n / np.linalg.norm(n)


def signed_distance(p, a, b) -> float:
    """Signed distance of p from the line a->b (positive on the air side)."""
    return float(np.dot(vec(p) - vec(a), unit_normal(a, b)))


def mirror(p, a, b) -> np.ndarray:
    """Mirror point p in the line through a and b."""
    n = unit_normal(a, b)
    d = np.dot(vec(p) - vec(a), n)
    return vec(p) - 2.0 * d * n


def line_intersection(p1, p2, p3, p4):
    """Intersection of the infinite lines p1-p2 and p3-p4 (None if parallel)."""
    d1 = vec(p2) - vec(p1)
    d2 = vec(p4) - vec(p3)
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12 * (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-30):
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return vec(p1) + t * d1


def height_above_line(p, a, b) -> float:
    """Vertical height of p above the line through a and b (a.x != b.x)."""
    t = (p[0] - a[0]) / (b[0] - a[0])
    return float(p[1] - (a[1] + t * (b[1] - a[1])))


def path_difference(a, p, b) -> float:
    """Signed path difference |ap| + |pb| - |ab| (positive if p above line ab)."""
    delta = dist(a, p) + dist(p, b) - dist(a, b)
    return delta if height_above_line(p, a, b) >= 0.0 else -delta


def fresnel_zone_weight(s, r, a, b, extra_path: float):
    """Fresnel-zone weight of segment a-b for reflection from s to r.

    The Fresnel zone is the part of the (infinite) line through the
    segment where |sP| + |Pr| <= |s'r| + ``extra_path``, s' being the
    image of s.  The weight is the fraction of that interval covered by
    the segment.

    Returns ``(weight, image, specular_point)``; weight 0 if s or r is not
    on the air side of the segment.
    """
    a = vec(a)
    b = vec(b)
    seg_len = np.linalg.norm(b - a)
    if seg_len < 1e-9:
        return 0.0, None, None
    if signed_distance(s, a, b) <= 0.0 or signed_distance(r, a, b) <= 0.0:
        return 0.0, None, None
    s = vec(s)
    r = vec(r)
    image = mirror(s, a, b)
    u = (b - a) / seg_len

    p_spec = line_intersection(image, r, a, b)
    if p_spec is None:
        return 0.0, None, None

    # Ellipse with foci s, r and major axis |s'r| + extra_path.
    big_a = 0.5 * (dist(image, r) + extra_path)
    c_f = 0.5 * dist(s, r)
    big_b2 = big_a**2 - c_f**2
    if big_b2 <= 0.0:
        return 0.0, None, None
    centre = 0.5 * (s + r)
    if c_f > 1e-12:
        e1 = (r - s) / (2.0 * c_f)
    else:
        e1 = np.array([1.0, 0.0])
    e2 = np.array([-e1[1], e1[0]])
    q0 = a - centre
    x0, x1 = np.dot(q0, e1), np.dot(u, e1)
    y0, y1 = np.dot(q0, e2), np.dot(u, e2)
    qa = x1**2 / big_a**2 + y1**2 / big_b2
    qb = 2.0 * (x0 * x1 / big_a**2 + y0 * y1 / big_b2)
    qc = x0**2 / big_a**2 + y0**2 / big_b2 - 1.0
    disc = qb**2 - 4.0 * qa * qc
    if disc <= 0.0:
        return 0.0, image, p_spec
    sq = math.sqrt(disc)
    t1 = (-qb - sq) / (2.0 * qa)
    t2 = (-qb + sq) / (2.0 * qa)
    overlap = max(0.0, min(t2, seg_len) - max(t1, 0.0))
    return overlap / (t2 - t1), image, p_spec


def polyline_mean_separation(p1, p2, x0: float, x1: float, n: int = 16) -> float:
    """Mean vertical separation between two x-monotone polylines on [x0, x1]."""
    xs = np.linspace(x0, x1, n + 2)[1:-1]

    def interp(poly):
        poly = np.asarray(poly, dtype=float)
        order = np.argsort(poly[:, 0], kind="stable")
        return np.interp(xs, poly[order, 0], poly[order, 1])

    return float(np.mean(np.abs(interp(p1) - interp(p2))))
