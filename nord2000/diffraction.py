"""Wedge diffraction (Pierce's uniform asymptotic solution).

The wedge has two faces leaving the edge point E.  Angles are measured
clockwise from face 0 (source-side face) through the air to face 1, so
that the open (exterior) wedge angle is ``beta`` (2 pi for a thin
screen, slightly above pi for a gentle hill top) and nu = pi / beta.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import fresnel

from .geometry import cw_angle, dist, mirror, signed_distance, vec
from .ground import spherical_reflection

_PHASE = cmath.exp(1j * math.pi / 4.0) / math.sqrt(2.0)
_MIN_BETA_FACTOR = 1.002  # keep nu below 1 for nearly flat "edges"


def aux_fresnel(x: float) -> tuple[float, float]:
    """Auxiliary Fresnel functions f(x), g(x) for x >= 0."""
    if x > 5.0:
        x2 = (math.pi * x * x) ** 2
        f = (1.0 - 3.0 / x2) / (math.pi * x)
        g = (1.0 - 15.0 / x2) / (math.pi**2 * x**3)
        return f, g
    s, c = fresnel(x)
    arg = 0.5 * math.pi * x * x
    ca, sa = math.cos(arg), math.sin(arg)
    f = (0.5 - s) * ca - (0.5 - c) * sa
    g = (0.5 - c) * ca + (0.5 - s) * sa
    return float(f), float(g)


def diffraction_integral(x: float) -> complex:
    """Pierce's A_D(X) = sign(X) [f(|X|) - i g(|X|)]."""
    f, g = aux_fresnel(abs(x))
    sgn = 1.0 if x >= 0.0 else -1.0
    return sgn * complex(f, -g)


def m_nu(nu: float, theta: float) -> float:
    return (math.cos(nu * math.pi) - math.cos(nu * theta)) / (nu * math.sin(nu * math.pi))


@dataclass
class Edge:
    """A diffracting edge with its two wedge faces."""

    point: np.ndarray
    face0: np.ndarray          # direction from the edge along the source-side face
    face1: np.ndarray          # direction from the edge along the receiver-side face
    sigma0: float = math.inf   # flow resistivity of face 0 (inf = rigid)
    sigma1: float = math.inf
    left_index: int = -1       # profile vertex indices spanned by the edge
    right_index: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def beta(self) -> float:
        b = cw_angle(self.face0, self.face1)
        if b < 1e-9:
            b = 2.0 * math.pi
        return max(b, math.pi * _MIN_BETA_FACTOR)

    def angles(self, s, r) -> tuple[float, float, float]:
        """(theta_s, theta_r, beta) for source s and receiver r."""
        beta = self.beta
        e = self.point
        ts = cw_angle(self.face0, vec(s) - e)
        tr = cw_angle(self.face0, vec(r) - e)
        # Points slightly "inside" the wedge (numerical or geometric
        # approximations) are clamped onto the faces.
        if ts > beta:
            ts = 0.0 if ts > 0.5 * (beta + 2 * math.pi) else beta
        if tr > beta:
            tr = 0.0 if tr > 0.5 * (beta + 2 * math.pi) else beta
        return ts, tr, beta

    def face_points(self):
        e = self.point
        return (e, e + self.face0), (e, e + self.face1)


def _face_reflection(sigma, s, r, line_a, line_b, f, c, k):
    """Reflection coefficient and GO field for reflection in a wedge face line."""
    img = mirror(s, line_a, line_b)
    r2 = dist(img, r)
    ds = abs(signed_distance(s, line_a, line_b))
    dr = abs(signed_distance(r, line_a, line_b))
    sin_psi = min(1.0, (ds + dr) / max(r2, 1e-9))
    q = spherical_reflection(f, sigma, r2, sin_psi, c)
    return q, r2


def wedge_transfer(edge: Edge, s, r, f: float, c: float, *, rs=None, rr=None,
                   total_length=None, include_face_go=True):
    """Diffraction coefficient D and geometrical-optics parts for one wedge.

    Returns ``(D, go)`` where the diffracted field is
    exp(ikL)/L * D and ``go`` is a list of (amplitude, path_length)
    geometrical-optics rays (direct and face-reflected) that exist for
    this geometry.  ``rs``, ``rr`` and ``total_length`` override the
    distances used in Pierce's Gamma parameter (multiple-edge case).
    """
    k = 2.0 * math.pi * f / c
    e = edge.point
    ts, tr, beta = edge.angles(s, r)
    nu = math.pi / beta
    rs = dist(s, e) if rs is None else rs
    rr = dist(e, r) if rr is None else rr
    L = rs + rr if total_length is None else total_length
    gamma = math.sqrt(k * rs * rr / (math.pi * L))

    (a0, b0), (a1, b1) = edge.face_points()
    # face 0 line oriented so that the air side is to the left
    q0, r0 = _face_reflection(edge.sigma0, s, r, b0, a0, f, c, k)
    q1, r1 = _face_reflection(edge.sigma1, s, r, a1, b1, f, c, k)

    tp = tr + ts
    # use the reflection coefficient of the face whose reflection boundary is nearer
    qf = q0 if abs(tp - math.pi) <= abs(2.0 * beta - tp - math.pi) else q1

    x_minus = gamma * m_nu(nu, tr - ts)
    x_plus = gamma * m_nu(nu, tp)
    D = _PHASE * (diffraction_integral(x_minus) + qf * diffraction_integral(x_plus))

    go = []
    if tr - ts < math.pi:
        go.append((1.0 + 0j, dist(s, r)))
    if include_face_go:
        if tp < math.pi:
            go.append((q0, r0))
        if 2.0 * beta - tp < math.pi:
            go.append((q1, r1))
    return D, go


def edge_field(edges: list[Edge], s, r, f: float, c: float) -> tuple[complex, float]:
    """Complex pressure (exp(ikr)/r convention) at r via one or two edges.

    Returns ``(p, L)`` with L the path length over the edge(s).
    """
    k = 2.0 * math.pi * f / c
    if len(edges) == 1:
        e = edges[0]
        L = dist(s, e.point) + dist(e.point, r)
        D, go = wedge_transfer(e, s, r, f, c)
        p = cmath.exp(1j * k * L) / L * D
        for amp, length in go:
            p += amp * cmath.exp(1j * k * length) / length
        return p, L

    if len(edges) == 2:
        e1, e2 = edges
        r1 = dist(s, e1.point)
        r2 = dist(e1.point, e2.point)
        r3 = dist(e2.point, r)
        L = r1 + r2 + r3
        D1, go1 = wedge_transfer(e1, s, e2.point, f, c, rs=r1, rr=r2 + r3,
                                 total_length=L, include_face_go=False)
        D2, go2 = wedge_transfer(e2, e1.point, r, f, c, rs=r1 + r2, rr=r3,
                                 total_length=L, include_face_go=False)
        # Geometrical (lit) part of each edge, expressed relative to exp(ikL)/L.
        T1 = D1
        if go1:
            lp = dist(s, e2.point) + r3
            T1 += L / lp * cmath.exp(1j * k * (lp - L))
        T2 = D2
        if go2:
            lp = r1 + dist(e1.point, r)
            T2 += L / lp * cmath.exp(1j * k * (lp - L))
        return cmath.exp(1j * k * L) / L * T1 * T2, L

    raise ValueError("edge_field supports one or two edges")
