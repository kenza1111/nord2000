"""Vertical terrain profiles between a source and a receiver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ground import RIGID, flow_resistivity


@dataclass
class Segment:
    a: np.ndarray
    b: np.ndarray
    sigma: float  # flow resistivity [kPa s m^-2]; inf = acoustically hard

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


class TerrainProfile:
    """Piecewise-linear terrain profile in the vertical propagation plane.

    Parameters
    ----------
    points : sequence of (x, z) with non-decreasing x.  Equal consecutive x
        values describe vertical faces (screens, walls).
    ground : Nord2000 class letter ("A" ... "H"), flow resistivity
        [kPa s m^-2], ``None`` for rigid, or a list with one entry per
        segment.
    """

    def __init__(self, points, ground="D"):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("points must be a sequence of at least two (x, z) pairs")
        if np.any(np.diff(pts[:, 0]) < 0):
            raise ValueError("profile x coordinates must be non-decreasing")
        nseg = len(pts) - 1
        if isinstance(ground, (list, tuple, np.ndarray)):
            if len(ground) != nseg:
                raise ValueError("need one ground entry per segment")
            sig = [flow_resistivity(g) for g in ground]
        else:
            sig = [flow_resistivity(ground)] * nseg
        self.points = pts
        self.sigma = np.asarray(sig, dtype=float)

    # ------------------------------------------------------------------ build
    @classmethod
    def flat(cls, x0: float, x1: float, ground="D", z: float = 0.0):
        return cls([(x0, z), (x1, z)], ground)

    def _copy(self, points, sigma) -> "TerrainProfile":
        new = TerrainProfile.__new__(TerrainProfile)
        new.points = np.asarray(points, dtype=float)
        new.sigma = np.asarray(sigma, dtype=float)
        return new

    @property
    def segments(self) -> list[Segment]:
        return [
            Segment(self.points[i].copy(), self.points[i + 1].copy(), float(self.sigma[i]))
            for i in range(len(self.points) - 1)
        ]

    def height_at(self, x: float) -> float:
        """Ground height at x (highest value at a vertical face)."""
        p = self.points
        if x <= p[0, 0]:
            return float(p[0, 1])
        if x >= p[-1, 0]:
            return float(p[-1, 1])
        zs = []
        for i in range(len(p) - 1):
            x0, x1 = p[i, 0], p[i + 1, 0]
            if x0 <= x <= x1:
                if x1 > x0:
                    t = (x - x0) / (x1 - x0)
                    zs.append(p[i, 1] + t * (p[i + 1, 1] - p[i, 1]))
                else:
                    zs.extend([p[i, 1], p[i + 1, 1]])
        return float(max(zs))

    def _split_at(self, x: float):
        """Return (points, sigma, index) with a vertex inserted at x."""
        p = list(map(tuple, self.points))
        s = list(self.sigma)
        for i in range(len(p) - 1):
            if p[i][0] <= x <= p[i + 1][0] and p[i + 1][0] > p[i][0]:
                if x == p[i][0]:
                    return p, s, i
                if x == p[i + 1][0]:
                    return p, s, i + 1
                t = (x - p[i][0]) / (p[i + 1][0] - p[i][0])
                z = p[i][1] + t * (p[i + 1][1] - p[i][1])
                p.insert(i + 1, (x, z))
                s.insert(i, s[i])
                return p, s, i + 1
        raise ValueError(f"x={x} outside the profile")

    def with_screen(self, x: float, height: float, thickness: float = 0.0,
                    ground=None) -> "TerrainProfile":
        """Return a copy with a screen (thin or thick barrier) at x.

        ``height`` is measured from the local ground; ``ground`` is the
        surface of the screen (``None`` = rigid).
        """
        sig = flow_resistivity(ground)
        p, s, i = self._split_at(x)
        zg = p[i][1]
        top = zg + height
        if thickness <= 0.0:
            p[i + 1:i + 1] = [(x, top), (x, zg)]
            s[i:i] = [sig, sig]
            return self._copy(p, s)
        prof = self._copy(p, s)
        p, s, j = prof._split_at(x + thickness)
        zg2 = p[j][1]
        new_p = p[:i + 1] + [(x, top), (x + thickness, top), (x + thickness, zg2)] + p[j + 1:]
        new_s = s[:i] + [sig, sig, sig] + s[j:]
        return self._copy(new_p, new_s)

    def clipped(self, x0: float, x1: float) -> "TerrainProfile":
        """Profile restricted to x0 <= x <= x1."""
        p = self.points
        if x0 < p[0, 0] or x1 > p[-1, 0]:
            raise ValueError("source/receiver outside the terrain profile")
        pts, sig = [(x0, self.height_at(x0))], []
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            if b[0] <= x0 or a[0] >= x1:
                continue
            sig.append(self.sigma[i])
            if b[0] < x1:
                pts.append(tuple(b))
        pts.append((x1, self.height_at(x1)))
        if len(sig) != len(pts) - 1:  # degenerate (single vertical edge etc.)
            sig = (sig + [self.sigma[-1]] * len(pts))[: len(pts) - 1]
        return self._copy(pts, sig)

    def refined(self, max_length: float) -> "TerrainProfile":
        """Subdivide segments longer than ``max_length``."""
        pts, sig = [tuple(self.points[0])], []
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_length)))
            for k in range(1, n + 1):
                pts.append(tuple(a + (b - a) * k / n))
                sig.append(self.sigma[i])
        return self._copy(pts, sig)

    def curved(self, xs: float, xr: float, radius: float) -> "TerrainProfile":
        """Transform the profile so that circular rays of ``radius`` become straight.

        A ray between x=xs and x=xr with radius of curvature ``radius``
        (positive = bending downwards) deviates from the straight line by
        (x - xs)(xr - x) / (2 radius).  Subtracting this from the terrain
        lets the propagation be computed with straight rays.
        """
        p = self.points.copy()
        x = p[:, 0]
        p[:, 1] = p[:, 1] - (x - xs) * (xr - x) / (2.0 * radius)
        return self._copy(p, self.sigma.copy())


__all__ = ["TerrainProfile", "Segment", "RIGID"]
