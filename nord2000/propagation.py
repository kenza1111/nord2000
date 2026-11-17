"""Nord2000 point-to-point propagation over a vertical terrain profile.

The receiver level for a point source of sound power L_W is

    L_R = L_W + dL_d + dL_a + dL_t

with

* dL_d  geometrical divergence, -10 lg(4 pi R^2)
* dL_a  air absorption (ISO 9613-1) along the direct distance R
* dL_t  terrain effect (ground reflection, screening/diffraction and
        refraction) relative to free field, computed per 1/3-octave band.

Terrain effect
--------------
1. Refraction: the log-lin sound speed profile is replaced by an
   equivalent linear profile; the curved rays of that profile are
   straightened by an equivalent curvature of the terrain (downward
   refraction -> concave terrain, upward refraction -> convex terrain
   producing a shadow zone).
2. Screening edges are the vertices of the upper convex hull of the
   source, the terrain and the receiver.  More than two edges are reduced
   to two by merging adjacent edges into equivalent edges.
3. Unscreened propagation: direct ray plus a ground-reflected ray for
   every terrain segment, each weighted by the fraction of its Fresnel
   zone lying on the segment and using the spherical-wave reflection
   coefficient of the segment.
4. Screened propagation: wedge diffraction (Pierce) over one or two
   edges, combined with Fresnel-weighted ground reflections on the source
   side and on the receiver side (image sources / image receivers).
   Reflections in the wedge faces themselves are included in the wedge
   solution with the faces' reflection coefficients.
5. Edges slightly below the line of sight are handled by interpolating
   between the screened and unscreened solutions when the path difference
   is smaller than ``transition * wavelength``.
6. All contributions are added partially coherently using coherence
   coefficients for band averaging and atmospheric turbulence.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import Weather
from .bands import NOMINAL_FREQUENCIES, a_weighting, energetic_sum
from .coherence import Contribution, partially_coherent_level
from .diffraction import Edge, edge_field
from .geometry import (
    cross,
    cw_angle,
    dist,
    fresnel_zone_weight,
    line_intersection,
    path_difference,
    signed_distance,
    vec,
)
from .ground import spherical_reflection
from .terrain import TerrainProfile


@dataclass
class PropagationResult:
    """Per-band propagation terms (all in dB, arrays over ``frequencies``)."""

    frequencies: np.ndarray
    distance: float
    divergence: np.ndarray
    air_absorption: np.ndarray
    terrain: np.ndarray
    gradient: float
    n_edges: int

    @property
    def attenuation(self) -> np.ndarray:
        """Total propagation effect dL_d + dL_a + dL_t (to add to L_W)."""
        return self.divergence + self.air_absorption + self.terrain

    @property
    def excess_attenuation(self) -> np.ndarray:
        """Effect relative to free field without air absorption (= dL_t)."""
        return self.terrain

    def receiver_levels(self, sound_power) -> np.ndarray:
        """Receiver 1/3-octave levels for the source power spectrum ``sound_power``."""
        return np.asarray(sound_power, dtype=float) + self.attenuation

    def receiver_level_a(self, sound_power) -> float:
        """A-weighted total receiver level [dB(A)]."""
        lp = self.receiver_levels(sound_power) + a_weighting(self.frequencies)
        return float(energetic_sum(lp))


class Nord2000:
    """Nord2000 general outdoor sound propagation model.

    Parameters
    ----------
    weather : :class:`Weather`, default neutral (no refraction).
    frequencies : centre frequencies [Hz], default 25 Hz - 10 kHz 1/3 octaves.
    fresnel_fraction : Fresnel zones are defined by the extra path length
        wavelength / ``fresnel_fraction``.
    transition : edges below the line of sight with |path difference| <
        ``transition`` * wavelength are treated with a gradual transition.
    max_edges : maximum number of diffraction edges (Nord2000 uses 2).
    shadow_floor : level [dB re free field] added energetically to screened
        and shadow-zone results.  It is a simple stand-in for the energy
        scattered into shadow zones by turbulence (which Nord2000 models
        separately).  ``None`` disables it.
    """

    def __init__(self, weather: Weather | None = None, frequencies=None,
                 fresnel_fraction: float = 8.0, transition: float = 0.25,
                 max_edges: int = 2, shadow_floor: float | None = -30.0):
        self.weather = weather or Weather()
        self.frequencies = np.asarray(
            NOMINAL_FREQUENCIES if frequencies is None else frequencies, dtype=float
        )
        self.fresnel_fraction = fresnel_fraction
        self.transition = transition
        if max_edges not in (1, 2):
            raise ValueError("max_edges must be 1 or 2")
        self.max_edges = max_edges
        self.shadow_floor = shadow_floor

    # ----------------------------------------------------------------- public
    def propagate(self, profile: TerrainProfile, source_x: float, source_height: float,
                  receiver_x: float, receiver_height: float) -> PropagationResult:
        """Compute propagation from a source to a receiver along ``profile``.

        Heights are given above the local terrain.
        """
        if source_x == receiver_x:
            raise ValueError("source and receiver must be horizontally separated")
        if source_x > receiver_x:
            profile = _mirrored(profile)
            source_x, receiver_x = -source_x, -receiver_x

        w = self.weather
        c0 = w.c0
        xs, xr = float(source_x), float(receiver_x)
        S = np.array([xs, profile.height_at(xs) + source_height])
        R = np.array([xr, profile.height_at(xr) + receiver_height])
        d = xr - xs
        r_direct = dist(S, R)

        prof = profile.clipped(xs, xr)
        gradient = w.equivalent_gradient(source_height, receiver_height, d)
        if gradient != 0.0:
            radius = c0 / gradient
            prof = prof.refined(d / 50.0).curved(xs, xr, radius)

        geo = _Geometry(S, R, prof)
        edges = geo.hull_edges()
        while len(edges) > self.max_edges:
            edges = geo.merge_edges(edges)

        terrain = np.empty(len(self.frequencies))
        for n, f in enumerate(self.frequencies):
            terrain[n] = self._terrain_effect(geo, edges, f)

        divergence = np.full(len(self.frequencies), -10.0 * math.log10(4.0 * math.pi * r_direct**2))
        air = -w.absorption(self.frequencies) * r_direct
        return PropagationResult(self.frequencies.copy(), r_direct, divergence, air,
                                 terrain, gradient, len(edges))

    # --------------------------------------------------------------- internal
    def _level(self, geo: "_Geometry", contribs, f) -> float:
        w = self.weather
        return partially_coherent_level(contribs, f, w.c0, w.temperature_kelvin,
                                        w.Cv2, w.CT2, geo.S[0], geo.R[0])

    def _screened_level(self, geo, edges, f, c, lam) -> float:
        level = self._level(geo, geo.screened(edges, f, c, lam / self.fresnel_fraction), f)
        if self.shadow_floor is not None:
            level = 10.0 * math.log10(10.0 ** (level / 10.0) + 10.0 ** (self.shadow_floor / 10.0))
        return level

    def _terrain_effect(self, geo: "_Geometry", edges: list[Edge], f: float) -> float:
        c = self.weather.c0
        lam = c / f
        width = self.transition * lam

        if not edges:
            base = self._level(geo, geo.unscreened(f, c, lam / self.fresnel_fraction), f)
            cand = geo.best_candidate(geo.S, geo.R, 1, len(geo.points) - 2)
            if cand is None or -cand[1] >= width:
                return base
            t = -cand[1] / width
            screened = self._screened_level(geo, [cand[0]], f, c, lam)
            return (1.0 - t) * screened + t * base

        base = self._screened_level(geo, edges, f, c, lam)
        if len(edges) >= self.max_edges:
            return base
        e = edges[0]
        cands = [
            geo.best_candidate(geo.S, e.point, 1, e.left_index - 1),
            geo.best_candidate(e.point, geo.R, e.right_index + 1, len(geo.points) - 2),
        ]
        cands = [cd for cd in cands if cd is not None]
        if not cands:
            return base
        cand = max(cands, key=lambda cd: cd[1])
        if -cand[1] >= width:
            return base
        t = -cand[1] / width
        both = sorted([e, cand[0]], key=lambda ed: ed.point[0])
        screened2 = self._screened_level(geo, both, f, c, lam)
        return (1.0 - t) * screened2 + t * base


def _mirrored(profile: TerrainProfile) -> TerrainProfile:
    pts = profile.points[::-1].copy()
    pts[:, 0] = -pts[:, 0]
    new = TerrainProfile.__new__(TerrainProfile)
    new.points = pts
    new.sigma = profile.sigma[::-1].copy()
    return new


class _Geometry:
    """Geometry of one source-receiver pair over a (transformed) profile."""

    def __init__(self, S, R, prof: TerrainProfile):
        self.S = vec(S)
        self.R = vec(R)
        self.prof = prof
        self.points = prof.points
        self.segments = prof.segments

    # ------------------------------------------------------------- edges
    def _neighbour(self, i, step):
        j = i + step
        while 0 <= j < len(self.points) and np.allclose(self.points[j], self.points[i]):
            j += step
        return min(max(j, 0), len(self.points) - 1)

    def raw_beta(self, i) -> float:
        p = self.points
        a = p[self._neighbour(i, -1)] - p[i]
        b = p[self._neighbour(i, +1)] - p[i]
        beta = cw_angle(a, b)
        return 2.0 * math.pi if beta < 1e-9 else beta

    def edge_at(self, i) -> Edge:
        p = self.points
        s = self.prof.sigma
        return Edge(
            point=p[i].copy(),
            face0=p[self._neighbour(i, -1)] - p[i],
            face1=p[self._neighbour(i, +1)] - p[i],
            sigma0=float(s[i - 1]),
            sigma1=float(s[i]),
            left_index=i,
            right_index=i,
        )

    def hull_edges(self) -> list[Edge]:
        """Interior vertices of the upper convex hull of S, terrain and R."""
        chain = [(self.S, -1)]
        seq = [(self.points[i], i) for i in range(1, len(self.points) - 1)] + [(self.R, -2)]
        for p, idx in seq:
            while len(chain) >= 2 and cross(chain[-2][0], chain[-1][0], p) >= 0.0:
                chain.pop()
            chain.append((p, idx))
        return [self.edge_at(idx) for _, idx in chain[1:-1]]

    def merge_edges(self, edges: list[Edge]) -> list[Edge]:
        """Replace the adjacent pair of edges costing least path length by one edge."""
        chain = [self.S] + [e.point for e in edges] + [self.R]
        best = None
        for i in range(len(edges) - 1):
            prev, e1, e2, nxt = chain[i], edges[i], edges[i + 1], chain[i + 3]
            P = line_intersection(prev, e1.point, e2.point, nxt)
            if P is None or not (prev[0] <= P[0] <= nxt[0]):
                P = 0.5 * (e1.point + e2.point)
            cost = dist(prev, P) + dist(P, nxt) - (
                dist(prev, e1.point) + dist(e1.point, e2.point) + dist(e2.point, nxt))
            if best is None or cost < best[0]:
                best = (cost, i, P)
        _, i, P = best
        e1, e2 = edges[i], edges[i + 1]
        merged = Edge(point=P, face0=e1.face0, face1=e2.face1, sigma0=e1.sigma0,
                      sigma1=e2.sigma1, left_index=e1.left_index, right_index=e2.right_index)
        return edges[:i] + [merged] + edges[i + 2:]

    def best_candidate(self, A, B, i0, i1):
        """Convex terrain vertex in index range [i0, i1] closest to (below) line AB.

        Returns ``(edge, signed_path_difference)`` or None.
        """
        best = None
        for i in range(max(i0, 1), min(i1, len(self.points) - 2) + 1):
            p = self.points[i]
            if not (A[0] < p[0] < B[0]):
                continue
            if self.raw_beta(i) <= math.pi + 1e-4:
                continue
            delta = path_difference(A, p, B)
            if delta > 0.0:
                continue  # already a hull edge
            if best is None or delta > best[1]:
                best = (i, delta)
        if best is None:
            return None
        return self.edge_at(best[0]), best[1]

    # ------------------------------------------------------- contributions
    def unscreened(self, f, c, extra_path):
        S, R = self.S, self.R
        k = 2.0 * math.pi * f / c
        r1 = dist(S, R)
        refl = []
        for seg in self.segments:
            w, img, P = fresnel_zone_weight(S, R, seg.a, seg.b, extra_path)
            if w <= 0.0:
                continue
            r2 = dist(img, R)
            sin_psi = (signed_distance(S, seg.a, seg.b) + signed_distance(R, seg.a, seg.b)) / r2
            q = spherical_reflection(f, seg.sigma, r2, sin_psi, c)
            refl.append((w, q, r2, P))
        scale = _weight_scale(refl)
        contribs = [Contribution(1.0 + 0j, r1, [S, R])]
        for w, q, r2, P in refl:
            amp = scale * w * q * r1 / r2 * cmath.exp(1j * k * (r2 - r1))
            contribs.append(Contribution(amp, r2, [S, P, R]))
        return contribs

    def _side_images(self, origin, target, seg_range, f, c, extra_path):
        """Fresnel-weighted image points of ``origin`` for the sub-path origin-target."""
        out = []
        for j in seg_range:
            seg = self.segments[j]
            w, img, P = fresnel_zone_weight(origin, target, seg.a, seg.b, extra_path)
            if w <= 0.0:
                continue
            r2 = dist(img, target)
            sin_psi = (signed_distance(origin, seg.a, seg.b)
                       + signed_distance(target, seg.a, seg.b)) / r2
            q = spherical_reflection(f, seg.sigma, r2, sin_psi, c)
            out.append((w, q, r2, P, img))
        scale = _weight_scale(out)
        return [(img, scale * w * q, P) for w, q, _, P, img in out]

    def screened(self, edges: list[Edge], f, c, extra_path):
        S, R = self.S, self.R
        k = 2.0 * math.pi * f / c
        e_first, e_last = edges[0], edges[-1]
        nseg = len(self.segments)

        sources = [(S, 1.0 + 0j, [S])]
        for img, amp, P in self._side_images(S, e_first.point,
                                             range(0, max(e_first.left_index - 1, 0)),
                                             f, c, extra_path):
            sources.append((img, amp, [S, P]))
        receivers = [(R, 1.0 + 0j, [R])]
        for img, amp, P in self._side_images(R, e_last.point,
                                             range(e_last.right_index + 1, nseg),
                                             f, c, extra_path):
            receivers.append((img, amp, [P, R]))

        r1 = dist(S, R)
        p_free = cmath.exp(1j * k * r1) / r1
        edge_pts = [e.point for e in edges]
        contribs = []
        for sx, sa, spath in sources:
            for rx, ra, rpath in receivers:
                p, L = edge_field(edges, sx, rx, f, c)
                contribs.append(Contribution(sa * ra * p / p_free, L,
                                             spath + edge_pts + rpath))
        return contribs


def _weight_scale(items) -> float:
    total = sum(item[0] for item in items)
    return 1.0 / total if total > 1.0 else 1.0
