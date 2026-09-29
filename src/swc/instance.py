"""Problem instances for the daily selective collection problem.

Node indexing used everywhere:
    0            depot (trucks start here)
    1..n         bins
    n + 1        disposal / transfer station (every used truck unloads here, then returns to 0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Params, VehicleType

ZONE_TYPES = list(("industrial", "semi_industrial", "residential", "commercial",
                   "recreational", "political", "historical"))


@dataclass
class Instance:
    name: str
    xy_m: np.ndarray                 # (n+2, 2) coordinates in metres, rows ordered as nodes
    fill: np.ndarray                 # (n,) current fill fraction of each bin, 0..1(+)
    rate_mean_per_h: np.ndarray      # (n,) mean fill increment per hour (fraction of bin)
    rate_sd_per_h: np.ndarray        # (n,) sd of the hourly increment
    zone: list                       # (n,) zone type names
    vehicles: list                   # list[VehicleType], one entry per physical truck
    params: Params = field(default_factory=Params)
    dist_m: np.ndarray | None = None  # (n+2, n+2) road distances in metres (int)
    bins_per_point: np.ndarray | None = None  # (n,) number of identical bins at each collection point

    def __post_init__(self):
        if self.bins_per_point is None:
            self.bins_per_point = np.ones(len(self.fill), dtype=int)
        if self.dist_m is None:
            d = np.linalg.norm(self.xy_m[:, None, :] - self.xy_m[None, :, :], axis=2)
            self.dist_m = np.rint(d * self.params.ops.circuity).astype(np.int64)

    # --- convenience -----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.fill)

    @property
    def depot(self) -> int:
        return 0

    @property
    def disposal(self) -> int:
        return self.n + 1

    @property
    def bins(self) -> range:
        return range(1, self.n + 1)

    def demand_kg(self) -> np.ndarray:
        """Waste collected if a bin is emptied tonight (kg, integer), capped at the bin volume."""
        full = self.params.ops.bin_full_kg
        return np.rint(np.clip(self.fill, 0.0, 1.0) * full * self.bins_per_point).astype(np.int64)

    def service_s(self) -> np.ndarray:
        """Service time at each collection point (s): lift-and-tip time x number of containers."""
        return (self.params.ops.service_s_per_bin * self.bins_per_point).astype(np.int64)

    def travel_s(self) -> np.ndarray:
        v = self.params.ops.speed_kmh / 3.6
        return np.rint(self.dist_m / v).astype(np.int64)

    def sensitivity(self) -> np.ndarray:
        s = self.params.zone_sensitivity
        return np.array([s[z] for z in self.zone], dtype=float)

    def subset(self, bin_idx: list[int], name: str | None = None) -> "Instance":
        """Keep only the listed bins (1-based node indices)."""
        rows = [0] + list(bin_idx) + [self.n + 1]
        b = [i - 1 for i in bin_idx]
        return Instance(
            name=name or f"{self.name}_sub{len(bin_idx)}",
            xy_m=self.xy_m[rows], fill=self.fill[b], rate_mean_per_h=self.rate_mean_per_h[b],
            rate_sd_per_h=self.rate_sd_per_h[b], zone=[self.zone[i] for i in b],
            vehicles=list(self.vehicles), params=self.params,
            dist_m=self.dist_m[np.ix_(rows, rows)], bins_per_point=self.bins_per_point[b],
        )


def default_fleet(params: Params, counts: dict | None = None) -> list[VehicleType]:
    counts = counts or {"new_compactor": 1, "old_compactor": 2, "small_loader": 1}
    fleet = []
    for name, c in counts.items():
        fleet += [params.fleet_types[name]] * c
    return fleet


def _assign_zones(xy: np.ndarray, rng: np.random.Generator, n_centres: int = 8,
                  mix: dict | None = None) -> list[str]:
    """Voronoi-style zoning: random zone centres, each bin takes its nearest centre's type."""
    mix = mix or {"residential": 0.40, "commercial": 0.18, "semi_industrial": 0.10,
                  "industrial": 0.05, "recreational": 0.12, "political": 0.07, "historical": 0.08}
    types, probs = zip(*mix.items())
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    centres = rng.uniform(lo, hi, size=(n_centres, 2))
    ctype = rng.choice(types, size=n_centres, p=np.array(probs) / sum(probs))
    nearest = np.argmin(np.linalg.norm(xy[:, None] - centres[None], axis=2), axis=1)
    return [str(ctype[k]) for k in nearest]


def from_thesis_excel(path: str | Path, params: Params | None = None, seed: int = 0,
                      grid_unit_m: float = 50.0) -> Instance:
    """The 90-bin set used in the MSc thesis (synthetic grid).

    grid_unit_m: the thesis grid (-50..50) is interpreted as 50 m per unit, i.e. a
    5 km x 5 km area, roughly the size of Tehran District 6. Zones are assigned
    spatially (the thesis did not record a zone per bin). The depot is thesis node 20;
    the transfer station is placed outside the district, south-east.
    """
    import pandas as pd

    params = params or Params()
    raw = pd.read_excel(path, sheet_name="container", header=1).dropna(how="all", axis=1)
    raw.columns = ["id", "x", "y", "fill_pct", "rate_pct_h", "thesis_class"]
    raw = raw.dropna(subset=["id"]).astype({"id": int})
    depot_id = 20
    bins = raw[raw.id != depot_id]
    depot = raw[raw.id == depot_id][["x", "y"]].to_numpy(float)[0]
    disposal = np.array([70.0, -70.0])
    xy = np.vstack([depot, bins[["x", "y"]].to_numpy(float), disposal]) * grid_unit_m
    rng = np.random.default_rng(seed)
    rate = bins.rate_pct_h.to_numpy(float) / 100.0
    inst = Instance(
        name="thesis90",
        xy_m=xy,
        fill=bins.fill_pct.to_numpy(float) / 100.0,
        rate_mean_per_h=rate,
        rate_sd_per_h=rate * params.risk.fill_rate_cv * np.sqrt(24.0),
        zone=_assign_zones(xy[1:-1], rng),
        vehicles=default_fleet(params),
        params=params,
    )
    inst.bin_ids = bins.id.tolist()  # original thesis numbering, for traceability
    return inst


def generate_synthetic(n: int, seed: int, params: Params | None = None, layout: str = "clustered",
                       area_m: float = 5000.0, fleet_counts: dict | None = None,
                       bins_per_point: tuple[int, int] = (1, 1)) -> Instance:
    """Seeded synthetic instance family for the computational study.

    layout: 'uniform' or 'clustered' (bins around street-block centres).
    Fill levels are drawn so that roughly 25% are above 70%, 45% in 30-70% and 30% below 30%,
    mirroring the red/grey/green shares in the thesis data.
    bins_per_point: range of identical bins aggregated at one collection point (street segment),
    the aggregation the thesis itself used to keep instances tractable.
    """
    params = params or Params()
    rng = np.random.default_rng(seed)
    if layout == "uniform":
        pts = rng.uniform(0, area_m, size=(n, 2))
    else:
        k = max(3, n // 8)
        centres = rng.uniform(0.1 * area_m, 0.9 * area_m, size=(k, 2))
        lab = rng.integers(0, k, size=n)
        pts = np.clip(centres[lab] + rng.normal(0, 0.06 * area_m, size=(n, 2)), 0, area_m)
    depot = np.array([0.5 * area_m, 0.5 * area_m])
    disposal = np.array([1.4 * area_m, -0.4 * area_m])
    xy = np.vstack([depot, pts, disposal])
    band = rng.choice(3, size=n, p=[0.30, 0.45, 0.25])
    fill = np.where(band == 0, rng.uniform(0.0, 0.3, n),
                    np.where(band == 1, rng.uniform(0.3, 0.7, n), rng.uniform(0.7, 1.0, n)))
    zone = _assign_zones(pts, rng)
    # mean fill increment per hour (fraction of a bin); ~0.2-0.45 per day (ASSUMPTION until calibrated)
    base = {"commercial": 0.018, "recreational": 0.017, "residential": 0.014, "historical": 0.013,
            "political": 0.010, "semi_industrial": 0.009, "industrial": 0.008}
    rate = np.array([base[z] for z in zone]) * rng.lognormal(0.0, 0.25, n)
    return Instance(
        name=f"syn_{layout}_n{n}_s{seed}", xy_m=xy, fill=fill, rate_mean_per_h=rate,
        rate_sd_per_h=rate * params.risk.fill_rate_cv * np.sqrt(24.0), zone=zone,
        vehicles=default_fleet(params, fleet_counts), params=params,
        bins_per_point=rng.integers(bins_per_point[0], bins_per_point[1] + 1, size=n),
    )
