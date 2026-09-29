"""Tehran District 6 case study built from OpenStreetMap (ODbL, (c) OpenStreetMap contributors).

Construction (every step is a documented assumption; see TEHRAN_ASSUMPTIONS):
1. District boundary: OSM administrative boundary 'District 6, Tehran' (21.4 km2).
2. Road network: OSM 'drive' network clipped to the boundary, largest strongly connected
   component (one-way streets respected; distances are shortest paths in metres).
3. Bins: ONE 1100 L bin per alley, i.e. per unique undirected segment of type
   residential / living_street / unclassified (the Tehran 'kucheh').
4. Zones from OSM tags, in priority order: historical (museum/historic within 100 m),
   political (diplomatic/government within 60 m), recreational (park within 50 m),
   commercial (commercial/retail land use), industrial, else residential.
5. Waste generation: district population x per-capita generation x non-resident uplift, spread
   over bins with zone weights and lognormal heterogeneity.
6. Collection points: bins clustered (k-means) into K points (the thesis's own aggregation),
   each snapped to the nearest network node.
7. Depot: District 6 municipality building (OSM townhall). Transfer station: no Tehran transfer
   station is mapped in OSM, so it is modelled as the district's southern arterial exit plus a
   fixed off-network leg of `transfer_leg_km` each way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import MODEL_VERSION, Params, VehicleType
from .instance import Instance

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "osm_cache"
OVERPASS_MAIN = "https://overpass-api.de/api"          # network was first downloaded (and cached) here
OVERPASS = "https://overpass.private.coffee/api"       # mirror used for feature queries
CRS = 32639  # UTM 39N

# Low-cost global benchmark fleet (USD per shift / per km). ASSUMPTIONS to verify:
# crew: driver + 2 loaders at about 3 USD/h x 8 h; diesel 0.9 USD/L; maintenance 0.10-0.40 USD/km;
# new compactor about 120 kUSD over 10 years x 300 shifts (plus financing) -> about 40 USD/shift.
TEHRAN_FLEET = {
    "new_compactor": VehicleType("new_compactor", 7000, 112.0, 0.65, 0.45, 0.65),
    "old_compactor": VehicleType("old_compactor", 6000, 80.0, 1.12, 0.65, 0.95),
    "small_loader": VehicleType("small_loader", 2500, 63.0, 0.42, 0.30, 0.42),
}


@dataclass
class TehranAssumptions:
    population: float = 250_000          # District 6 residents: 251,384 at the 2016 census, rounded (sensitivity +/-30%)
    kg_per_capita_day: float = 0.85      # Tehran MSW generation per capita (verify; sensitivity)
    nonresident_uplift: float = 1.30     # offices, universities, commerce: large daytime population
    bin_volume_m3: float = 1.1
    n_points: int = 300
    transfer_leg_km: float = 6.0         # off-network distance exit -> transfer station (sensitivity 3-12 km)
    zone_weight: dict = field(default_factory=lambda: {
        "commercial": 1.3, "recreational": 1.2, "residential": 1.0, "historical": 1.0,
        "political": 0.7, "semi_industrial": 0.6, "industrial": 0.5})
    fleet_counts: dict = field(default_factory=lambda: {"new_compactor": 8, "old_compactor": 18, "small_loader": 8})
    seed: int = 0


def _ox():
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CACHE)
    ox.settings.overpass_url = OVERPASS_MAIN
    ox.settings.overpass_rate_limit = False   # skip the /status pause check (it times out on mirrors)
    ox.settings.requests_timeout = 300
    ox.settings.log_console = False
    return ox


def _features(ox, poly, tags):
    try:
        return ox.features_from_polygon(poly, tags).to_crs(CRS)
    except Exception:
        return None


def build(a: TehranAssumptions | None = None, params: Params | None = None, save: bool = True) -> Instance:
    import networkx as nx
    from shapely.geometry import Point
    from sklearn.cluster import KMeans

    a = a or TehranAssumptions()
    ox = _ox()
    boundary = ox.geocode_to_gdf("District 6, Tehran, Iran")
    poly = boundary.geometry.iloc[0]
    G = ox.graph_from_polygon(poly, network_type="drive", simplify=True, truncate_by_edge=True)
    G = ox.truncate.largest_component(G, strongly=True)
    Gp = ox.project_graph(G, to_crs=CRS)
    nodes, edges = ox.graph_to_gdfs(Gp)
    ox.settings.overpass_url = OVERPASS

    # --- 3. one bin per alley (unique undirected segment) -------------------------------
    hw = edges["highway"].apply(lambda v: v if isinstance(v, str) else v[0])
    alley = edges[hw.isin(["residential", "living_street", "unclassified"])].reset_index()
    alley["key2"] = [tuple(sorted((u, v))) for u, v in zip(alley["u"], alley["v"])]
    alley = alley.drop_duplicates("key2")
    mid = alley.geometry.interpolate(0.5, normalized=True)
    bx, by = mid.x.to_numpy(), mid.y.to_numpy()
    nb = len(bx)

    # --- 4. zones ----------------------------------------------------------------------
    def near(gdf, r):
        if gdf is None or gdf.empty:
            return np.zeros(nb, bool)
        geom = gdf.geometry.buffer(r).union_all()
        return np.array([geom.contains(Point(x, y)) for x, y in zip(bx, by)])

    hist = _features(ox, poly, {"tourism": "museum", "historic": True})
    polit = _features(ox, poly, {"office": ["diplomatic", "government"], "amenity": "embassy"})
    park = _features(ox, poly, {"leisure": "park"})
    comm = _features(ox, poly, {"landuse": ["commercial", "retail"]})
    ind = _features(ox, poly, {"landuse": "industrial"})
    zone = np.array(["residential"] * nb, dtype=object)
    for z, mask in [("industrial", near(ind, 0)), ("commercial", near(comm, 0)), ("recreational", near(park, 50)),
                    ("political", near(polit, 60)), ("historical", near(hist, 100))]:
        zone[mask] = z  # later assignments have priority

    # --- 5. waste generation per bin (fraction of a bin per day) -----------------------------
    rng = np.random.default_rng(a.seed)
    total_kg = a.population * a.kg_per_capita_day * a.nonresident_uplift
    w = np.array([a.zone_weight[z] for z in zone]) * rng.lognormal(0.0, 0.25, nb)
    kg_bin = total_kg * w / w.sum()
    full_kg = (params or Params()).ops.waste_density_kg_m3 * a.bin_volume_m3
    frac_day = kg_bin / full_kg

    # --- 6. collection points ------------------------------------------------------------------
    km = KMeans(n_clusters=a.n_points, n_init=4, random_state=a.seed).fit(np.c_[bx, by])
    lab = km.labels_
    cent = km.cluster_centers_
    node_ids = ox.distance.nearest_nodes(Gp, cent[:, 0], cent[:, 1])
    m = np.bincount(lab, minlength=a.n_points)
    rate_day = np.array([frac_day[lab == k].mean() for k in range(a.n_points)])
    sens_order = ["industrial", "semi_industrial", "residential", "commercial", "recreational", "political", "historical"]
    pzone = []
    for k in range(a.n_points):
        zs, cnt = np.unique(zone[lab == k], return_counts=True)
        best = zs[cnt == cnt.max()]
        pzone.append(max(best, key=sens_order.index))

    # --- 7. depot, transfer station, distances --------------------------------------------
    th = _features(ox, poly, {"amenity": "townhall"})
    if th is not None and not th.empty:
        named = th[th.get("name", "").astype(str).str.contains("شهرداری منطقه")]
        g = (named if not named.empty else th).geometry.centroid.iloc[0]
        depot_node = ox.distance.nearest_nodes(Gp, g.x, g.y)
    else:
        depot_node = ox.distance.nearest_nodes(Gp, nodes.x.mean(), nodes.y.mean())
    art = edges[hw.isin(["trunk", "primary"])].reset_index()
    ends = np.unique(np.r_[art["u"].to_numpy(), art["v"].to_numpy()])
    exit_node = ends[np.argmin(nodes.loc[ends, "y"].to_numpy())]

    route_nodes = [depot_node] + list(node_ids) + [exit_node]
    N = len(route_nodes)
    D = np.zeros((N, N), dtype=np.int64)
    for i, s in enumerate(route_nodes):
        lengths = nx.single_source_dijkstra_path_length(Gp, s, weight="length")
        D[i] = [int(round(lengths[t])) for t in route_nodes]
    leg = int(a.transfer_leg_km * 1000)
    D[:, -1] += leg          # to the transfer station
    D[-1, :] += leg          # from the transfer station
    D[-1, -1] = 0
    xy = np.c_[nodes.loc[route_nodes, "x"].to_numpy(), nodes.loc[route_nodes, "y"].to_numpy()]

    params = params or Params()
    params.fleet_types = dict(TEHRAN_FLEET)
    params.ops = type(params.ops)(**{**params.ops.__dict__, "bin_volume_m3": a.bin_volume_m3})
    vehicles = []
    for name, c in a.fleet_counts.items():
        vehicles += [params.fleet_types[name]] * c
    rate_h = rate_day / 24.0
    fill0 = np.clip(rng.uniform(0.2, 1.0, a.n_points) * rate_day * 2, 0, 1)   # start mid-cycle
    inst = Instance(name="tehran_d6", xy_m=xy, fill=fill0, rate_mean_per_h=rate_h,
                    rate_sd_per_h=rate_h * params.risk.fill_rate_cv * np.sqrt(24.0), zone=list(pzone),
                    vehicles=vehicles, params=params, dist_m=D, bins_per_point=m)
    inst.meta = {"bins": int(nb), "points": a.n_points, "total_t_day": total_kg / 1000,
                 "days_to_full_mean": float(np.mean(1 / np.maximum(frac_day, 1e-9))),
                 "bin_fill_per_day_mean": float(frac_day.mean()),
                 "zones_bins": {z: int((zone == z).sum()) for z in set(zone)},
                 "network_nodes": len(Gp), "alley_km": float(alley["length"].sum() / 1000),
                 "assumptions": {k: v for k, v in a.__dict__.items()}}
    if save:
        out = ROOT / "data" / "processed"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "tehran_d6.npz", xy=xy, dist=D, fill0=fill0, rate_day=rate_day, m=m,
                            zone=np.array(pzone), bins_xy=np.c_[bx, by], bins_zone=zone.astype(str),
                            bins_label=lab)
        (out / "tehran_d6_meta.json").write_text(json.dumps(inst.meta, indent=1, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
    return inst


# Instance version used by the article's Tehran experiments. v2 splits large collection points
# (see `load`); results of v1 (unsplit, 300 points) are kept in results/ folders without the tag.
INSTANCE_VERSION = MODEL_VERSION
MAX_BINS_PER_POINT = 15   # 15 bins x 1.1 m3 x 200 kg/m3 x 2 (doubled capacity) = 6.6 t < 7 t payload


def _split_points(z: dict, max_bins: int) -> dict:
    """Split collection points with more than `max_bins` bins into co-located sub-points.

    The model serves each point with one truck (no split collection). With doubled containers a
    point of 16+ bins can hold more than the largest payload (7 t) and could then never be
    collected. Co-located sub-points (zero distance between them) let two trucks share it.
    """
    m = z["m"]
    idx, sizes = [], []
    for k, mk in enumerate(m):
        parts = int(np.ceil(mk / max_bins))
        base, extra = divmod(int(mk), parts)
        for p in range(parts):
            idx.append(k)
            sizes.append(base + (1 if p < extra else 0))
    idx = np.array(idx)
    rows = np.r_[0, idx + 1, len(m) + 1]
    return dict(dist=z["dist"][np.ix_(rows, rows)], xy=z["xy"][rows], fill0=z["fill0"][idx],
                rate_day=z["rate_day"][idx], zone=z["zone"][idx], m=np.array(sizes))


def load(params: Params | None = None, fleet_counts: dict | None = None, rate_mult: float = 1.0,
         transfer_leg_km: float | None = None, max_bins_per_point: int | None = MAX_BINS_PER_POINT) -> Instance:
    """Rebuild the Instance from the saved arrays (no network access).

    rate_mult scales waste generation (population / per-capita sensitivity);
    transfer_leg_km overrides the off-network distance to the transfer station (built with 6 km);
    max_bins_per_point splits large points into co-located sub-points (None = original 300 points).
    """
    z = dict(np.load(ROOT / "data" / "processed" / "tehran_d6.npz", allow_pickle=True))
    if max_bins_per_point is not None:
        z = _split_points(z, max_bins_per_point)
    dist = z["dist"].copy()
    if transfer_leg_km is not None:
        delta = int(round((transfer_leg_km - TehranAssumptions().transfer_leg_km) * 1000))
        dist[:-1, -1] += delta
        dist[-1, :-1] += delta
    params = params or Params()
    params.fleet_types = dict(TEHRAN_FLEET)
    counts = fleet_counts or TehranAssumptions().fleet_counts
    vehicles = []
    for name, c in counts.items():
        vehicles += [params.fleet_types[name]] * c
    rate_h = z["rate_day"] * rate_mult / 24.0
    return Instance(name="tehran_d6", xy_m=z["xy"], fill=z["fill0"].copy(), rate_mean_per_h=rate_h,
                    rate_sd_per_h=rate_h * params.risk.fill_rate_cv * np.sqrt(24.0), zone=list(z["zone"]),
                    vehicles=vehicles, params=params, dist_m=dist, bins_per_point=z["m"])
