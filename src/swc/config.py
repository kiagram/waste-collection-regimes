"""Model parameters.

Every number here is a documented ASSUMPTION until replaced with Tehran data
(fleet records, fuel logs, municipal cost sheets, the BWM expert survey).
Keep the source/justification column up to date: reviewers will ask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Version of the article experiments. v2: service time per container (60 s x m_i) and large Tehran
# collection points split into co-located sub-points. v3: v2 + routing engine with a greedy warm
# start, truck-type swaps and best of 2 runs x 5,000 iterations (the 1,500-iteration cold-start engine
# of v1/v2 left 5-39% on the table at district scale; see scripts/17_heuristic_quality.py).
MODEL_VERSION = "v3"

DIESEL_CO2_KG_PER_L = 2.68  # IPCC 2006 default for diesel combustion (approx.)


@dataclass(frozen=True)
class VehicleType:
    name: str
    capacity_kg: int            # legal payload
    fixed_cost: float           # cost units per shift if used (driver+crew, depreciation)
    var_cost_per_km: float      # cost units per km (fuel, tyres, maintenance)
    fuel_empty_l_per_km: float  # stop-and-go collection fuel use, empty
    fuel_full_l_per_km: float   # same, at full payload (linear in load in between)
    co2_kg_per_l: float = DIESEL_CO2_KG_PER_L

    # Load-dependent emissions: e(load) = a + b * load   [g CO2 per metre]
    @property
    def a_g_per_m(self) -> float:
        return self.fuel_empty_l_per_km * self.co2_kg_per_l  # kg/km == g/m

    @property
    def b_g_per_m_per_kg(self) -> float:
        slope_l_per_km_per_kg = (self.fuel_full_l_per_km - self.fuel_empty_l_per_km) / self.capacity_kg
        return slope_l_per_km_per_kg * self.co2_kg_per_l


# ASSUMPTION: three truck classes typical of an ageing megacity fleet. The key economic
# feature: old trucks are depreciated (low fixed cost per shift) but burn more fuel and need
# more maintenance; new Euro-V trucks carry lease/depreciation cost but emit less.
DEFAULT_FLEET_TYPES = {
    "new_compactor": VehicleType("new_compactor", 7000, 150.0, 1.00, 0.45, 0.65),
    "old_compactor": VehicleType("old_compactor", 6000, 60.0, 1.35, 0.65, 0.95),
    "small_loader": VehicleType("small_loader", 2500, 55.0, 0.80, 0.30, 0.42),
}


@dataclass(frozen=True)
class OpsParams:
    speed_kmh: float = 25.0             # night-shift average urban speed (ASSUMPTION)
    service_s_per_bin: int = 60         # lift-and-tip time per 1100 L container; a point takes this x m_i (ASSUMPTION)
    unload_s: int = 1200                # unloading at the transfer station (ASSUMPTION)
    shift_s: int = 6 * 3600             # 22:00-04:00 night shift (ASSUMPTION)
    max_trips: int = 2                  # trips to the transfer station per truck and shift (ASSUMPTION)
    bin_volume_m3: float = 1.1          # 1100 L wheeled container
    waste_density_kg_m3: float = 200.0  # uncompacted MSW in bins (ASSUMPTION; Tehran is organic-rich)
    circuity: float = 1.30              # road / Euclidean distance ratio until OSM distances are used

    @property
    def bin_full_kg(self) -> float:
        return self.bin_volume_m3 * self.waste_density_kg_m3


@dataclass(frozen=True)
class RiskParams:
    """Overflow-risk valuation.

    penalty_i = overflow_cost * zone_sensitivity_i * risk_i
    risk_i is P(overflow before the next opportunity) ('prob') or the expected
    overflowing fraction of a bin ('excess').
    """
    overflow_cost: float = 40.0   # cost units per overflow event in the most sensitive zone (sensitivity analysis!)
    horizon_h: float = 24.0       # time to the next collection opportunity if skipped tonight
    fill_rate_cv: float = 0.35    # CV of DAILY fill increments (placeholder until Wyndham calibration)
    mode: str = "count"           # penalty risk measure: 'count' | 'prob' | 'excess'
    # Zone-dependent chance constraint: bin i MUST be collected tonight if
    # P(overflow before next opportunity) > alpha_i, with alpha_i interpolated between
    # alpha_max (least sensitive zone) and alpha_min (most sensitive zone).
    alpha_min: float = 0.05
    alpha_max: float = 0.50


# PLACEHOLDER zone sensitivities in [0, 1]. They must be replaced by the
# second-level Bayesian BWM weights from the new expert survey.
DEFAULT_ZONE_SENSITIVITY = {
    "industrial": 0.15,
    "semi_industrial": 0.25,
    "residential": 0.55,
    "commercial": 0.65,
    "recreational": 0.85,
    "political": 0.90,
    "historical": 1.00,
}


@dataclass
class ScaleParams:
    """Integer scaling used inside CP-SAT (floats are recomputed exactly by the checker)."""
    # Fine enough that per-kg load terms on short arcs round with <0.5% error.
    cost_scale: int = 10_000   # cost units -> 1e-4 units
    co2_scale: int = 10_000    # grams -> 0.1 mg


@dataclass
class Params:
    fleet_types: dict = field(default_factory=lambda: dict(DEFAULT_FLEET_TYPES))
    ops: OpsParams = field(default_factory=OpsParams)
    risk: RiskParams = field(default_factory=RiskParams)
    zone_sensitivity: dict = field(default_factory=lambda: dict(DEFAULT_ZONE_SENSITIVITY))
    scale: ScaleParams = field(default_factory=ScaleParams)
