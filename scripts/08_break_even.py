"""Sensor-cost break-even across regimes (uses the regime-map results).

Fixed-daily collection needs no sensors; threshold and risk-based policies do. A sensor-based
policy 'pays off' only if its daily saving vs fixed-daily exceeds the daily cost of the sensor
network. Costs are low-cost global benchmarks (ASSUMPTIONS, USD):
    sensor (ultrasonic, LoRaWAN) 60 + installation 15, life 5 y, battery/maintenance 5 /y
    gateway 600, life 5 y, backhaul 100 /y, one per GATEWAY_KM2 of served area
    network server / platform (open-source ChirpStack, hosting) 600 /y
Also reports the break-even sensor cost per bin-day (the maximum a municipality should pay).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

SENSOR, INSTALL, LIFE, MAINT = 60.0, 15.0, 5.0, 5.0
GATEWAY, BACKHAUL, GATEWAY_KM2, PLATFORM = 600.0, 100.0, 3.0, 600.0


def sensor_cost_per_day(bins: int, area_km2: float) -> float:
    per_bin_year = (SENSOR + INSTALL) / LIFE + MAINT
    gws = int(np.ceil(area_km2 / GATEWAY_KM2))
    network_year = gws * (GATEWAY / LIFE + BACKHAUL) + PLATFORM
    return (bins * per_bin_year + network_year) / 365.0


def main():
    """v4: the value of sensing is the paired saving in social cost (operating + cleanup + valued
    overflow) of the best sensor policy over the best policy without sensors, both tuned
    leave-one-replication-out (06_regime_map.py). Break-even = the most a municipality could pay per
    bin-day for sensing; it pays off where the saving exceeds the network cost above."""
    from swc.config import MODEL_VERSION
    from swc.experiments import paired_ci
    reps = sys.argv[1] if len(sys.argv) > 1 else "3"
    reg = ROOT / "results" / f"regime_map_{MODEL_VERSION}_n150_r{reps}_it5000"
    t = pd.read_csv(reg / "tuned_by_rep.csv")
    rows = []
    for (mult, cv, layout), g in t.groupby(["rate_mult", "cv", "layout"]):
        ns = g[g.tuned_family == "best_no_sensor"].set_index("rep")
        se = g[g.tuned_family == "best_sensor"].set_index("rep").loc[ns.index]
        fd = g[g.tuned_family == "fixed_daily"].set_index("rep").loc[ns.index]
        sav, hw = paired_ci(ns.social_cost - se.social_cost)
        sav_op, hw_op = paired_ci(ns.cost_per_day - se.cost_per_day)
        sav_fd, _ = paired_ci(fd.social_cost - se.social_cost)
        bins, net_cost = float(se.bins.mean()), float(se.sensor_cost_day.mean())
        rows.append(dict(days_to_full=round(g.days_to_full.mean(), 1), cv=cv, layout=layout,
                         best_no_sensor="/".join(sorted(set(ns.chosen))), best_sensor="/".join(sorted(set(se.chosen))),
                         saving_social=round(sav, 1), saving_social_hw=round(hw, 1),
                         saving_operating=round(sav_op, 1), saving_operating_hw=round(hw_op, 1),
                         saving_vs_fixed_daily=round(sav_fd, 1), sensor_cost_day=round(net_cost, 1),
                         net_per_day=round(sav - net_cost, 1),
                         pays_off="yes" if sav - hw > net_cost else ("no" if sav + hw < net_cost else "uncertain"),
                         breakeven_usd_per_bin_day=round(sav / bins, 3)))
    out = pd.DataFrame(rows).sort_values(["cv", "layout", "days_to_full"])
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    out.to_csv(reg / "break_even.csv", index=False)
    print()
    print("saved", reg / "break_even.csv")


if __name__ == "__main__":
    main()
