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
    from swc.config import MODEL_VERSION
    reg = ROOT / "results" / f"regime_map_{MODEL_VERSION}_n150_r5_d20_it5000"
    d = pd.read_csv(reg / "summary_by_rep.csv")
    BINS = 1039                                   # bins in the n=150 synthetic instances (seed 100)
    AREA = (5.0 * (150 / 40) ** 0.5 / 1.6) ** 2   # km2 of the synthetic area
    daily_sensor = sensor_cost_per_day(BINS, AREA)
    print(f"sensor network cost: {daily_sensor:.1f} USD/day for {BINS} bins over {AREA:.1f} km2 "
          f"({daily_sensor / BINS:.3f} USD per bin-day)\n")

    rows = []
    for (mult, cv), g in d.groupby(["rate_mult", "cv"]):
        base = g[g.policy == "fixed_daily"].set_index("rep")
        for pol in ["threshold70", "risk_myopic"]:
            x = g[g.policy == pol].set_index("rep")
            saving = (base.cost_per_day - x.cost_per_day)
            net = saving - daily_sensor
            rows.append(dict(days_to_full=round(g.days_to_full.mean(), 1), cv=cv, policy=pol,
                             saving_per_day=round(saving.mean(), 1), net_per_day=round(net.mean(), 1),
                             pays_off="yes" if net.mean() > 0 else "no",
                             breakeven_usd_per_bin_day=round(saving.mean() / BINS, 3),
                             d_overflow_vs_fixed=round((x.overflow_bins_per_day - base.overflow_bins_per_day).mean(), 1)))
    t = pd.DataFrame(rows).sort_values(["cv", "days_to_full", "policy"])
    print(t.to_string(index=False))
    t.to_csv(reg / "break_even.csv", index=False)
    print("\nsaved", reg / "break_even.csv")


if __name__ == "__main__":
    main()
