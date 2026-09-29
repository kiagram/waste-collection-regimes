"""Overflow-price sweep: how the risk-based policy trades cost against overflows.

Run in the regime where the decision matters (≈6 days to full, CV = 1.0; see R7), on the
same instances and fill realisations as the regime map, so fixed-daily / threshold baselines
from the regime map are directly comparable (common random numbers).
usage: python scripts/07_overflow_price_sweep.py [reps] [days] [pyvrp_iters]
"""
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from swc.config import MODEL_VERSION, Params
from swc.instance import generate_synthetic
from swc.simulate import DayRecord, RiskMyopic, SimConfig, draw_increments, simulate, summarise

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
days = int(sys.argv[2]) if len(sys.argv) > 2 else 20
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
n, MULT, CV = 150, 0.5, 1.0
PRICES = [15.0, 30.0, 60.0, 120.0, 240.0]
cfg = SimConfig(days=days, engine="pyvrp", pyvrp_s=120, pyvrp_iters=iters)
out = ROOT / "results" / f"overflow_sweep_{MODEL_VERSION}_n{n}_m{MULT}_cv{CV}_r{reps}_d{days}"
out.mkdir(parents=True, exist_ok=True)

rows = []
for price in PRICES:
    for rep in range(reps):
        P = Params()
        P.risk = replace(P.risk, overflow_cost=price, fill_rate_cv=CV)
        fleet = {"new_compactor": max(1, n // 60), "old_compactor": max(2, n // 30), "small_loader": max(1, n // 60)}
        inst = generate_synthetic(n, seed=100 + rep, params=P, bins_per_point=(4, 10), fleet_counts=fleet,
                                  area_m=5000.0 * (n / 40) ** 0.5 / 1.6)
        inst.rate_mean_per_h = inst.rate_mean_per_h * MULT
        inst.rate_sd_per_h = inst.rate_sd_per_h * MULT
        inc = draw_increments(inst, days, seed=1000 + rep)   # identical to the regime map
        f = out / f"days_price{price:g}_rep{rep}.csv"
        t0 = time.perf_counter()
        if f.exists():
            d = pd.read_csv(f)
            recs = [DayRecord(**{k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in d.to_dict("records")]
        else:
            recs = simulate(inst, RiskMyopic(), inc, cfg)
            pd.DataFrame([asdict(r) for r in recs]).to_csv(f, index=False)
        s = summarise(recs, burn_in=3)
        s.update(price=price, rep=rep, sim_s=time.perf_counter() - t0)
        rows.append(s)
        print(f"price={price:<6g} rep={rep} cost={s['cost_per_day']:7.1f} overflow={s['overflow_bins_per_day']:5.1f} "
              f"weighted={s['overflow_weighted_per_day']:5.1f} served={s['points_served_per_day']:5.1f} "
              f"({s['sim_s']:.0f}s)", flush=True)
sw = pd.DataFrame(rows)
sw.to_csv(out / "summary_by_rep.csv", index=False)
# sweep table, with fixed-daily and threshold baselines from the regime map (same instances and increments)
reg = pd.read_csv(ROOT / "results" / f"regime_map_{MODEL_VERSION}_n{n}_r5_d{days}_it{iters}" / "summary_by_rep.csv")
reg = reg[(reg.rate_mult == MULT) & (reg.cv == CV) & (reg.rep < reps)]
base = reg[reg.policy == "fixed_daily"].set_index("rep").cost_per_day
tab = []
for name, g in [(p, reg[reg.policy == p]) for p in ("fixed_daily", "threshold70")] + \
        [(f"risk, price={pr:g}", sw[sw.price == pr]) for pr in PRICES]:
    g = g.set_index("rep")
    tab.append(dict(setting=name, cost=round(g.cost_per_day.mean()),
                    dcost=round(((g.cost_per_day / base - 1) * 100).mean(), 1),
                    overflow=round(g.overflow_bins_per_day.mean(), 1),
                    weighted=round(g.overflow_weighted_per_day.mean(), 1), served=round(g.points_served_per_day.mean())))
pd.DataFrame(tab).to_csv(out / "sweep_table.csv", index=False)
print(pd.DataFrame(tab).to_string(index=False))
print("saved", out)
