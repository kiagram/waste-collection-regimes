"""Analyse the zone-sensitivity BWM survey (article/survey/bwm_questionnaire_fa.docx).

Input: article/survey/responses.csv (see responses_template.csv), one row per expert:
    respondent_id, organisation, role, years_experience, best, worst,
    bo_<zone> ... (Best-to-Others, 1-9), ow_<zone> ... (Others-to-Worst, 1-9),
    alpha_<zone> ... (max acceptable overflow probability, %), days_to_full_<zone>, variability_<zone>
Output (article/results/bwm_survey/):
    validation.csv  (structural issues + input-based CR vs Liang et al. 2020 thresholds, per expert)
    weights.csv     (Bayesian group weights, 95% credible intervals, sensitivities scaled to [0, 1])
    credal.csv      (P(zone i more sensitive than zone j))
    alpha.csv       (median acceptable overflow probability per zone -> model tolerances)
usage:  python scripts/11_bwm_analysis.py            # survey responses
        python scripts/11_bwm_analysis.py --demo     # thesis 5 decision-makers (4 criteria)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swc.bwm import bayesian_bwm, input_based_cr, linear_bwm, validate

ZONES = ["residential", "commercial", "recreational", "governmental_diplomatic", "historical_cultural",
         "educational_medical", "industrial"]


def analyse(names, best, worst, BO, OW, ids, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows, keep = [], []
    for k in range(len(ids)):
        iss = validate(best[k], worst[k], BO[k], OW[k])
        cr, thr, ok = input_based_cr(best[k], worst[k], BO[k], OW[k])
        w, xi = linear_bwm(best[k], worst[k], BO[k], OW[k]) if not iss else (np.full(len(names), np.nan), np.nan)
        valid = not iss and ok is not False
        rows.append(dict(respondent=ids[k], best=names[best[k]], worst=names[worst[k]], issues="; ".join(iss),
                         CR_input=round(cr, 4), threshold=thr, consistent=ok, included=valid,
                         **{f"w_linear_{n}": round(v, 4) for n, v in zip(names, w)}))
        if valid:
            keep.append(k)
    v = pd.DataFrame(rows)
    v.to_csv(out / "validation.csv", index=False)
    print(v[["respondent", "best", "worst", "CR_input", "threshold", "consistent", "included", "issues"]].to_string(index=False))
    if len(keep) < 2:
        print("\nfewer than 2 valid responses: Bayesian aggregation skipped")
        return
    res = bayesian_bwm(BO[keep], OW[keep], draws=4000, burn=6000, seed=0)
    ci = res.interval()
    wt = pd.DataFrame({"zone": names, "weight_mean": res.mean.round(4), "ci_low": ci[:, 0].round(4),
                       "ci_high": ci[:, 1].round(4), "sensitivity_0_1": (res.mean / res.mean.max()).round(3)})
    wt.to_csv(out / "weights.csv", index=False)
    pd.DataFrame(res.credal().round(3), index=names, columns=names).to_csv(out / "credal.csv")
    print(f"\nBayesian BWM over {len(keep)} valid experts (acceptance {res.accept}):")
    print(wt.to_string(index=False))


def main():
    if "--demo" in sys.argv:
        names = ["cost", "social", "carbon_emission", "waste_emission"]
        dms = [(0, 3, [1, 4, 5, 6], [6, 2, 2, 1]), (0, 3, [1, 3, 3, 3], [4, 1, 1, 1]),
               (1, 2, [3, 2, 2, 4], [2, 2, 1, 1]), (1, 2, [2, 1, 2, 3], [2, 3, 1, 1]),
               (1, 0, [3, 1, 2, 3], [1, 3, 1, 1])]
        best, worst = [d[0] for d in dms], [d[1] for d in dms]
        BO, OW = np.array([d[2] for d in dms], float), np.array([d[3] for d in dms], float)
        analyse(names, best, worst, BO, OW, [f"DM{i + 1}" for i in range(5)], ROOT / "results" / "bwm_thesis_demo")
        return
    df = pd.read_csv(ROOT / "survey" / "responses.csv")
    idx = {z: i for i, z in enumerate(ZONES)}
    best = [idx[b] for b in df["best"]]
    worst = [idx[w] for w in df["worst"]]
    BO = df[[f"bo_{z}" for z in ZONES]].to_numpy(float)
    OW = df[[f"ow_{z}" for z in ZONES]].to_numpy(float)
    out = ROOT / "results" / "bwm_survey"
    analyse(ZONES, best, worst, BO, OW, df["respondent_id"].tolist(), out)
    alpha = df[[f"alpha_{z}" for z in ZONES]].median().rename(lambda c: c.replace("alpha_", ""))
    alpha.to_frame("median_acceptable_overflow_pct").to_csv(out / "alpha.csv")
    print("\nmedian acceptable overflow probability (%):", alpha.to_dict())


if __name__ == "__main__":
    main()
