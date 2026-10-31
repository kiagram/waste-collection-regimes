# When does sensor-based waste collection pay off?

Code, data and results for the article

> Sanaei Namaghi, K., Mirzapour Al-e-hashem, S.M.J. *When does sensor-based waste collection pay off?
> Risk-priced selective routing and a regime analysis for a megacity district.* Manuscript under review.

The software plans nightly municipal waste collection with fill-level information and evaluates
collection policies in a rolling-horizon simulation. The results in `results/` are version **v4** of the
analysis (`MODEL_VERSION` in `src/swc/config.py`).

- **Risk-priced selective routing** (`src/swc/model_cpsat.py`, `risk.py`): each night, every collection point is a
  candidate; skipping it costs the zone-weighted overflow risk that collecting it would avoid,
  $c^o\omega m (P_{skip} - P_{serve})$, with Gamma-distributed fill increments; points whose overflow probability
  exceeds a zone tolerance must be served (chance constraint, with unavoidable violations reported); the fleet is
  heterogeneous and multi-trip, and CO₂ depends on each truck's load. Exact CP-SAT (OR-Tools) formulation with a
  lexicographic ε-constraint method for cost–CO₂ fronts (`pareto.py`) and a carbon-priced objective.
- **District-scale engine** (`heuristic_pyvrp.py`): PyVRP with a greedy warm start that packs trips close to
  payload, exact truck-type swaps and best-of-two runs, all on the same (optionally carbon-priced) objective.
- **Policies and simulation** (`simulate.py`, `experiments.py`): policies without sensors (daily, every k days,
  elapsed-time rules, the risk model without readings) and with sensors (thresholds, myopic and look-ahead risk
  policies); spilled waste stays on the street until it is collected, with a cleanup cost and a daily mass
  balance; common random numbers; leave-one-replication-out tuning of every policy family; the value of sensing
  is measured against the best policy without sensors.
- **Case study** (`tehran.py`): Tehran District 6 built from OpenStreetMap, one modelled 1,100 L container per
  alley, with optional finer aggregation of collection points.
- **Forecasting** (`wyndham.py`, `forecast.py`): per-bin overflow probabilities on the open Wyndham City
  smart-bin data, the operational Gamma model, LightGBM and conformalised quantile regression, compared with
  panel-aware tests (Diebold–Mariano on date-averaged losses, moving-block bootstrap over dates).
- **Expert weights** (`bwm.py`): Bayesian best–worst method with input-based consistency checks.

## Installation

Python 3.13 is used for the published results.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -q tests
```

## Reproducing the article

`bash reproduce.sh [processes]` re-runs every experiment (two to three days on 16 cores). All stopping rules are
iteration-based and all random streams are seeded, so results do not depend on the machine or its load. Per-day
simulation outputs (`days_*.csv`) are included, so every summary, table and figure can be checked without
re-running the simulations: `21_tables.py` writes the article's tables and `15_figures.py` its figures from the
result files alone.

| Article | Script | Output in `results/` |
|---|---|---|
| Fig. 1 (District 6 instance) | `swc.tehran.build()` (needs internet), `15_figures.py --map` | `data/processed/tehran_d6.npz`, `figures/fig1_tehran_d6.png` |
| Section 6.1 (engine quality) | `04_validate_heuristic.py`, `17_heuristic_quality.py`, `20_long_search.py`, `18_engine_bias.py` | `validate_pyvrp_v4_n12.csv`, `heuristic_quality_v4/`, `long_search_v4_it20000x4/` |
| Table 3, Fig. 2 (value of sensing across regimes) | `06_regime_map.py` (3 replications, then 8 in the cells in `regime_map_v4_extra_cells.csv`), `08_break_even.py 8` | `regime_map_v4_n150_r8_it5000/`, `figures/fig2_value_of_sensing.png` |
| Fig. 3 (cost–overflow frontiers) | `07_overflow_price_sweep.py 3 5000` | `frontiers_v4_n150_cv1_r3_it5000/`, `figures/fig3_frontiers.png` |
| Table 4, Fig. 4 (District 6 options) | `09_tehran_sim.py 5 5000` | `tehran_d6_v4_r5_it5000/`, `figures/fig4_tehran_options.png` |
| Table 7 and Section 6.4 (robustness) | `12_tehran_sensitivity.py 3 5000 5`, `13_cost_robustness.py`, `18_engine_bias.py 8` | `tehran_sensitivity_v4_r3_it5000/`, `tehran_d6_v4_r5_it5000/cost_robustness.csv`, `heuristic_quality_v4/bias_*.csv` |
| Section 4.2 (warm-up check) | `22_warmup.py` | `warmup_mser5_v4*.csv` |
| Table 5, Fig. 5 (forecasting) | `05_forecast_benchmark.py <h> <threshold> [tolerance]` | `forecast_benchmark_v4_*/`, `figures/fig5_forecast_calib.png` |
| Fig. 6 (fleet) | `10_fleet_tradeoff.py 3 5000` | `fleet_tradeoff_tehran_v4/`, `figures/fig6_fleet_tradeoff.png` |
| Table 6, Fig. 7 (exact fronts) | `16_exact_fronts_tehran.py 6 12 6 120 8 full` (and `selective`) | `exact_fronts_tehran_v4_*/`, `figures/fig7_exact_fronts.png` |
| Section 6.6 (carbon proxy) | `19_carbon_proxy.py 6 12 120 8` | `carbon_proxy_v4_s6_n12_tl120/` |
| Tables 3–7 | `21_tables.py` | `tables_v4/` |
| Expert survey (Section 5.4) | `11_bwm_analysis.py` (reads `survey/responses.csv`) | `bwm_survey/` |

Other scripts: `02_pareto_demo.py` (cost–CO₂ front on a synthetic instance). `regime_map_v4_n150_r3_it5000/`
holds the summaries of the 3-replication stage, which the frontier script reads; its day files are the
replication 0–2 files of the 8-replication folder.

## Data and licences

- **Code:** MIT licence (`LICENSE`).
- **Wyndham City smart-bin data** (`data/raw/wyndham_*.json`): Wyndham City Council, via data.gov.au,
  Creative Commons Attribution 2.5 Australia. See `data/raw/SOURCE.md`.
- **Tehran District 6 instance** (`data/processed/tehran_d6*`): derived from OpenStreetMap,
  © OpenStreetMap contributors, Open Database Licence (ODbL) 1.0. The containers are modelled (one per alley),
  not a verified inventory.
- **Results and figures** (`results/`): CC BY 4.0.
- **Survey instrument** (`survey/`): Best–Worst questionnaire in English and Persian and an empty response
  template; no responses are included.

All cost and fleet parameters are low-cost global benchmarks and documented assumptions (`src/swc/config.py`,
`src/swc/tehran.py`), not municipal records. Zone sensitivities are illustrative values set by the authors;
the questionnaire and `11_bwm_analysis.py` let a municipality elicit its own.

## Versions

- **v4** (this version): penalty on the avoidable overflow risk with Gamma tails, spilled waste kept in the
  system, no-sensor baselines and leave-one-out tuning, panel-aware forecast tests, carbon-consistent seed
  selection, longer horizons and replications.
- **v3**: the results of the first submission, available in the git history (commit `c9876e4`).

## Citation

See `CITATION.cff`. Please cite the article once published.
