# When does sensor-based waste collection pay off?

Code, data and results for the article

> Sanaei Namaghi, K., Mirzapour Al-e-hashem, S.M.J. *When does sensor-based waste collection pay off?
> Risk-priced selective routing and a regime analysis for a megacity district.* Manuscript under review.

The software plans nightly municipal waste collection with fill-level information and evaluates
collection policies in a rolling-horizon simulation:

- **Risk-priced selective routing** (`src/swc/model_cpsat.py`): each night, every collection point is a
  candidate; skipping it costs its zone-weighted expected overflow, points whose overflow probability exceeds a
  zone tolerance must be served (chance constraint), the fleet is heterogeneous and multi-trip, and CO₂ depends
  on each truck's load. Exact CP-SAT (OR-Tools) formulation with a lexicographic ε-constraint method for
  cost–CO₂ fronts (`pareto.py`).
- **District-scale engine** (`heuristic_pyvrp.py`): PyVRP with a greedy warm start that packs trips close to
  payload, exact truck-type swaps and best-of-two runs; an optional fleet-descent loop.
- **Policies and simulation** (`simulate.py`): fixed daily collection, a 70% threshold rule and risk-based
  policies (myopic and look-ahead), with common random numbers.
- **Case study** (`tehran.py`): Tehran District 6 built from OpenStreetMap, one 1,100 L container per alley.
- **Forecasting** (`wyndham.py`, `forecast.py`): calibrated per-bin overflow probabilities on the open
  Wyndham City smart-bin data, compared with LightGBM and conformalised quantile regression.
- **Expert weights** (`bwm.py`): Bayesian best–worst method with input-based consistency checks.

## Installation

Python 3.13 is used for the published results.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -q tests
```

## Reproducing the article

`bash reproduce.sh` re-runs every experiment (about 5 hours on 16 cores; the simulation scripts can be split
across processes with `SHARD=i/N`). The table below maps each part of the article to its script and output.
Per-day simulation outputs (`days_*.csv`) are included, so the summaries and figures can be checked without re-running the simulations.

| Article | Script | Output in `results/` |
|---|---|---|
| Fig. 1 (District 6 instance) | `swc.tehran.build()` (needs internet), `15_figures.py` | `data/processed/tehran_d6.npz`, `figures/fig3_tehran_d6.png` |
| Section 6.1 (engine quality) | `04_validate_heuristic.py`, `17_heuristic_quality.py`, `18_engine_bias.py` | `validate_pyvrp_v3_n12.csv`, `heuristic_quality_v3/` |
| Table 3, Fig. 2 (regime map) | `06_regime_map.py 150 5 20 5000` | `regime_map_v3_*/`, `figures/fig1_regime_map.png` |
| Fig. 3 (break-even) | `08_break_even.py`, `15_figures.py` | `regime_map_v3_*/break_even.csv`, `figures/fig2_break_even.png` |
| Table 4, Fig. 4 (District 6 options) | `09_tehran_sim.py 5 20 5000`, `14_robustness_summary.py` | `tehran_d6_v3_*/`, `figures/fig4_tehran_options.png` |
| Table 5, Fig. 5 (forecasting) | `05_forecast_benchmark.py 1` (and `2`, `3`) | `forecast_benchmark*/`, `figures/fig6_forecast_calib.png` |
| Fig. 6 (fleet) | `10_fleet_tradeoff.py 3 5000` | `fleet_tradeoff_tehran_v3/`, `figures/fig5_fleet_tradeoff.png` |
| Table 6, Fig. 7 (exact fronts) | `16_exact_fronts_tehran.py 6 12 6 120 12 full` (and `selective`) | `exact_fronts_tehran_v3_*/`, `figures/fig7_exact_fronts.png` |
| Table 7 (robustness) | `12_tehran_sensitivity.py 3 20 5000`, `13_cost_robustness.py`, `14_robustness_summary.py`, `07_overflow_price_sweep.py 5 20 5000`, `03_policy_sim.py 150 5 20 10 60 pyvrp 1.0 5000` | `tehran_sensitivity_v3_*/`, `tehran_d6_v3_*/cost_robustness.csv`, `overflow_sweep_v3_*/`, `policy_sim_v3_*/` |
| Expert survey (Section 5.4) | `11_bwm_analysis.py` (reads `survey/responses.csv`) | `bwm_survey/` |

Other scripts: `02_pareto_demo.py` (cost–CO₂ front on a synthetic instance).

## Data and licences

- **Code:** MIT licence (`LICENSE`).
- **Wyndham City smart-bin data** (`data/raw/wyndham_*.json`): Wyndham City Council, via data.gov.au,
  Creative Commons Attribution 2.5 Australia. See `data/raw/SOURCE.md`.
- **Tehran District 6 instance** (`data/processed/tehran_d6*`): derived from OpenStreetMap,
  © OpenStreetMap contributors, Open Database Licence (ODbL) 1.0.
- **Results and figures** (`results/`): CC BY 4.0.
- **Survey instrument** (`survey/`): Best–Worst questionnaire in English and Persian and an empty response
  template; no responses are included.

All cost and fleet parameters are low-cost global benchmarks and documented assumptions (`src/swc/config.py`,
`src/swc/tehran.py`), not municipal records. Zone sensitivities are placeholders until the expert survey is
complete.

## Citation

See `CITATION.cff`. Please cite the article once published.
