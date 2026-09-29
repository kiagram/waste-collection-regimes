#!/bin/bash
# Re-runs every experiment reported in the article (results tagged v3). About 5 hours on 16 cores.
# Usage: bash reproduce.sh            (set PY to your interpreter if it is not .venv)
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=.venv/bin/python
L=results/logs; mkdir -p $L
echo "start $(date)"

# forecasting on the Wyndham data
for h in 1 2 3; do $PY scripts/05_forecast_benchmark.py $h > $L/forecast_h$h.log 2>&1; done

# phase A: simulations, sharded across processes
pids=()
for i in 0 1 2 3 4 5 6; do SHARD=$i/7 $PY scripts/09_tehran_sim.py 5 20 5000 > $L/sim_shard$i.log 2>&1 & pids+=($!); done
for i in 0 1 2 3 4; do SHARD=$i/5 $PY scripts/12_tehran_sensitivity.py 3 20 5000 > $L/sens_shard$i.log 2>&1 & pids+=($!); done
for i in 0 1 2 3; do SHARD=$i/4 $PY scripts/06_regime_map.py 150 5 20 5000 > $L/regime_shard$i.log 2>&1 & pids+=($!); done
wait "${pids[@]}"
$PY scripts/06_regime_map.py 150 5 20 5000 > $L/regime.log 2>&1
$PY scripts/09_tehran_sim.py 5 20 5000 > $L/tehran_sim.log 2>&1
$PY scripts/12_tehran_sensitivity.py 3 20 5000 > $L/tehran_sens.log 2>&1

# phase B: sweeps, fleet, exact fronts, engine quality
pids=()
$PY scripts/07_overflow_price_sweep.py 5 20 5000 > $L/sweep.log 2>&1 & pids+=($!)
$PY scripts/03_policy_sim.py 150 5 20 10 60 pyvrp 1.0 5000 > $L/policy_sim.log 2>&1 & pids+=($!)
$PY scripts/10_fleet_tradeoff.py 3 5000 > $L/fleet.log 2>&1 & pids+=($!)
$PY scripts/04_validate_heuristic.py 12 8 120 3 > $L/validate.log 2>&1
$PY scripts/16_exact_fronts_tehran.py 6 12 6 120 12 full > $L/fronts_full.log 2>&1
$PY scripts/16_exact_fronts_tehran.py 6 12 6 120 12 selective > $L/fronts_selective.log 2>&1
wait "${pids[@]}"
$PY scripts/17_heuristic_quality.py 14 > $L/heuristic_quality.log 2>&1

# summaries and figures
$PY scripts/08_break_even.py > $L/break_even.log 2>&1
$PY scripts/13_cost_robustness.py > $L/cost_robustness.log 2>&1
$PY scripts/14_robustness_summary.py > $L/robustness_summary.log 2>&1
$PY scripts/18_engine_bias.py > $L/engine_bias.log 2>&1
$PY scripts/15_figures.py > $L/figures.log 2>&1
echo "ALL DONE $(date)"
