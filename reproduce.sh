#!/bin/bash
# Re-runs every experiment reported in the article (results tagged v4). Two to three days on 16 cores.
# Every simulation script is resumable and can be split across processes with SHARD=i/N; an unsharded run
# afterwards reuses the saved day files and writes the summaries.
# Usage: bash reproduce.sh [processes]      (set PY to your interpreter if it is not .venv)
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=.venv/bin/python
N=${1:-8}
L=results/logs; mkdir -p $L
echo "start $(date)"
sharded() {  # sharded <name> <script> <args...>: N shards, then the summary run
  local name=$1; shift; local pids=()
  for i in $(seq 0 $((N - 1))); do SHARD=$i/$N $PY "$@" > $L/${name}_shard$i.log 2>&1 & pids+=($!); done
  wait "${pids[@]}"
  $PY "$@" > $L/$name.log 2>&1
}

# forecasting on the Wyndham data (Table 5, Fig. 5)
for h in 1 2 3; do
  $PY scripts/05_forecast_benchmark.py $h 0.8 > $L/forecast_h${h}_thr0.8.log 2>&1
  $PY scripts/05_forecast_benchmark.py $h 1.0 > $L/forecast_h${h}_thr1.log 2>&1
  $PY scripts/05_forecast_benchmark.py $h 0.8 0.2 > $L/forecast_h${h}_tol0.2.log 2>&1
done

# District 6 (Table 4, Fig. 4) and its sensitivity scenarios (Table 7)
sharded tehran scripts/09_tehran_sim.py 5 5000
sharded tehran_sensitivity scripts/12_tehran_sensitivity.py 3 5000 5

# regime map (Table 3, Fig. 2): 3 replications everywhere, then 8 in the cells left uncertain
sharded regime3 scripts/06_regime_map.py 150 3 5000
mkdir -p results/regime_map_v4_n150_r8_it5000
cp results/regime_map_v4_n150_r3_it5000/days_*.csv results/regime_map_v4_n150_r8_it5000/
sharded regime8 scripts/06_regime_map.py 150 8 5000 results/regime_map_v4_extra_cells.csv
$PY scripts/08_break_even.py 8 > $L/break_even.log 2>&1

# cost-overflow frontiers (Fig. 3)
sharded frontiers scripts/07_overflow_price_sweep.py 3 5000

# fleet, exact fronts, carbon proxy (Section 6.6, Table 6, Figs. 6-7)
$PY scripts/10_fleet_tradeoff.py 3 5000 > $L/fleet.log 2>&1
$PY scripts/16_exact_fronts_tehran.py 6 12 6 120 8 full > $L/fronts_full.log 2>&1
$PY scripts/16_exact_fronts_tehran.py 6 12 6 120 8 selective > $L/fronts_selective.log 2>&1
$PY scripts/19_carbon_proxy.py 6 12 120 8 > $L/carbon_proxy.log 2>&1

# engine quality (Section 6.1)
$PY scripts/04_validate_heuristic.py 12 8 120 3 > $L/validate.log 2>&1
$PY scripts/17_heuristic_quality.py $N > $L/heuristic_quality.log 2>&1
sharded long_search scripts/20_long_search.py 20000 4

# robustness, warm-up, tables and figures
$PY scripts/13_cost_robustness.py > $L/cost_robustness.log 2>&1
$PY scripts/18_engine_bias.py 8 > $L/engine_bias.log 2>&1
$PY scripts/22_warmup.py > $L/warmup.log 2>&1
$PY scripts/21_tables.py > $L/tables.log 2>&1
$PY scripts/15_figures.py > $L/figures.log 2>&1
echo "ALL DONE $(date)"
