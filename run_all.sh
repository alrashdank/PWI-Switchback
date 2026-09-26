#!/usr/bin/env bash
# Regenerates every result file and the results summary from fixed seeds.
# Usage: bash run_all.sh   (or chmod +x run_all.sh && ./run_all.sh)
# Requires: pip install -r requirements.txt
# Approximate single-core runtime: about 70 minutes in total, most of it the Monte Carlo and the size checks.
set -e
mkdir -p results_v6
python3 pwi_study_v6.py --base                                     # base case, oracle convergence
python3 pwi_study_v6.py --all --chunk-size 40                      # Monte Carlo, 300 plants, in chunks
python3 - << 'PY'
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results_v6/mc_chunk*.csv"))]).sort_values("world")
df.to_csv("results_v6/mc_worlds.csv", index=False); print(len(df), "plants")
PY
python3 pwi_carryover_v6.py --reps 30 --policy-reps 30 --part both  # carry-over sweep and policy check
for c in 0.0 0.5 1.0; do python3 pwi_carryover_v6.py --part pilot --c $c --pilot-reps 500 --weeks 12; done
for c in 0.0 0.5; do python3 pwi_carryover_v6.py --part pilot --c $c --pilot-reps 300 --weeks 24; done
python3 size_check_v6.py four_day 0.0 120 250                     # size checks of the pilot test (S6)
python3 size_check_v6.py two_day 0.0 50 120
python3 size_check_v6.py four_day 0.5 120 250
python3 size_check_v6.py two_day 0.5 50 120
python3 size_variants.py two_day                                   # harder weak nulls (Table S6c)
python3 size_variants.py four_day
python3 size_variants.py daily                                     # one-day design, symmetric weak null
python3 h1_check.py                                                # H1 decision-rule operating characteristics
python3 h1_check_alpha01.py
python3 summarise_results.py > /dev/null                          # results_summary.json
echo "done"
