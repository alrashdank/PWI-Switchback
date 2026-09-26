# PWI-Switchback: code and results

Simulation code and results for a study of cross-waste coupling in Lean 4.0 manufacturing cells, estimated with randomised switchback designs (K. Al-Rashdan, 2026).

Setup: `pip install -r requirements.txt` (Python 3.12). Then `bash run_all.sh` regenerates every result file and the results summary from fixed seeds (about 70 minutes on one core). The result files in `results_v6/` are plain CSV and JSON and can be plotted with any software.

| Step | Command | Output |
|---|---|---|
| Base case, oracle convergence | `python3 pwi_study_v6.py --base` | `results_v6/base_*`, `oracle_convergence.json` |
| Monte Carlo, 300 plants | `python3 pwi_study_v6.py --all --chunk-size 40` (or `--chunk k`) | `results_v6/mc_chunk*.csv` → `mc_worlds.csv` |
| Carry-over sweep and policy check | `python3 pwi_carryover_v6.py --reps 30 --policy-reps 30 --part both` | `results_v6/carryover_*.csv` |
| Pilot evaluation (size, power, bias) | `python3 pwi_carryover_v6.py --part pilot --c 0.5 --weeks 24 --pilot-reps 300` etc. | `results_v6/pilot_c*_w*.csv` |
| Pilot test size checks | `python3 size_check_v6.py four_day 0.0 120 250`, `two_day 0.0 50 120`, and the same at c = 0.5 | `results_v6/sizecheck_*.csv` |
| Harder weak nulls, one-day design | `python3 size_variants.py two_day`, `four_day`, `daily` | `results_v6/sizevariant_*.csv` |
| H1 decision rule | `python3 h1_check.py`, `python3 h1_check_alpha01.py` | `results_v6/h1_check*.csv` |
| Results summary | `python3 summarise_results.py` | `results_summary.json` |

Package versions are pinned in `requirements.txt`. Licence: MIT.
