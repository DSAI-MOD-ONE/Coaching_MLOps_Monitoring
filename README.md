# Coaching Session — MLOps Monitoring for Production Models

**Module 3 · Machine Learning & GenAI · 3 hours (1 h concepts + 1 h 40 min hands-on + 20 min share)**

Your HDB price model works today. Will it still work next year? This session teaches you how to **watch a live model**, catch problems early, and fix them safely, using real HDB resale data from 2017 to 2026.

## The story
A model trained on 2017–2021 resale deals "goes live" in **January 2022**. We replay what really happened up to 2026:
the inputs look almost the same, yet the error climbs from **11.5% to 28.5%** and the model under-prices every flat.

## What's in this folder
```
slides/MLOps_Monitoring_slides.pptx      Concept slides (NTU template, speaker notes included)
notebooks/
  01_data_drift.ipynb                     PSI, KS & chi-square (and their big-data trap), new categories, unseen-town test
  02_concept_drift_and_performance.ipynb  Monthly scorecard, "same flat, higher price", segments, tree-model ceiling,
                                          the slow-clock sign check, monitoring a forecast
  03_retraining_and_rollback.ipynb        Time-based validation, retrain triggers, window backtest,
                                          champion vs challenger gate, rollback after a bad release
  04_optional_tools_mlflow_evidently.ipynb  (stretch) MLflow tracking + model registry, Evidently drift report
  mlops_utils.py                          Helper functions (data loading, model, mini model registry) — no need to edit
data/hdb_resale.csv                       data.gov.sg resale prices, Jan 2017 – Oct 2026 (offline copy)
requirements.txt
```

## Setup (5 minutes, before class)
```
pip install -r requirements.txt
```
Open the notebooks **from the `notebooks` folder** and run them in order (01 → 02 → 03). Cells marked **🔧 YOUR TURN** have a `____` blank to fill; hints are under **▸ Hint**.

- The data is included, so no internet is needed. To get the newest months, call `load_hdb(refresh=True)` once.
- Notebook 03 has one cell that trains about 25 models (1–2 minutes). That's normal.
- Notebook 04 needs `pip install "mlflow>=3" "evidently>=0.7"`. Older Evidently tutorials online use a different code style and will not work.

## Key ideas, in one line each
| Idea | Plain English | HDB example |
|---|---|---|
| Data drift | The inputs look different from training | Flats sold today have shorter remaining leases |
| PSI | One number: how much did this column move? (< 0.1 stable, > 0.25 big) | Lease PSI 0.17 → 0.60; floor area ~0.03 |
| New categories | Values the model never saw — they fail **silently** | `3Gen` flats (first resale June 2022); a future Tengah / Prime estate |
| Concept drift | Same inputs, different answer.change in target | Same 4-room Tampines flat: +38% since 2021 |
| Bias | Is the model always too high or too low? | −S$194k per flat by 2025 |
| Sign check | 10+ of the last 12 months wrong in the same direction = alarm | v1 hits the alarm in Aug 2022 |
| Time-based validation | Test on the newest months, not a random split | Random split promised 7%; reality was 12.5% |
| Sliding window | Retrain on recent data only | 7% error vs 23% for never retraining |
| Champion / challenger | A new model goes live only if it wins on recent unseen data | Gate: better overall, no flat type worse |
| Rollback | Keep every version + a "live" pointer | Bad v3 → `set_champion("v2")` |

## Bring to the share-out
1. One drift or performance finding on **your group's** model.
2. Your **retraining policy** table (notebook 03, Part F).
