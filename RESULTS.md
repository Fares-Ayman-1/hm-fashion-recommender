# Results

Generated from MLflow on 2026-11-15 18:34 UTC by `python -m hmrec.report`.
Metric: **MAP@12** (the competition metric; higher is better) on validation week 104
(2020-09-16 to 2020-09-22), over the 68,984 customers who bought something that week.
The ranker was trained on label weeks [101, 102, 103]; nothing from the validation week is used.

| Model | MAP@12 | MLflow run |
|---|---:|---|
| Two-stage: 4 candidate sources + LightGBM LambdaRank | **0.03278** | `9feec56d76c14f9f952f18caa1a15535` |
| Repurchase, then age-band popularity (heuristic) | 0.02600 | `0d2870e50d884fc099f74cc0d4178eef` |
| Global top-12 of last week | 0.00898 | `0d2870e50d884fc099f74cc0d4178eef` |

The ranker changes MAP@12 by **+26.1%** relative to the best baseline.

## Candidate recall (share of validation-week purchases that stage 1 surfaces)

| Source | Recall |
|---|---:|
| All sources combined | 0.1596 |
| Repurchase (own recent purchases) | 0.0277 |
| Age-band popularity | 0.0712 |
| Bought together (same week) | 0.0857 |
| CLIP similarity | 0.0180 |

Average candidates per customer: 181.9. Recall caps what the ranker can achieve: it can only reorder what stage 1 proposes.

## Training setup (from the run's parameters)

- 3,872,929 training rows (negatives sampled at 0.25), 42 features, 400 boosting rounds, learning rate 0.05, 63 leaves
- CLIP similarity source used: True
