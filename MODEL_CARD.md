# Model card: H&M two-stage recommender

## What it does
For every H&M customer, it recommends the 12 articles they are most likely to buy next week. It works
in two stages:
1. **Candidates.** Four sources propose articles:
   - the customer's own recent purchases;
   - last week's best sellers in the customer's age band;
   - articles bought in the same week as the customer's recent purchases;
   - nearest neighbours in CLIP text-embedding space.
2. **Ranking.** A LightGBM LambdaRank model orders the candidates, using user, item and user-item
   features.

Each recommendation records which sources suggested it, so it can be explained.

## Intended use
- A portfolio demonstration of a production-style recommender.
- **Not** for commercial use: the data licence is non-commercial (see Training data).

## Training data
H&M Personalized Fashion Recommendations (Kaggle):
- 31.8M purchases by 1.37M customers of 105k articles, 2018-09-20 to 2020-09-22.
- Customer ages and membership status; article names, categories and descriptions.

The data is used under the competition's non-commercial and academic terms, and it is not redistributed.
Customer ids are hashed in the source data; the model never sees names or contact details.

## Evaluation
- **Metric:** MAP@12, the competition metric.
- **Validation week:** 2020-09-16 to 2020-09-22, scored on every customer who bought that week.
- **Training data for the ranker:** the three weeks before the validation week. Candidates and
  features only use purchases from before the week being scored (tested).
- **Comparison:** global top-12 and a repurchase-then-popular heuristic.

Results are in [RESULTS.md](RESULTS.md), generated from MLflow.

## Known limitations
- **Popularity bias.** Most customers have little history. Their lists lean on age-band popularity,
  so niche tastes are under-served.
- **Cold-start articles.** Only articles sold in the last 4 weeks are recommended, so new launches
  enter only after their first sales.
- **Text-only similarity.** CLIP similarity uses product text, not photos. Downloading the 28.7 GB
  photo archive was impractical here; image embeddings are supported with `--images`.
- **Age as a proxy.** Age bands are a coarse stand-in for taste. Customers with no recorded age fall
  back to global popularity.
- **One-week horizon.** Recommendations are refreshed weekly; nothing reacts within the week.

## Monitoring
`python -m hmrec.monitoring` compares the latest week with the week before:
- **Catalogue churn:** an alert when articles first sold in the last 2 weeks make up more than 30%
  of purchases.
- **Popularity shift:** an alert when the top-100 articles of the two weeks overlap by less than 0.40
  (Jaccard).
- **Price drift:** an alert when the PSI of transaction prices exceeds 0.2.

Any alert recommends retraining before the weekly schedule.
