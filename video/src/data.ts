import { staticFile } from "remotion";
import metricsJson from "../../reports/serving/metrics.json";
import importanceJson from "../../reports/feature_importance.json";
import kaggleJson from "../../reports/kaggle_score.json";
import monitoringJson from "../../reports/monitoring.json";
import demoJson from "../public/data/demo.json";

// Every number on screen comes from the project's own reports (MLflow-generated) or the derived demo.json
// (video/scripts/extract.py). README/RESULTS-only numbers are in README_NUM with their source.

export type Reason = "repurchase" | "popular" | "cooccur" | "similar";
export type Item = { id: string; name: string; type: string; colour: string; img: boolean };
export type Rec = Item & { rank: number; reasons: Reason[] };
type Demo = {
  customer: { short: string; band: string };
  past: (Item & { date: string })[];
  recs: Rec[];
  stats: { customers: number; transactions: number; articles: number; last_week: number; scored_customers: number };
  images: number;
};
export const DEMO = demoJson as unknown as Demo;
export const img = (it: Item) => (it.img ? staticFile(`data/img/${it.id}.jpg`) : null);

const m = metricsJson as {
  k: number;
  valid_week: number;
  test_week: number;
  map12: { lgbm_ranker: number; baseline_popular: number; baseline_repurchase_popular: number };
  candidate_recall: Record<"all" | Reason, number>;
  mlflow_run_ids: { baselines: string; lgbm: string };
};
const k = kaggleJson as { private: string; public: string; date: string };
const mon = monitoringJson as {
  week: number;
  values: { new_article_share: number; top100_overlap: number; price_psi: number };
  alerts: Record<string, boolean>;
  retrain_recommended: boolean;
};

export const NUM = {
  k: m.k,
  validWeek: m.valid_week,
  testWeek: m.test_week,
  ranker: m.map12.lgbm_ranker,
  repPop: m.map12.baseline_repurchase_popular,
  popular: m.map12.baseline_popular,
  lift: m.map12.lgbm_ranker / Math.max(m.map12.baseline_popular, m.map12.baseline_repurchase_popular) - 1,
  recall: m.candidate_recall,
  runId: m.mlflow_run_ids.lgbm,
  kagglePrivate: Number(k.private),
  kagglePublic: Number(k.public),
  mon: mon.values,
  monAlerts: Object.values(mon.alerts).filter(Boolean).length,
  retrain: mon.retrain_recommended,
  customers: DEMO.stats.customers,
  transactions: DEMO.stats.transactions,
  articles: DEMO.stats.articles,
  scored: DEMO.stats.scored_customers,
};

/** Numbers that only exist in README.md / RESULTS.md / MODEL_CARD.md (cited in the number table). */
export const README_NUM = {
  candidatesPerCustomer: 181.9, // RESULTS.md "Average candidates per customer: 181.9"
  features: 42, // RESULTS.md "42 features"
  validBuyers: 68984, // README/RESULTS "68,984 customers who bought something that week"
  validDates: "2020-09-16 to 2020-09-22", // README/RESULTS validation week
  recallNoClip: 0.155, // README "raises combined candidate recall only from 15.5% to 16.0%"
  mapNoClip: 0.03267, // README "the same pipeline without the CLIP source scored 0.03267"
  noise: 0.0002, // README "LightGBM's run-to-run variation (about ±0.0002)"
  // monitoring thresholds, MODEL_CARD.md
  churnLimit: 0.3,
  overlapLimit: 0.4,
  psiLimit: 0.2,
};

// Feature importance (LightGBM gain), top 8 with readable labels; raw names stay small in mono.
const LABELS: Record<string, string> = {
  rep_weeks_ago: "Weeks since they last bought it",
  department_no: "Article department",
  i_cnt_1w: "Sales last week",
  rep_rank: "Rank among their own purchases",
  co_score_max: "Strongest bought-together link",
  i_trend: "Sales trend (last week vs month)",
  colour_group_code: "Colour",
  u_weeks_since_last: "Weeks since their last visit",
};
const fi = Object.entries(importanceJson as Record<string, number>).sort((a, b) => b[1] - a[1]);
const fiTotal = fi.reduce((a, [, v]) => a + v, 0);
export const IMPORTANCE = fi.slice(0, 8).map(([name, gain]) => ({ name, label: LABELS[name] ?? name, share: gain / fiTotal }));
export const FEATURES_IN_FILE = fi.length;

export const REASON: Record<Reason, { label: string; text: string; color: string }> = {
  // labels and text from app/dashboard.py and src/hmrec/serve/api.py (REASON_TEXT)
  repurchase: { label: "Bought before", text: "You bought this recently", color: "#0C1733" },
  popular: { label: "Popular for your age", text: "Popular last week with customers your age", color: "#D3AE6C" },
  cooccur: { label: "Bought together", text: "Often bought together with your recent purchases", color: "#7EA4FF" },
  similar: { label: "Similar style", text: "Similar to items you bought", color: "#62D6A2" },
};

// formatting helpers
export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const compact = (n: number, d = 2) => (n >= 1e6 ? `${(n / 1e6).toFixed(d)}M` : `${Math.round(n / 1e3)}k`);
