"""Streamlit demo: a customer's recent purchases next to their 12 recommendations (with reasons).

uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SERVING = Path(os.environ.get("SERVING_DIR", ROOT / "reports" / "serving"))
THUMBS = Path(os.environ.get("THUMBS_DIR", ROOT / "data" / "thumbs"))
REASON = {
    "repurchase": ("Bought before", "#c99a3f"),
    "popular": ("Popular for your age", "#4f78e0"),
    "cooccur": ("Bought together", "#2fa37a"),
    "similar": ("Similar style", "#a07bd8"),
}

st.set_page_config(page_title="H&M recommender", layout="wide")


@st.cache_data
def load() -> tuple[pl.DataFrame, dict[int, dict], pl.DataFrame, pl.DataFrame | None, dict]:
    recs = pl.read_parquet(SERVING / "recommendations.parquet")
    arts = {r["article_id"]: r for r in pl.read_parquet(SERVING / "articles.parquet").iter_rows(named=True)}
    hist = pl.read_parquet(SERVING / "history.parquet")
    sim_path = SERVING / "similar.parquet"
    sim = pl.read_parquet(sim_path) if sim_path.exists() else None
    meta = json.loads((SERVING / "metrics.json").read_text(encoding="utf-8"))
    return recs, arts, hist, sim, meta


def card(col, a: int, arts: dict[int, dict], caption: str = "", chips: list[str] | None = None) -> None:  # type: ignore[no-untyped-def]
    p = THUMBS / f"{a:010d}.jpg"
    if p.exists():
        col.image(str(p), use_container_width=True)
    else:
        col.markdown(
            "<div style='aspect-ratio:3/4;background:#20232a;border-radius:8px;display:flex;"
            "align-items:center;"
            "justify-content:center;color:#8a8f99;font-size:12px'>no photo</div>",
            unsafe_allow_html=True,
        )
    r = arts.get(a, {})
    col.markdown(
        f"**{r.get('prod_name', a)}**  \n<span style='color:#8a8f99;font-size:12px'>"
        f"{r.get('product_type_name', '')} · {r.get('colour_group_name', '')}{caption}</span>",
        unsafe_allow_html=True,
    )
    if chips:
        col.markdown(
            " ".join(
                f"<span style='background:{REASON[c][1]}22;color:{REASON[c][1]};"
                f"border:1px solid {REASON[c][1]}66;"
                f"border-radius:999px;padding:1px 8px;font-size:11px'>{REASON[c][0]}</span>"
                for c in chips
                if c in REASON
            ),
            unsafe_allow_html=True,
        )


@st.cache_data
def photo_coverage(hist: pl.DataFrame, recs: pl.DataFrame) -> dict[str, float]:
    """Share of each demo customer's shown items (6 recent purchases + 12 recommendations) with a photo."""
    have = {int(p.stem) for p in THUMBS.glob("*.jpg")}
    out = {}
    for cid in hist["customer_id"].unique(maintain_order=True).to_list():
        past = hist.filter(pl.col("customer_id") == cid)["article_id"].unique(maintain_order=True).head(6)
        rec = recs.filter(pl.col("customer_id") == cid)["article_id"]
        items = past.to_list() + (rec[0].to_list() if len(rec) else [])
        out[cid] = sum(a in have for a in items) / max(len(items), 1)
    return out


if not (SERVING / "recommendations.parquet").exists():
    st.error("No serving bundle found. Run `python -m hmrec.train` first.")
    st.stop()

recs, arts, hist, sim, meta = load()
st.title("H&M personalised recommendations")
st.caption(
    f"Two-stage recommender (4 candidate sources + LightGBM LambdaRank) · trained {meta['trained_at']} · "
    f"recommending the week after the last known purchases"
)

m = meta["map12"]
c1, c2, c3 = st.columns(3)
best = max(m["baseline_popular"], m["baseline_repurchase_popular"])
c1.metric(
    "Ranker · MAP@12", f"{m['lgbm_ranker']:.4f}", f"{m['lgbm_ranker'] / best - 1:+.1%} vs best baseline"
)
c2.metric("Repurchase + popular · MAP@12", f"{m['baseline_repurchase_popular']:.4f}")
c3.metric("Global top-12 · MAP@12", f"{m['baseline_popular']:.4f}")

tab1, tab2 = st.tabs(["Customer view", "Similar items"])
with tab1:
    coverage = photo_coverage(hist, recs)
    customers = sorted(coverage, key=lambda c: -coverage[c])
    cid = st.selectbox(
        "Customer (demo customers, best photo coverage first)",
        customers,
        format_func=lambda c: f"{c[:12]}… · {coverage[c]:.0%} of items with photos",
    )
    st.subheader("Recent purchases")
    past = (
        hist.filter(pl.col("customer_id") == cid)
        .unique("article_id", keep="first", maintain_order=True)
        .head(6)
    )
    cols = st.columns(6)
    for col, row in zip(cols, past.iter_rows(named=True), strict=False):
        card(col, row["article_id"], arts, f" · {row['t_dat']}")

    st.subheader("Recommended for next week")
    row = recs.filter(pl.col("customer_id") == cid).row(0, named=True)
    for start in (0, 6):
        cols = st.columns(6)
        for col, a, r in zip(
            cols, row["article_id"][start : start + 6], row["reasons"][start : start + 6], strict=False
        ):
            card(col, a, arts, chips=[x for x in r.split(",") if x] or ["popular"])

with tab2:
    if sim is None:
        st.info("CLIP similarity was not computed for this bundle.")
    else:
        options = sim["article_id"].unique().sort().to_list()
        with_photo = [a for a in options if (THUMBS / f"{a:010d}.jpg").exists()] or options
        a = st.selectbox(
            "Article", with_photo, format_func=lambda x: f"{x} · {arts.get(x, {}).get('prod_name', '')}"
        )
        cols = st.columns(7)
        card(cols[0], a, arts, " · selected")
        nb = sim.filter(pl.col("article_id") == a).sort("sim_score", descending=True).head(6)
        for col, r in zip(cols[1:], nb.iter_rows(named=True), strict=False):
            card(col, r["neighbor"], arts, f" · sim {r['sim_score']:.2f}")
