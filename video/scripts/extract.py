"""Extract the video's demo data from the serving bundle (run from the project root).

    .venv/Scripts/python video/scripts/extract.py

Writes video/public/data/demo.json and a handful of thumbnails to video/public/data/img/ (both git-ignored:
they are derived from the competition data). The demo customer is the one the Streamlit demo opens first
(best photo coverage), so the hook, the "explainable" scene and the screen recording show the same person.
Only a short prefix of the customer hash is written, plus the age band the popularity source uses.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SERVING = ROOT / "reports" / "serving"
INTERIM = ROOT / "data" / "interim"
THUMBS = ROOT / "data" / "thumbs"
OUT = ROOT / "video" / "public" / "data"
MAX_IMAGES = 16  # DATA_RULES: a few product images, no bulk
N_PAST = 4
AGE_BANDS = [(20, "<20"), (25, "20-24"), (30, "25-29"), (40, "30-39"), (50, "40-49"), (60, "50-59")]


def band(age: int | None) -> str:
    if age is None:
        return "unknown"
    return next((label for limit, label in AGE_BANDS if age < limit), "60+")


def thumb(a: int) -> Path:
    return THUMBS / f"{a:010d}.jpg"


def main() -> None:
    have = {int(p.stem) for p in THUMBS.glob("*.jpg")}
    hist = pl.read_parquet(SERVING / "history.parquet")
    cids = hist["customer_id"].unique(maintain_order=True).to_list()
    recs = (
        pl.scan_parquet(SERVING / "recommendations.parquet")
        .filter(pl.col("customer_id").is_in(cids))
        .collect()
    )
    rec_map = {r["customer_id"]: (r["article_id"], r["reasons"]) for r in recs.iter_rows(named=True)}

    # Same ordering as app/dashboard.py: 6 recent purchases + 12 recommendations, best photo coverage first.
    coverage: dict[str, float] = {}
    for cid in cids:
        past = hist.filter(pl.col("customer_id") == cid)["article_id"].unique(maintain_order=True).head(6)
        items = past.to_list() + list(rec_map[cid][0])
        coverage[cid] = sum(a in have for a in items) / max(len(items), 1)
    cid = sorted(coverage, key=lambda c: -coverage[c])[0]

    arts = pl.read_parquet(SERVING / "articles.parquet")
    info = {r["article_id"]: r for r in arts.iter_rows(named=True)}
    ages = pl.read_parquet(INTERIM / "customers.parquet", columns=["customer_id", "age"])
    age = ages.filter(pl.col("customer_id") == cid)["age"].to_list()

    def article(a: int) -> dict:
        r = info[a]
        return {
            "id": f"{a:010d}",
            "name": r["prod_name"],
            "type": r["product_type_name"],
            "colour": r["colour_group_name"],
            "img": a in have,
        }

    past_rows = (
        hist.filter(pl.col("customer_id") == cid)
        .unique("article_id", keep="first", maintain_order=True)
        .head(6)
        .filter(pl.col("article_id").is_in(list(have)))
        .head(N_PAST)
    )
    past = [{**article(r["article_id"]), "date": str(r["t_dat"])} for r in past_rows.iter_rows(named=True)]
    ids, reasons = rec_map[cid]
    rec = [
        {**article(a), "rank": i, "reasons": [x for x in r.split(",") if x] or ["popular"]}
        for i, (a, r) in enumerate(zip(ids, reasons, strict=True), start=1)
    ]

    # thumbnails: only the ones this customer's cards use, capped
    OUT.mkdir(parents=True, exist_ok=True)
    img_dir = OUT / "img"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir()
    used = [x["id"] for x in past + rec if x["img"]]
    assert len(set(used)) <= MAX_IMAGES, f"{len(set(used))} product images > {MAX_IMAGES}"
    for a in set(used):
        shutil.copy(thumb(int(a)), img_dir / f"{a}.jpg")

    # dataset scale, straight from the prepared parquet files (matches the README's 1.37M / 31.8M / 105k)
    def rows(name: str) -> int:
        return pq.ParquetFile(INTERIM / f"{name}.parquet").metadata.num_rows

    weeks = pl.scan_parquet(INTERIM / "transactions.parquet").select(pl.col("week").max()).collect().item()
    out = {
        "customer": {"short": cid[:8], "band": band(age[0] if age else None)},
        "coverage": coverage[cid],
        "past": past,
        "recs": rec,
        "stats": {
            "customers": rows("customers"),
            "transactions": rows("transactions"),
            "articles": rows("articles"),
            "last_week": int(weeks),
            "scored_customers": pl.scan_parquet(SERVING / "recommendations.parquet")
            .select(pl.len())
            .collect()
            .item(),
        },
        "images": len(set(used)),
    }
    (OUT / "demo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"customer {cid[:8]}… · {len(past)} past · {len(rec)} recs · {len(set(used))} images")
    print("stats:", out["stats"])


if __name__ == "__main__":
    main()
