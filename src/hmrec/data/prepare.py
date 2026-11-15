"""Turn the raw H&M files into compact parquet + image thumbnails.

Input: ``paths.raw`` holding either the competition zip or the extracted CSVs (and ``images/``).
Outputs in ``paths.interim``:
  transactions.parquet   t_dat, week, customer (uint64), article_id (int32), price, channel
  customers.parquet      customer (uint64), customer_id (original hex string), age, status columns
  articles.parquet       article_id + names, descriptions and category codes
and 256-px JPEG thumbnails in ``paths.thumbs`` (one per article with an image).

The 64-character customer hash is reduced to the uint64 of its last 16 hex digits (collision-free on
this data; checked at run time), which cuts the transactions table from ~3.5 GB to ~0.5 GB in RAM.

    uv run python -m hmrec.data.prepare [--keep-zip]
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl

from hmrec.config import Config, load_config

CSVS = ["articles.csv", "customers.csv", "transactions_train.csv", "sample_submission.csv"]
THUMB = 256


def week_expr(test_week_start: date, test_week: int) -> pl.Expr:
    """Week number such that the hidden test week (starting ``test_week_start``) is ``test_week``."""
    days = (pl.col("t_dat") - pl.lit(test_week_start)).dt.total_days()
    return (days.floordiv(7) + test_week).cast(pl.Int16)


def customer_key(col: str = "customer_id") -> pl.Expr:
    return pl.col(col).str.slice(-16).str.to_integer(base=16, dtype=pl.UInt64)


def find_zip(raw: Path) -> Path | None:
    zips = sorted(raw.glob("*.zip"))
    return zips[0] if zips else None


def extract_csvs(raw: Path) -> None:
    z = find_zip(raw)
    if all((raw / c).exists() for c in CSVS) or z is None:
        return
    with zipfile.ZipFile(z) as zf:
        for c in CSVS:
            if not (raw / c).exists():
                print(f"  extracting {c}")
                zf.extract(c, raw)


def prepare_tables(cfg: Config) -> None:
    raw, out = cfg.paths.raw, cfg.paths.interim
    out.mkdir(parents=True, exist_ok=True)

    customers = pl.read_csv(raw / "customers.csv", infer_schema_length=10_000).select(
        customer_key().alias("customer"),
        "customer_id",
        pl.col("age").cast(pl.Int8),
        pl.col("FN").fill_null(0).cast(pl.Int8).alias("fn"),
        pl.col("Active").fill_null(0).cast(pl.Int8).alias("active"),
        pl.col("club_member_status").fill_null("NONE").cast(pl.Categorical),
        pl.col("fashion_news_frequency").fill_null("NONE").str.to_uppercase().cast(pl.Categorical),
    )
    if customers["customer"].n_unique() != customers.height:
        raise ValueError("customer key collision: the 16-hex-digit key is not unique on this data")
    customers.write_parquet(out / "customers.parquet")

    articles = pl.read_csv(
        raw / "articles.csv", schema_overrides={"article_id": pl.Int32}, infer_schema_length=0
    )
    int_cols = [c for c in articles.columns if c.endswith(("_no", "_code", "_id")) and c != "article_id"]
    articles = articles.with_columns(pl.col(int_cols).cast(pl.Int32, strict=False)).with_columns(
        pl.col("article_id").cast(pl.Int32)
    )
    articles.write_parquet(out / "articles.parquet")

    tx = (
        pl.scan_csv(raw / "transactions_train.csv", schema_overrides={"article_id": pl.Int32})
        .select(
            pl.col("t_dat").str.to_date(),
            customer_key().alias("customer"),
            pl.col("article_id").cast(pl.Int32),
            pl.col("price").cast(pl.Float32),
            pl.col("sales_channel_id").cast(pl.Int8).alias("channel"),
        )
        .with_columns(week_expr(cfg.test_week_start, cfg.test_week).alias("week"))
    )
    tx.sink_parquet(out / "transactions.parquet")
    n = pl.scan_parquet(out / "transactions.parquet").select(pl.len()).collect().item()
    print(f"tables: {customers.height} customers, {articles.height} articles, {n} transactions")


def _thumbs_worker(args: tuple[str, list[str], str]) -> int:
    from PIL import Image

    zip_path, names, out_dir = args
    done = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in names:
            dst = Path(out_dir) / Path(name).name
            if dst.exists():
                continue
            try:
                with Image.open(io.BytesIO(zf.read(name))) as src:
                    rgb = src.convert("RGB")
                    rgb.thumbnail((THUMB, THUMB))
                    rgb.save(dst, "JPEG", quality=85)
                done += 1
            except Exception as e:
                print(f"  skipped {name}: {e}")
    return done


def make_thumbnails(cfg: Config, workers: int = 8) -> None:
    z = find_zip(cfg.paths.raw)
    if z is None:
        print("no zip found; skipping thumbnails")
        return
    out = cfg.paths.thumbs
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.startswith("images/") and n.endswith(".jpg")]
    chunks = [names[i :: workers * 4] for i in range(workers * 4)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        made = sum(ex.map(_thumbs_worker, [(str(z), c, str(out)) for c in chunks]))
    print(f"thumbnails: {made} new, {len(names)} images in the archive, {time.perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--keep-zip", action="store_true", help="keep the 34 GB archive after extraction")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 4))
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.perf_counter()
    extract_csvs(cfg.paths.raw)
    prepare_tables(cfg)
    make_thumbnails(cfg, a.workers)
    z = find_zip(cfg.paths.raw)
    if z is not None and not a.keep_zip:
        z.unlink()
        print(f"deleted {z.name}")
    print(f"prepared in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
