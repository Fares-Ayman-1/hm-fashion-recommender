"""Download what the project needs from the H&M competition with the Kaggle API.

The full competition archive is 28.7 GB, almost all of it product photos. Training only needs the
three CSVs (~3.7 GB), so they are fetched one by one. Photos are fetched only for the articles
shown in the demo (``--images``), one request each, throttled to stay under the API rate limit.

    uv run python scripts/download_data.py                 # the three CSVs + SHA-256 manifest
    uv run python scripts/download_data.py --images 800    # thumbnails for the demo articles

Requires KAGGLE_API_TOKEN (or ~/.kaggle/kaggle.json) and the competition rules accepted on kaggle.com.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmrec.config import Config, load_config

CSVS = ["articles.csv", "customers.csv", "transactions_train.csv"]


def api():  # type: ignore[no-untyped-def]
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    from kaggle.api.kaggle_api_extended import KaggleApi

    k = KaggleApi()
    k.authenticate()
    return k


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(k, cfg: Config, name: str, dest: Path) -> None:  # type: ignore[no-untyped-def]
    dest.mkdir(parents=True, exist_ok=True)
    k.competition_download_file(cfg.competition, name, path=str(dest), quiet=True)
    z = dest / (Path(name).name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()


def download_csvs(cfg: Config) -> None:
    k = api()
    raw = cfg.paths.raw
    for name in CSVS:
        if (raw / name).exists():
            continue
        t0 = time.perf_counter()
        fetch(k, cfg, name, raw)
        size = (raw / name).stat().st_size
        print(f"{name}: {size / 1e6:.0f} MB in {time.perf_counter() - t0:.0f}s", flush=True)
    manifest = {
        "competition": cfg.competition,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {n: {"bytes": (raw / n).stat().st_size, "sha256": sha256(raw / n)} for n in CSVS},
    }
    (raw.parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_images(cfg: Config, article_ids: list[int], pause: float = 0.6) -> None:
    """Fetch photos for the given articles and store 256-px thumbnails (originals are not kept)."""
    from PIL import Image

    k = api()
    out, tmp = cfg.paths.thumbs, cfg.paths.raw / "_img"
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for a in article_ids:
        aid = f"{int(a):010d}"
        dst = out / f"{aid}.jpg"
        if dst.exists():
            continue
        try:
            found = next(tmp.rglob(f"{aid}.jpg"), None)  # Kaggle keeps the images/<prefix>/ folder path
            if found is None:
                fetch(k, cfg, f"images/{aid[:3]}/{aid}.jpg", tmp)
                found = next(tmp.rglob(f"{aid}.jpg"), None)
            if found is None:
                raise FileNotFoundError(aid)
            with Image.open(found) as src:
                rgb = src.convert("RGB")
                rgb.thumbnail((256, 256))
                rgb.save(dst, "JPEG", quality=85)
            found.unlink()
            done += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  {aid}: {type(e).__name__}: {str(e)[:120]}")
            if "429" in str(e):
                time.sleep(30)
        time.sleep(pause)
    print(f"images: {done} downloaded, {skipped} unavailable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--images",
        type=int,
        default=0,
        help="download photos for this many demo articles (needs the serving bundle)",
    )
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.images:
        import polars as pl

        recs = pl.read_parquet(cfg.paths.reports / "serving" / "demo_articles.parquet")
        download_images(cfg, recs["article_id"].head(a.images).to_list())
    else:
        download_csvs(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
