"""Submit reports/submission.csv to Kaggle (late submission) and record the score.

The CSV (1.37M rows, ~258 MB) is zipped first; Kaggle accepts compressed submissions.
Writes reports/kaggle_score.json, which `python -m hmrec.report` adds to RESULTS.md.

    uv run python scripts/kaggle_submit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmrec.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--message", default="Two-stage: 4 candidate sources + LightGBM LambdaRank")
    a = ap.parse_args()
    cfg = load_config()
    sub = cfg.paths.reports / "submission.csv"
    if not sub.exists():
        print("no submission.csv; run `python -m hmrec.train` first", file=sys.stderr)
        return 1
    zipped = sub.with_suffix(".zip")
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(sub, arcname="submission.csv")
    print(f"zipped {sub.stat().st_size / 1e6:.0f} MB -> {zipped.stat().st_size / 1e6:.0f} MB", flush=True)

    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    t0 = time.perf_counter()
    api.competition_submit(str(zipped), a.message, cfg.competition, quiet=True)
    print(f"uploaded in {(time.perf_counter() - t0) / 60:.1f} min; waiting for the score...", flush=True)

    for _ in range(120):
        time.sleep(15)
        subs = api.competition_submissions(cfg.competition) or []
        latest = subs[0] if subs else None
        status = str(getattr(latest, "status", "")).lower()
        if latest is not None and "complete" in status:
            score = {
                "private": getattr(latest, "private_score", None),
                "public": getattr(latest, "public_score", None),
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "description": getattr(latest, "description", a.message),
            }
            (cfg.paths.reports / "kaggle_score.json").write_text(
                json.dumps(score, indent=2), encoding="utf-8"
            )
            print(json.dumps(score, indent=2))
            return 0
        if latest is not None and "error" in status:
            print(
                f"Kaggle reported an error: {getattr(latest, 'error_description', status)}", file=sys.stderr
            )
            return 1
    print("timed out waiting for the score; check the Submissions tab on Kaggle", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
