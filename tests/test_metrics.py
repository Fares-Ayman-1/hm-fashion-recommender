from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hmrec.metrics import ap_at_k, candidate_recall, map_at_k


def test_ap_hand_calculated() -> None:
    # hits at ranks 1 and 3 of 3 bought items: (1/1 + 2/3) / min(3, 12) = 5/9
    assert ap_at_k([10, 20, 30], [10, 99, 20, 98], k=12) == pytest.approx(5 / 9)
    # one bought item found at rank 2: (1/2) / 1
    assert ap_at_k([7], [1, 7, 2], k=12) == pytest.approx(0.5)
    # duplicates in the prediction never count twice
    assert ap_at_k([7], [7, 7, 7], k=12) == pytest.approx(1.0)
    assert ap_at_k([1, 2], [3, 4], k=12) == 0.0


def test_denominator_caps_at_k() -> None:
    actual = list(range(20))
    assert ap_at_k(actual, list(range(12)), k=12) == pytest.approx(1.0)


def test_vectorised_matches_reference() -> None:
    rng = np.random.default_rng(3)
    pred_rows, act_rows, ref = [], [], []
    for c in range(200):
        bought = list(set(rng.integers(0, 30, rng.integers(1, 6)).tolist()))
        recs = rng.permutation(30)[:12].tolist()
        act_rows += [(c, a) for a in bought]
        pred_rows += [(c, a, r) for r, a in enumerate(recs, start=1)]
        ref.append(ap_at_k(bought, recs))
    pred = pl.DataFrame(pred_rows, schema=["customer", "article_id", "rank"], orient="row")
    actual = pl.DataFrame(act_rows, schema=["customer", "article_id"], orient="row")
    assert map_at_k(pred, actual) == pytest.approx(float(np.mean(ref)))


def test_customers_without_predictions_score_zero() -> None:
    pred = pl.DataFrame({"customer": [1], "article_id": [5], "rank": [1]})
    actual = pl.DataFrame({"customer": [1, 2], "article_id": [5, 6]})
    assert map_at_k(pred, actual) == pytest.approx(0.5)


def test_candidate_recall() -> None:
    cands = pl.DataFrame({"customer": [1, 1, 2], "article_id": [5, 6, 7]})
    actual = pl.DataFrame({"customer": [1, 2, 2], "article_id": [5, 8, 7]})
    assert candidate_recall(cands, actual) == pytest.approx(2 / 3)
