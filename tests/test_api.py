"""API contract: success shapes, one error envelope, pagination, unknown params, images."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hmrec.config import Config


@pytest.fixture(scope="module")
def client(trained: tuple[Path, Config, dict[str, object]]) -> Iterator[TestClient]:
    _, cfg, _ = trained
    mp = pytest.MonkeyPatch()
    mp.setenv("SERVING_DIR", str(cfg.paths.reports / "serving"))
    cfg.paths.thumbs.mkdir(parents=True, exist_ok=True)
    mp.setenv("THUMBS_DIR", str(cfg.paths.thumbs))
    from hmrec.serve import api

    api.get_store.cache_clear()
    with TestClient(api.app) as c:
        yield c
    api.get_store.cache_clear()
    mp.undo()


def assert_error(resp, status: int, code: str) -> None:  # type: ignore[no-untyped-def]
    assert resp.status_code == status
    body = resp.json()
    assert set(body) == {"error"} and body["error"]["code"] == code


def test_model_and_customers(client: TestClient) -> None:
    m = client.get("/v1/model").json()
    assert set(m["map12"]) == {"lgbm_ranker", "baseline_popular", "baseline_repurchase_popular"}
    page = client.get("/v1/customers", params={"limit": 3, "offset": 1}).json()
    assert len(page["data"]) == 3 and page["pagination"]["offset"] == 1


def test_recommendations_with_reasons(client: TestClient) -> None:
    cid = client.get("/v1/customers", params={"limit": 1}).json()["data"][0]
    body = client.get(f"/v1/customers/{cid}/recommendations", params={"k": 5}).json()
    assert [r["rank"] for r in body["data"]] == [1, 2, 3, 4, 5]
    allowed = {"repurchase", "popular", "cooccur", "similar"}
    for r in body["data"]:
        assert r["reasons"] and set(r["reasons"]) <= allowed and r["explanation"]
    hist = client.get(f"/v1/customers/{cid}/purchases", params={"limit": 3}).json()
    assert 1 <= len(hist["data"]) <= 3


def test_articles_similar_and_image(
    client: TestClient, trained: tuple[Path, Config, dict[str, object]]
) -> None:
    _, cfg, _ = trained
    cid = client.get("/v1/customers", params={"limit": 1}).json()["data"][0]
    aid = client.get(f"/v1/customers/{cid}/recommendations").json()["data"][0]["article"]["article_id"]
    assert client.get(f"/v1/articles/{aid}").json()["article_id"] == aid
    sim = client.get(f"/v1/articles/{aid}/similar", params={"k": 3}).json()
    assert len(sim["data"]) <= 3 and all(-1 <= s["similarity"] <= 1 for s in sim["data"])
    assert_error(client.get(f"/v1/articles/{aid}/image"), 404, "image_not_available")
    from PIL import Image

    Image.new("RGB", (8, 8)).save(cfg.paths.thumbs / f"{aid:010d}.jpg")
    r = client.get(f"/v1/articles/{aid}/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_errors_share_one_envelope(client: TestClient) -> None:
    assert_error(client.get("/v1/customers/nope/recommendations"), 404, "customer_not_found")
    assert_error(client.get("/v1/articles/1"), 404, "article_not_found")
    assert_error(client.get("/v1/customers", params={"limit": 500}), 422, "validation_error")
    assert_error(client.get("/v1/customers", params={"page": 2}), 422, "unknown_query_parameter")
    assert_error(client.post("/v1/customers"), 405, "method_not_allowed")
