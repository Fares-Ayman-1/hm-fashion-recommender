"""Read-only recommendation API over the serving bundle written by ``hmrec.train``.

Recommendations are computed in batch for every customer (the H&M setting: a weekly refresh) and
served from memory; each item says which candidate sources suggested it.

    SERVING_DIR=reports/serving THUMBS_DIR=data/thumbs uv run uvicorn hmrec.serve.api:app --port 8010
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from hmrec import __version__

ROOT = Path(__file__).resolve().parents[3]
REASON_TEXT = {
    "repurchase": "You bought this recently",
    "popular": "Popular last week with customers your age",
    "cooccur": "Often bought together with your recent purchases",
    "similar": "Similar to items you bought",
}


# ------------------------------------------------------------------------------------- data store
class Store:
    def __init__(self, root: Path, thumbs: Path):
        if not (root / "recommendations.parquet").exists():
            raise FileNotFoundError(f"no serving bundle in {root}; run `python -m hmrec.train` first")
        self.thumbs = thumbs
        recs = pl.read_parquet(root / "recommendations.parquet")
        self.recs = dict(
            zip(
                recs["customer_id"].to_list(),
                zip(recs["article_id"].to_list(), recs["reasons"].to_list(), strict=True),
                strict=True,
            )
        )
        self.articles = {
            r["article_id"]: r for r in pl.read_parquet(root / "articles.parquet").iter_rows(named=True)
        }
        hist = pl.read_parquet(root / "history.parquet")
        self.history = {k[0]: g for k, g in hist.group_by("customer_id", maintain_order=True)}
        self.demo_customers = sorted(self.history)
        sim = root / "similar.parquet"
        self.similar: dict[int, list[tuple[int, float]]] = {}
        if sim.exists():
            for k, g in (
                pl.read_parquet(sim)
                .sort(["article_id", "sim_score"], descending=[False, True])
                .group_by("article_id", maintain_order=True)
            ):
                self.similar[int(k[0])] = list(
                    zip(g["neighbor"].to_list(), g["sim_score"].to_list(), strict=True)
                )
        self.metrics: dict[str, Any] = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    def has_image(self, a: int) -> bool:
        return (self.thumbs / f"{a:010d}.jpg").exists()


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(
        Path(os.environ.get("SERVING_DIR", ROOT / "reports" / "serving")),
        Path(os.environ.get("THUMBS_DIR", ROOT / "data" / "thumbs")),
    )


# ------------------------------------------------------------------------------------- schemas
class ErrorBody(BaseModel):
    code: str = Field(examples=["customer_not_found"])
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Article(BaseModel):
    article_id: int = Field(examples=[706016001])
    name: str | None
    product_type: str | None
    colour: str | None
    section: str | None
    image_url: str | None = Field(description="Thumbnail URL, or null when no photo is available")


class Recommendation(BaseModel):
    rank: int
    article: Article
    reasons: list[str] = Field(examples=[["repurchase", "cooccur"]])
    explanation: str


class RecommendationResponse(BaseModel):
    customer_id: str
    data: list[Recommendation]


class Purchase(BaseModel):
    date: date
    article: Article


class PurchasesResponse(BaseModel):
    customer_id: str
    data: list[Purchase]


class SimilarItem(BaseModel):
    article: Article
    similarity: float


class SimilarResponse(BaseModel):
    article_id: int
    data: list[SimilarItem]


class Pagination(BaseModel):
    limit: int
    offset: int
    total: int


class CustomerPage(BaseModel):
    data: list[str]
    pagination: Pagination


class ModelInfo(BaseModel):
    api_version: str
    model: str
    trained_at: str
    k: int
    validation_week: int
    map12: dict[str, float] = Field(description="MAP@12 on the validation week per model (higher is better)")
    candidate_recall: dict[str, float]
    mlflow_run_ids: dict[str, str]


# ------------------------------------------------------------------------------------- app + errors
app = FastAPI(
    title="H&M Fashion Recommender API",
    version=__version__,
    description="Top-12 personalised recommendations for 1.37M H&M customers, "
    "with the reason behind each item. "
    'Read-only. Errors always use `{"error": {"code", "message", "details"}}`.',
)
ERRORS: dict[int | str, dict[str, Any]] = {
    404: {
        "model": ErrorResponse,
        "description": "`customer_not_found`, `article_not_found`, `history_not_available`, "
        "`image_not_available`",
    },
    422: {"model": ErrorResponse, "description": "`validation_error` or `unknown_query_parameter`"},
}


def _err(status: int, code: str, message: str, details: list[dict[str, Any]] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message, "details": details}}
    )


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"field": ".".join(str(p) for p in e["loc"]), "issue": e["msg"]} for e in exc.errors()]
    return _err(422, "validation_error", "The request is invalid.", details)


@app.exception_handler(StarletteHTTPException)
async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return _err(exc.status_code, exc.detail["code"], exc.detail["message"], exc.detail.get("details"))
    codes = {404: "not_found", 405: "method_not_allowed"}
    return _err(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
    return _err(500, "internal_error", "Something went wrong on our side.")


def allow_params(*names: str) -> Callable[[Request], None]:
    def check(request: Request) -> None:
        unknown = sorted(set(request.query_params) - set(names))
        if unknown:
            raise HTTPException(
                422,
                {
                    "code": "unknown_query_parameter",
                    "message": f"Unknown query parameter(s): {', '.join(unknown)}.",
                    "details": [{"field": u} for u in unknown],
                },
            )

    return check


def customer_or_404(customer_id: str) -> str:
    if customer_id not in get_store().recs:
        raise HTTPException(
            404, {"code": "customer_not_found", "message": f"No customer with id '{customer_id}'."}
        )
    return customer_id


def article_or_404(article_id: int) -> int:
    if article_id not in get_store().articles:
        raise HTTPException(
            404, {"code": "article_not_found", "message": f"No article with id {article_id}."}
        )
    return article_id


def to_article(a: int) -> Article:
    s = get_store()
    r = s.articles.get(a, {})
    return Article(
        article_id=a,
        name=r.get("prod_name"),
        product_type=r.get("product_type_name"),
        colour=r.get("colour_group_name"),
        section=r.get("section_name"),
        image_url=f"/v1/articles/{a}/image" if s.has_image(a) else None,
    )


# ------------------------------------------------------------------------------------- routes
@app.get("/health", tags=["meta"], summary="Liveness check")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/v1/model",
    tags=["meta"],
    summary="Model version and validation scores",
    response_model=ModelInfo,
    dependencies=[Depends(allow_params())],
)
def model_info() -> ModelInfo:
    m = get_store().metrics
    return ModelInfo(
        api_version=__version__,
        model="LightGBM LambdaRank over 4 candidate sources",
        trained_at=m["trained_at"],
        k=m["k"],
        validation_week=m["valid_week"],
        map12=m["map12"],
        candidate_recall=m["candidate_recall"],
        mlflow_run_ids=m["mlflow_run_ids"],
    )


@app.get(
    "/v1/customers",
    tags=["customers"],
    summary="List demo customers (those with purchase history served)",
    response_model=CustomerPage,
    responses=ERRORS,
    dependencies=[Depends(allow_params("limit", "offset"))],
)
def list_customers(
    limit: Annotated[int, Query(ge=1, le=100)] = 20, offset: Annotated[int, Query(ge=0)] = 0
) -> CustomerPage:
    ids = get_store().demo_customers
    return CustomerPage(
        data=ids[offset : offset + limit], pagination=Pagination(limit=limit, offset=offset, total=len(ids))
    )


@app.get(
    "/v1/customers/{customer_id}/recommendations",
    tags=["customers"],
    summary="Personalised top-k",
    response_model=RecommendationResponse,
    responses=ERRORS,
    dependencies=[Depends(allow_params("k"))],
)
def recommendations(
    customer_id: Annotated[str, Depends(customer_or_404)], k: Annotated[int, Query(ge=1, le=12)] = 12
) -> RecommendationResponse:
    arts, reasons = get_store().recs[customer_id]
    data = []
    for i, (a, r) in enumerate(zip(arts[:k], reasons[:k], strict=True), start=1):
        rs = [x for x in r.split(",") if x]
        expl = "; ".join(REASON_TEXT[x] for x in rs) if rs else "Popular this week"
        data.append(
            Recommendation(rank=i, article=to_article(int(a)), reasons=rs or ["popular"], explanation=expl)
        )
    return RecommendationResponse(customer_id=customer_id, data=data)


@app.get(
    "/v1/customers/{customer_id}/purchases",
    tags=["customers"],
    summary="Recent purchases (demo customers)",
    response_model=PurchasesResponse,
    responses=ERRORS,
    dependencies=[Depends(allow_params("limit"))],
)
def purchases(
    customer_id: Annotated[str, Depends(customer_or_404)], limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> PurchasesResponse:
    g = get_store().history.get(customer_id)
    if g is None:
        raise HTTPException(
            404,
            {
                "code": "history_not_available",
                "message": "Purchase history is only served for demo customers (see /v1/customers).",
            },
        )
    rows = g.head(limit).iter_rows(named=True)
    return PurchasesResponse(
        customer_id=customer_id,
        data=[Purchase(date=r["t_dat"], article=to_article(r["article_id"])) for r in rows],
    )


@app.get(
    "/v1/articles/{article_id}",
    tags=["articles"],
    summary="Article details",
    response_model=Article,
    responses=ERRORS,
    dependencies=[Depends(allow_params())],
)
def article(article_id: Annotated[int, Depends(article_or_404)]) -> Article:
    return to_article(article_id)


@app.get(
    "/v1/articles/{article_id}/similar",
    tags=["articles"],
    summary="Most similar articles (CLIP)",
    response_model=SimilarResponse,
    responses=ERRORS,
    dependencies=[Depends(allow_params("k"))],
)
def similar(
    article_id: Annotated[int, Depends(article_or_404)], k: Annotated[int, Query(ge=1, le=12)] = 6
) -> SimilarResponse:
    nb = get_store().similar.get(article_id, [])[:k]
    return SimilarResponse(
        article_id=article_id,
        data=[SimilarItem(article=to_article(int(a)), similarity=round(float(s), 4)) for a, s in nb],
    )


@app.get(
    "/v1/articles/{article_id}/image",
    tags=["articles"],
    summary="Article thumbnail (JPEG)",
    responses={404: ERRORS[404]},
    dependencies=[Depends(allow_params())],
    response_class=FileResponse,
)
def image(article_id: Annotated[int, Depends(article_or_404)]) -> FileResponse:
    p = get_store().thumbs / f"{article_id:010d}.jpg"
    if not p.exists():
        raise HTTPException(
            404, {"code": "image_not_available", "message": f"No photo for article {article_id}."}
        )
    return FileResponse(p, media_type="image/jpeg")
