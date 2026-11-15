"""CLIP embeddings for every article (GPU if available).

Text (default): product name, type, colour, pattern and description through the CLIP text encoder.
With ``--images``: the article thumbnail through the CLIP image encoder is added, and the embedding
is the L2-normalised mean of the two. Image mode needs thumbnails for (nearly) every article; with the
28.7 GB photo archive impractical to download here, the default is text-only so all articles are
embedded the same way.
Output: data/interim/article_emb.npy (float16, n x 512) and article_emb_ids.npy.

    uv sync --extra embed && uv run python -m hmrec.embed [--images]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import polars as pl

from hmrec.config import Config, load_config

MODEL = "openai/clip-vit-base-patch32"


def article_text(articles: pl.DataFrame) -> list[str]:
    cols = ["prod_name", "product_type_name", "colour_group_name", "graphical_appearance_name", "detail_desc"]
    df = articles.select([pl.col(c).cast(pl.String).fill_null("") for c in cols])
    return [". ".join(x for x in row if x)[:300] for row in df.iter_rows()]


def _features(out: Any) -> Any:
    """Projected embedding: a tensor in transformers < 5, ``.pooler_output`` of the output in >= 5."""
    return out if hasattr(out, "float") else out.pooler_output


def run(cfg: Config, batch: int = 256, use_images: bool = False) -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip: Any = CLIPModel.from_pretrained(MODEL)  # Any: the typed stubs vary between transformers versions
    model = clip.to(device).eval()
    if device == "cuda":
        model = model.half()
    proc = CLIPProcessor.from_pretrained(MODEL)

    articles = pl.read_parquet(cfg.paths.interim / "articles.parquet").sort("article_id")
    ids = articles["article_id"].to_numpy()
    texts = article_text(articles)
    n = len(ids)
    out = np.zeros((n, 512), dtype=np.float32)
    has_img = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            tok = proc(text=texts[sl], return_tensors="pt", padding=True, truncation=True, max_length=77).to(
                device
            )
            t = _features(model.get_text_features(**tok)).float()
            t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()

            imgs, idx = [], []
            for j, a in enumerate(ids[sl]):
                p = cfg.paths.thumbs / f"{int(a):010d}.jpg"
                if use_images and p.exists():
                    imgs.append(Image.open(p).convert("RGB"))
                    idx.append(j)
            v = np.zeros_like(t)
            if imgs:
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                if device == "cuda":
                    px = px.half()
                e = _features(model.get_image_features(pixel_values=px)).float()
                v[idx] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
                has_img[i + np.array(idx)] = True
            emb = t + v
            out[sl] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            if (i // batch) % 50 == 0:
                print(f"  {i + len(texts[sl])}/{n} articles, {time.perf_counter() - t0:.0f}s", flush=True)

    np.save(cfg.paths.interim / "article_emb.npy", out.astype(np.float16))
    np.save(cfg.paths.interim / "article_emb_ids.npy", ids.astype(np.int32))
    print(
        f"embeddings: {n} articles ({has_img.sum()} with images) on {device} "
        f"in {time.perf_counter() - t0:.0f}s"
    )


def load(cfg: Config) -> tuple[np.ndarray, np.ndarray] | None:
    f = cfg.paths.interim / "article_emb.npy"
    if not f.exists():
        return None
    return np.load(f).astype(np.float32), np.load(cfg.paths.interim / "article_emb_ids.npy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--images", action="store_true", help="also use the CLIP image encoder on thumbnails")
    a = ap.parse_args()
    run(load_config(a.config), a.batch, a.images)


if __name__ == "__main__":
    main()
