from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from socket import error as SocketError
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.recommender import (
    blended_recommendations,
    content_based_recommendations,
    create_engine,
    evaluate_hybrid_recommender,
    user_based_recommendations,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = PROJECT_ROOT / "web"
HOST = "127.0.0.1"
DEFAULT_PORT = 8501


@lru_cache(maxsize=1)
def get_engine():
    return create_engine()


METRICS_CACHE = PUBLIC_DIR / "model_metrics.json"


def get_model_metrics() -> dict[str, Any]:
    if METRICS_CACHE.exists():
        with open(METRICS_CACHE) as f:
            return json.load(f)
    engine = get_engine()
    metrics = evaluate_hybrid_recommender(
        engine.products,
        engine.interactions,
        k=5,
        content_weight=0.55,
        relevant_rating=4.0,
        max_users=200,
    )
    payload = asdict(metrics)
    with open(METRICS_CACHE, "w") as f:
        json.dump(payload, f)
    return payload


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def product_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    description = clean_text(row["description"])
    return {
        "product_id": clean_text(row["product_id"]),
        "title": clean_text(row["title"]),
        "category": clean_text(row["category"]),
        "brand": clean_text(row["brand"]),
        "price": float(row.get("price", 0) or 0),
        "description": description[:700],
        "image_url": clean_text(row.get("image_url", "")),
        "product_url": clean_text(row.get("product_url", "")),
        "average_rating": float(row.get("average_rating", 0) or 0),
        "rating_count": int(row.get("rating_count", 0) or 0),
    }


def product_option_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": clean_text(row["product_id"]),
        "title": clean_text(row["title"]),
        "category": clean_text(row["category"]),
        "brand": clean_text(row["brand"]),
    }


def frame_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        item = product_payload(row)
        for score in ("content_score", "collaborative_score", "hybrid_score"):
            if score in row and not pd.isna(row[score]):
                item[score] = float(row[score])
        items.append(item)
    return items


def bootstrap_payload() -> dict[str, Any]:
    engine = get_engine()
    products = engine.products.sort_values(["category", "title"])
    interactions = engine.interactions
    categories = sorted(products["category"].dropna().unique().tolist())
    active_users = (
        interactions.groupby("user_id")
        .size()
        .sort_values(ascending=False)
        .head(250)
        .reset_index(name="rating_count")
    )
    users = [
        {
            "user_id": clean_text(row["user_id"]),
            "label": f"User {index + 1} ({int(row['rating_count'])} ratings)",
        }
        for index, row in active_users.iterrows()
    ]

    return {
        "stats": {
            "products": len(products),
            "users": interactions["user_id"].nunique(),
            "ratings": len(interactions),
            "average_rating": round(float(interactions["rating"].mean()), 2),
            "categories": len(categories),
        },
        "model_metrics": get_model_metrics(),
        "products": [product_option_payload(row) for _, row in products.iterrows()],
        "categories": categories,
        "users": users,
    }


def recommendations_payload(params: dict[str, list[str]]) -> dict[str, Any]:
    engine = get_engine()
    products = engine.products
    fallback_product = clean_text(products.iloc[0]["product_id"])
    fallback_user = clean_text(engine.interactions.iloc[0]["user_id"])

    product_id = params.get("product_id", [fallback_product])[0]
    user_id = params.get("user_id", [fallback_user])[0]
    top_n = max(3, min(12, int(params.get("top_n", ["6"])[0])))
    content_weight = max(0.0, min(1.0, float(params.get("content_weight", ["0.55"])[0])))

    selected = products.loc[products["product_id"] == product_id]
    selected_product = product_payload(selected.iloc[0] if not selected.empty else products.iloc[0])

    content = content_based_recommendations(engine, product_id=product_id, top_n=top_n)
    user = user_based_recommendations(engine, user_id=user_id, top_n=top_n)
    hybrid = blended_recommendations(
        engine,
        product_id=product_id,
        user_id=user_id,
        top_n=top_n,
        content_weight=content_weight,
    )

    return {
        "selected_product": selected_product,
        "content": frame_payload(content),
        "user": frame_payload(user),
        "hybrid": frame_payload(hybrid),
    }


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            self.send_json(bootstrap_payload())
            return
        if parsed.path == "/api/recommendations":
            try:
                self.send_json(recommendations_payload(parse_qs(parsed.query)))
            except (KeyError, ValueError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()


def main() -> None:
    get_engine()
    server = None
    port = DEFAULT_PORT
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        try:
            server = ThreadingHTTPServer((HOST, candidate), AppHandler)
            port = candidate
            break
        except SocketError:
            continue
    if server is None:
        raise OSError(f"No open port found from {DEFAULT_PORT} to {DEFAULT_PORT + 19}")
    print(f"Amazon recommender running at http://{HOST}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
