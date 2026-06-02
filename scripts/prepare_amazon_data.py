from __future__ import annotations

from pathlib import Path

import pandas as pd


PRODUCTS_URL = (
    "https://huggingface.co/datasets/am0507mu/Amazon-Reviews-Dataset/"
    "resolve/main/products.csv"
)
REVIEWS_URL = (
    "https://huggingface.co/datasets/am0507mu/Amazon-Reviews-Dataset/"
    "resolve/main/reviews.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MAX_PRODUCTS = 500
MAX_REVIEWS = 50_000


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def first_category(value: object) -> str:
    text = clean_text(value)
    if not text:
        return "Amazon"
    return text.split("->")[0].strip() or "Amazon"


def normalize_products(raw_products: pd.DataFrame, max_products: int) -> pd.DataFrame:
    products = raw_products.rename(
        columns={
            "asin": "product_id",
            "breadcrumbs": "category",
            "main_image": "image_url",
            "url": "product_url",
            "rating": "average_rating",
        }
    ).copy()

    products["product_id"] = products["product_id"].astype(str)
    products["title"] = products["title"].map(clean_text)
    products["description"] = products["description"].map(clean_text)
    products["category"] = products["category"].map(first_category)
    products["brand"] = products["title"].str.extract(r"^([A-Za-z0-9&' -]{2,35})", expand=False)
    products["brand"] = products["brand"].fillna("Amazon Seller").str.strip()
    products["image_url"] = products["image_url"].map(clean_text)
    products["product_url"] = products["product_url"].map(clean_text)
    products["average_rating"] = pd.to_numeric(
        products.get("average_rating", 0),
        errors="coerce",
    ).fillna(0)
    products["rating_count"] = pd.to_numeric(
        products.get("number_of_ratings", 0),
        errors="coerce",
    ).fillna(0).astype(int)
    products["price"] = 0.0

    products = products[
        products["product_id"].ne("")
        & products["title"].ne("")
        & products["image_url"].str.startswith("http")
    ]
    products = products.sort_values(
        ["rating_count", "average_rating"],
        ascending=False,
    ).head(max_products)

    return products[
        [
            "product_id",
            "title",
            "category",
            "brand",
            "price",
            "description",
            "image_url",
            "product_url",
            "average_rating",
            "rating_count",
        ]
    ]


def read_source_csv(path: Path, url: str, **kwargs: object) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    data = pd.read_csv(url, **kwargs)
    data.to_csv(path, index=False)
    return data


def normalize_interactions(
    raw_reviews: pd.DataFrame,
    product_ids: set[str],
) -> pd.DataFrame:
    reviews = raw_reviews.copy()
    if "asin" in reviews.columns:
        reviews["product_id"] = reviews["asin"]
    elif "product_asin" in reviews.columns:
        reviews["product_id"] = reviews["product_asin"]
    elif "parent_asin" in reviews.columns:
        reviews["product_id"] = reviews["parent_asin"]
    elif "id" in reviews.columns:
        reviews["product_id"] = reviews["id"]

    if "stars" in reviews.columns:
        reviews["rating"] = reviews["stars"]
    elif "star_rating" in reviews.columns:
        reviews["rating"] = reviews["star_rating"]

    if "product_id" not in reviews.columns:
        raise ValueError("reviews.csv must contain an id or asin column.")
    if "user_id" not in reviews.columns:
        raise ValueError("reviews.csv must contain a user_id column.")
    if "rating" not in reviews.columns:
        raise ValueError("reviews.csv must contain a stars or star_rating column.")

    reviews["product_id"] = reviews["product_id"].astype(str)
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews = reviews[
        reviews["product_id"].isin(product_ids)
        & reviews["user_id"].ne("")
        & reviews["rating"].between(1, 5)
    ]

    return reviews[["user_id", "product_id", "rating"]].drop_duplicates(
        ["user_id", "product_id"]
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    products_raw = read_source_csv(RAW_DIR / "amazon_products_source.csv", PRODUCTS_URL)
    products = normalize_products(products_raw, MAX_PRODUCTS)

    reviews_raw = read_source_csv(
        RAW_DIR / "amazon_reviews_source_sample.csv",
        REVIEWS_URL,
        nrows=MAX_REVIEWS,
    )
    interactions = normalize_interactions(reviews_raw, set(products["product_id"]))

    products = products[products["product_id"].isin(interactions["product_id"].unique())]
    if products.empty or interactions.empty:
        raise ValueError("No overlapping Amazon products and reviews were found.")

    products.to_csv(DATA_DIR / "products.csv", index=False)
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False)

    print(f"Wrote {len(products):,} products with images")
    print(f"Wrote {len(interactions):,} user-product ratings")
    print(f"Categories: {products['category'].nunique():,}")


if __name__ == "__main__":
    main()
