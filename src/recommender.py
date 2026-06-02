from __future__ import annotations

from dataclasses import dataclass
from math import log2
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "data" / "products.csv"
DEFAULT_INTERACTIONS_PATH = PROJECT_ROOT / "data" / "interactions.csv"


@dataclass(frozen=True)
class RecommendationEngine:
    products: pd.DataFrame
    interactions: pd.DataFrame
    content_similarity: pd.DataFrame
    user_item_matrix: pd.DataFrame
    user_similarity: pd.DataFrame


@dataclass(frozen=True)
class RankingMetrics:
    evaluated_users: int
    k: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    map_at_k: float
    ndcg_at_k: float
    catalog_coverage: float


def load_data(
    products_path: Path | str = DEFAULT_PRODUCTS_PATH,
    interactions_path: Path | str = DEFAULT_INTERACTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load product metadata and user interaction data."""
    products = pd.read_csv(products_path)
    interactions = pd.read_csv(interactions_path)

    required_product_columns = {"product_id", "title", "category", "brand", "description"}
    required_interaction_columns = {"user_id", "product_id", "rating"}
    missing_products = required_product_columns.difference(products.columns)
    missing_interactions = required_interaction_columns.difference(interactions.columns)

    if missing_products:
        raise ValueError(f"products.csv is missing columns: {sorted(missing_products)}")
    if missing_interactions:
        raise ValueError(f"interactions.csv is missing columns: {sorted(missing_interactions)}")

    products = products.copy()
    interactions = interactions.copy()
    products["product_id"] = products["product_id"].astype(str)
    products["title"] = products["title"].fillna("")
    products["category"] = products["category"].fillna("Amazon")
    products["brand"] = products["brand"].fillna("Amazon Seller")
    products["description"] = products["description"].fillna("")
    if "price" not in products.columns:
        products["price"] = 0.0
    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0)
    if "image_url" not in products.columns:
        products["image_url"] = ""
    products["image_url"] = products["image_url"].fillna("")
    if "product_url" not in products.columns:
        products["product_url"] = ""
    products["product_url"] = products["product_url"].fillna("")
    if "average_rating" not in products.columns:
        products["average_rating"] = 0.0
    products["average_rating"] = pd.to_numeric(
        products["average_rating"],
        errors="coerce",
    ).fillna(0)
    if "rating_count" not in products.columns:
        products["rating_count"] = 0
    products["rating_count"] = pd.to_numeric(
        products["rating_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    interactions["product_id"] = interactions["product_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce").fillna(0)
    return products, interactions


def build_engine(products: pd.DataFrame, interactions: pd.DataFrame) -> RecommendationEngine:
    """Build reusable similarity matrices for content and collaborative recommendations."""
    product_text = (
        products["title"].fillna("")
        + " "
        + products["category"].fillna("")
        + " "
        + products["brand"].fillna("")
        + " "
        + products["description"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(product_text)
    content_similarity = pd.DataFrame(
        cosine_similarity(tfidf_matrix),
        index=products["product_id"],
        columns=products["product_id"],
    )

    user_item_matrix = interactions.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean",
        fill_value=0,
    )
    user_similarity = pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    return RecommendationEngine(
        products=products,
        interactions=interactions,
        content_similarity=content_similarity,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
    )


def create_engine(
    products_path: Path | str = DEFAULT_PRODUCTS_PATH,
    interactions_path: Path | str = DEFAULT_INTERACTIONS_PATH,
) -> RecommendationEngine:
    products, interactions = load_data(products_path, interactions_path)
    return build_engine(products, interactions)


def _format_recommendations(
    engine: RecommendationEngine,
    scores: pd.Series,
    score_name: str,
    top_n: int,
) -> pd.DataFrame:
    scored = scores.sort_values(ascending=False).head(top_n).reset_index()
    scored.columns = ["product_id", score_name]
    return scored.merge(engine.products, on="product_id", how="left")


def content_based_recommendations(
    engine: RecommendationEngine,
    product_id: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend products with similar metadata and descriptions."""
    if product_id not in engine.content_similarity.index:
        raise ValueError(f"Unknown product_id: {product_id}")

    scores = engine.content_similarity.loc[product_id].drop(labels=[product_id])
    return _format_recommendations(engine, scores, "content_score", top_n)


def user_based_recommendations(
    engine: RecommendationEngine,
    user_id: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend unrated products using ratings from the most similar users."""
    if user_id not in engine.user_similarity.index:
        raise ValueError(f"Unknown user_id: {user_id}")

    similar_users = engine.user_similarity.loc[user_id].drop(labels=[user_id])
    weighted_ratings = similar_users.dot(engine.user_item_matrix.loc[similar_users.index])
    normalizer = similar_users.abs().sum()
    predicted_scores = weighted_ratings / normalizer if normalizer else weighted_ratings

    already_rated = engine.user_item_matrix.loc[user_id]
    predicted_scores = predicted_scores[already_rated == 0]
    return _format_recommendations(engine, predicted_scores, "collaborative_score", top_n)


def blended_recommendations(
    engine: RecommendationEngine,
    product_id: str,
    user_id: str,
    top_n: int = 5,
    content_weight: float = 0.55,
) -> pd.DataFrame:
    """Blend product similarity with collaborative scores for a selected user."""
    if not 0 <= content_weight <= 1:
        raise ValueError("content_weight must be between 0 and 1")

    content_scores = engine.content_similarity.loc[product_id].drop(labels=[product_id])
    collaborative = user_based_recommendations(
        engine,
        user_id=user_id,
        top_n=len(engine.products),
    ).set_index("product_id")["collaborative_score"]

    scores = pd.DataFrame(index=engine.products["product_id"])
    scores["content_score"] = content_scores
    scores["collaborative_score"] = collaborative
    scores = scores.fillna(0)
    scores = scores.drop(index=product_id, errors="ignore")
    scores["hybrid_score"] = (
        content_weight * scores["content_score"]
        + (1 - content_weight) * scores["collaborative_score"]
    )

    result = scores.sort_values("hybrid_score", ascending=False).head(top_n).reset_index()
    return result.merge(engine.products, on="product_id", how="left")


def _holdout_relevant_items(
    interactions: pd.DataFrame,
    relevant_rating: float,
    max_users: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relevant = interactions.loc[interactions["rating"] >= relevant_rating].copy()
    eligible_users = relevant["user_id"].value_counts()
    eligible_users = eligible_users.loc[eligible_users >= 2].head(max_users).index
    relevant = relevant.loc[relevant["user_id"].isin(eligible_users)]

    test_rows = (
        relevant.sort_values(["user_id", "rating", "product_id"])
        .groupby("user_id", as_index=False)
        .tail(1)
    )
    test_keys = set(zip(test_rows["user_id"], test_rows["product_id"]))
    train_rows = interactions.loc[
        ~interactions.apply(lambda row: (row["user_id"], row["product_id"]) in test_keys, axis=1)
    ].copy()
    return train_rows, test_rows


def evaluate_hybrid_recommender(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int = 5,
    content_weight: float = 0.55,
    relevant_rating: float = 4.0,
    max_users: int = 200,
) -> RankingMetrics:
    """Evaluate hybrid ranking with a deterministic leave-one-relevant-item holdout."""
    train_interactions, test_interactions = _holdout_relevant_items(
        interactions,
        relevant_rating=relevant_rating,
        max_users=max_users,
    )
    eval_engine = build_engine(products, train_interactions)

    hits = 0
    reciprocal_ranks: list[float] = []
    discounted_gains: list[float] = []
    recommended_products: set[str] = set()

    for row in test_interactions.itertuples():
        user_id = str(row.user_id)
        heldout_product = str(row.product_id)
        user_history = train_interactions.loc[train_interactions["user_id"] == user_id]
        if user_history.empty or user_id not in eval_engine.user_similarity.index:
            continue

        seed_product = str(
            user_history.sort_values(["rating", "product_id"], ascending=[False, True]).iloc[0][
                "product_id"
            ]
        )
        recommendations = blended_recommendations(
            eval_engine,
            product_id=seed_product,
            user_id=user_id,
            top_n=k,
            content_weight=content_weight,
        )
        ranked_ids = recommendations["product_id"].astype(str).tolist()
        recommended_products.update(ranked_ids)

        if heldout_product in ranked_ids:
            hits += 1
            rank = ranked_ids.index(heldout_product) + 1
            reciprocal_ranks.append(1 / rank)
            discounted_gains.append(1 / log2(rank + 1))
        else:
            reciprocal_ranks.append(0)
            discounted_gains.append(0)

    evaluated_users = len(test_interactions)
    if evaluated_users == 0:
        return RankingMetrics(0, k, 0, 0, 0, 0, 0, 0, 0)

    precision = hits / (evaluated_users * k)
    recall = hits / evaluated_users
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return RankingMetrics(
        evaluated_users=evaluated_users,
        k=k,
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        hit_rate_at_k=recall,
        map_at_k=sum(reciprocal_ranks) / evaluated_users,
        ndcg_at_k=sum(discounted_gains) / evaluated_users,
        catalog_coverage=len(recommended_products) / len(products),
    )
