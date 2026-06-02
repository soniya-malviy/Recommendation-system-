import pytest

from src.recommender import (
    blended_recommendations,
    content_based_recommendations,
    create_engine,
    evaluate_hybrid_recommender,
    user_based_recommendations,
)


@pytest.fixture(scope="module")
def engine():
    return create_engine()


@pytest.fixture(scope="module")
def product_id(engine):
    return engine.products["product_id"].iloc[0]


@pytest.fixture(scope="module")
def user_id(engine):
    user_counts = engine.interactions["user_id"].value_counts()
    return user_counts.index[0]


def test_content_based_recommendations_exclude_selected_product(engine, product_id):
    recommendations = content_based_recommendations(engine, product_id, top_n=5)

    assert len(recommendations) == 5
    assert product_id not in set(recommendations["product_id"])
    assert recommendations["image_url"].str.startswith("http").all()
    assert recommendations["content_score"].is_monotonic_decreasing


def test_user_based_recommendations_exclude_seen_products(engine, user_id):
    recommendations = user_based_recommendations(engine, user_id, top_n=5)
    seen_products = set(
        engine.interactions.loc[engine.interactions["user_id"] == user_id, "product_id"]
    )

    assert len(recommendations) == 5
    assert seen_products.isdisjoint(set(recommendations["product_id"]))
    assert recommendations["collaborative_score"].ge(0).all()


def test_blended_recommendations_include_all_score_columns(engine, product_id, user_id):
    recommendations = blended_recommendations(engine, product_id, user_id, top_n=5)

    assert len(recommendations) == 5
    assert {"hybrid_score", "content_score", "collaborative_score"}.issubset(
        recommendations.columns
    )
    assert recommendations["hybrid_score"].is_monotonic_decreasing


def test_unknown_ids_raise_clear_errors(engine):
    with pytest.raises(ValueError, match="Unknown product_id"):
        content_based_recommendations(engine, "missing")

    with pytest.raises(ValueError, match="Unknown user_id"):
        user_based_recommendations(engine, "missing")


def test_evaluate_hybrid_recommender_returns_ranking_metrics(engine):
    metrics = evaluate_hybrid_recommender(
        engine.products,
        engine.interactions,
        k=3,
        max_users=10,
    )

    assert metrics.evaluated_users > 0
    assert metrics.k == 3
    assert 0 <= metrics.precision_at_k <= 1
    assert 0 <= metrics.recall_at_k <= 1
    assert 0 <= metrics.f1_at_k <= 1
    assert 0 <= metrics.map_at_k <= 1
    assert 0 <= metrics.ndcg_at_k <= 1
    assert 0 <= metrics.catalog_coverage <= 1
