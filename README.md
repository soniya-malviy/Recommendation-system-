# Amazon Product Recommendation System

A compact Amazon recommendation project built with Python, Pandas, scikit-learn, and a lightweight vanilla HTML/CSS/JS frontend.

The app combines two recommendation strategies:

- Content-based filtering with TF-IDF over product titles, categories, brands, and descriptions.
- User-based collaborative filtering with cosine similarity over user-product ratings.
- Product cards with real Amazon product image URLs.
- A no-build frontend served by Python's standard library, so deployment stays simple.

## Project Structure

```text
.
├── app.py
├── data
│   ├── interactions.csv
│   ├── products.csv
│   └── raw
├── requirements.txt
├── scripts
│   └── prepare_amazon_data.py
├── src
│   ├── __init__.py
│   └── recommender.py
├── web
│   ├── app.css
│   ├── app.js
│   └── index.html
└── tests
    └── test_recommender.py
```

## Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the dashboard:

```bash
python3 app.py
```

Then open `http://127.0.0.1:8501`.

## Refresh Amazon Data

The current data is prepared from the public Hugging Face dataset `am0507mu/Amazon-Reviews-Dataset`, which includes Amazon product metadata, image URLs, and customer review ratings.

Regenerate the normalized app CSVs:

```bash
python3 scripts/prepare_amazon_data.py
```

Source references:

- `products.csv`: Amazon URL, ASIN, title, category breadcrumbs, rating counts, main product image, and description.
- `reviews.csv`: anonymized user IDs, Amazon product IDs, and star ratings.
