const state = {
  products: [],
  filteredProducts: [],
  categories: [],
  users: [],
  activeTab: "hybrid",
  recommendations: { hybrid: [], content: [], user: [] },
};

const els = {
  search: document.querySelector("#searchInput"),
  product: document.querySelector("#productSelect"),
  user: document.querySelector("#userSelect"),
  category: document.querySelector("#categorySelect"),
  topN: document.querySelector("#topNInput"),
  weight: document.querySelector("#weightInput"),
  stats: document.querySelector("#stats"),
  metrics: document.querySelector("#modelMetrics"),
  results: document.querySelector("#results"),
  selected: document.querySelector("#selectedProduct"),
  productCount: document.querySelector("#productCount"),
  tabs: document.querySelectorAll(".tab"),
};

const formatter = new Intl.NumberFormat("en-US");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function rating(product) {
  return `${Number(product.average_rating || 0).toFixed(1)} (${formatter.format(
    product.rating_count || 0,
  )})`;
}

function scoreText(product) {
  const scores = [
    ["Hybrid", product.hybrid_score],
    ["Content", product.content_score],
    ["User", product.collaborative_score],
  ].filter(([, value]) => Number.isFinite(value));
  return scores.map(([label, value]) => `${label} ${value.toFixed(3)}`).join(" · ");
}

function imageMarkup(product, className) {
  if (!product.image_url) {
    return `<div class="${className}"><span class="pill">No image</span></div>`;
  }
  return `<div class="${className}"><img src="${escapeHtml(product.image_url)}" alt="${escapeHtml(
    product.title,
  )}" loading="lazy" /></div>`;
}

function renderStats(stats) {
  const items = [
    ["Products", stats.products],
    ["Users", stats.users],
    ["Ratings", stats.ratings],
    ["Avg rating", stats.average_rating],
    ["Categories", stats.categories],
  ];
  els.stats.innerHTML = items
    .map(
      ([label, value]) => `
        <article class="stat">
          <span>${label}</span>
          <strong>${formatter.format(value)}</strong>
        </article>
      `,
    )
    .join("");
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function renderModelMetrics(metrics) {
  const items = [
    ["Precision@5", percent(metrics.precision_at_k)],
    ["Recall@5", percent(metrics.recall_at_k)],
    ["F1@5", percent(metrics.f1_at_k)],
    ["MAP@5", metrics.map_at_k.toFixed(3)],
    ["NDCG@5", metrics.ndcg_at_k.toFixed(3)],
    ["Coverage", percent(metrics.catalog_coverage)],
  ];
  els.metrics.innerHTML = `
    <div class="metrics-heading">
      <div>
        <span class="eyebrow">Model evaluation</span>
        <h2>Hybrid recommender metrics</h2>
      </div>
      <span class="metric-note">Leave-one-out holdout · ${formatter.format(metrics.evaluated_users)} users</span>
    </div>
    <div class="metric-grid">
      ${items
        .map(
          ([label, value]) => `
            <article class="metric-tile">
              <span>${label}</span>
              <strong>${value}</strong>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function populateSelect(select, options, valueKey, labelKey) {
  select.innerHTML = options
    .map((item) => {
      const value = typeof item === "string" ? item : item[valueKey];
      const label = typeof item === "string" ? item : item[labelKey];
      return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
    })
    .join("");
}

function productLabel(product) {
  return `${product.title} (${product.category})`;
}

function applyProductFilters() {
  const query = els.search.value.trim().toLowerCase();
  const category = els.category.value;
  const previous = els.product.value;
  state.filteredProducts = state.products.filter((product) => {
    const matchesCategory = !category || product.category === category;
    const haystack = `${product.title} ${product.brand} ${product.category}`.toLowerCase();
    return matchesCategory && (!query || haystack.includes(query));
  });
  if (!state.filteredProducts.length) {
    els.product.innerHTML = `<option value="">No matching products</option>`;
    els.productCount.textContent = "0 shown";
    return;
  }
  populateSelect(
    els.product,
    state.filteredProducts.map((product) => ({
      value: product.product_id,
      label: productLabel(product),
    })),
    "value",
    "label",
  );
  if (state.filteredProducts.some((product) => product.product_id === previous)) {
    els.product.value = previous;
  }
  els.productCount.textContent = `${state.filteredProducts.length} shown`;
}

function renderSelected(product) {
  els.selected.innerHTML = `
    ${imageMarkup(product, "selected-image")}
    <div class="selected-body">
      <div class="meta-row">
        <span class="pill">${escapeHtml(product.category)}</span>
        <span class="pill rating">${rating(product)}</span>
      </div>
      <h3>${escapeHtml(product.title)}</h3>
      <p class="description">${escapeHtml(product.description)}</p>
      ${
        product.product_url
          ? `<a class="amazon-link" href="${escapeHtml(product.product_url)}" target="_blank" rel="noreferrer">View product</a>`
          : ""
      }
    </div>
  `;
}

function renderCards() {
  const items = state.recommendations[state.activeTab] || [];
  if (!items.length) {
    els.results.innerHTML = `<div class="loading">No recommendations found for this selection.</div>`;
    return;
  }
  els.results.innerHTML = items
    .map(
      (product, index) => `
        <article class="product-card">
          ${imageMarkup(product, "card-image")}
          <div class="card-body">
            <span class="rank">Rank ${index + 1}</span>
            <h3 class="card-title">${escapeHtml(product.title)}</h3>
            <div class="meta-row">
              <span class="pill">${escapeHtml(product.brand || "Amazon Seller")}</span>
              <span class="pill rating">${rating(product)}</span>
            </div>
            <p class="description">${escapeHtml(product.description)}</p>
            <div class="card-footer">
              <span class="score">${escapeHtml(scoreText(product))}</span>
              ${
                product.product_url
                  ? `<a class="amazon-link" href="${escapeHtml(product.product_url)}" target="_blank" rel="noreferrer">Open</a>`
                  : ""
              }
            </div>
          </div>
        </article>
      `,
    )
    .join("");
}

async function loadRecommendations() {
  if (!els.product.value || !els.user.value) {
    return;
  }
  els.results.innerHTML = `<div class="loading">Building recommendations...</div>`;
  const params = new URLSearchParams({
    product_id: els.product.value,
    user_id: els.user.value,
    top_n: els.topN.value,
    content_weight: els.weight.value,
  });
  const response = await fetch(`/api/recommendations?${params}`);
  if (!response.ok) {
    const error = await response.json();
    els.results.innerHTML = `<div class="loading">${escapeHtml(error.error || "Request failed")}</div>`;
    return;
  }
  const data = await response.json();
  state.recommendations = {
    hybrid: data.hybrid,
    content: data.content,
    user: data.user,
  };
  renderSelected(data.selected_product);
  renderCards();
}

function bindEvents() {
  [els.search, els.category].forEach((input) => {
    input.addEventListener("input", () => {
      applyProductFilters();
      loadRecommendations();
    });
  });
  [els.product, els.user, els.topN, els.weight].forEach((input) => {
    input.addEventListener("change", loadRecommendations);
  });
  els.tabs.forEach((button) => {
    button.addEventListener("click", () => {
      els.tabs.forEach((tab) => tab.classList.remove("is-active"));
      button.classList.add("is-active");
      state.activeTab = button.dataset.tab;
      renderCards();
    });
  });
}

async function init() {
  els.results.innerHTML = `<div class="loading">Loading product workspace...</div>`;
  const response = await fetch("/api/bootstrap");
  const data = await response.json();
  state.products = data.products;
  state.categories = data.categories;
  state.users = data.users;

  renderStats(data.stats);
  renderModelMetrics(data.model_metrics);
  populateSelect(
    els.category,
    [{ value: "", label: "All categories" }, ...state.categories.map((category) => ({ value: category, label: category }))],
    "value",
    "label",
  );
  populateSelect(
    els.user,
    state.users.map((user) => ({ value: user.user_id, label: user.label })),
    "value",
    "label",
  );
  applyProductFilters();
  bindEvents();
  loadRecommendations();
}

init();
