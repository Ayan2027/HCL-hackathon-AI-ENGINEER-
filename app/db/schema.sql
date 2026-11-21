-- ShopSense AI - SQLite schema (FROZEN CONTRACT)
-- Safe to run many times: every statement uses IF NOT EXISTS.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------
-- Product catalogue
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,                  -- P-1001
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    subcategory     TEXT,
    brand           TEXT,                              -- NULL allowed (messy data)
    price           REAL CHECK (price IS NULL OR price >= 0),
    currency        TEXT NOT NULL DEFAULT 'INR',
    stock_qty       INTEGER NOT NULL DEFAULT 0 CHECK (stock_qty >= 0),
    rating          REAL CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5)),
    rating_count    INTEGER,
    attributes_json TEXT NOT NULL DEFAULT '{}',        -- {"color":"Blue","sizes_available":[8,9]}
    description     TEXT,
    review_summary  TEXT,
    reviews_json    TEXT NOT NULL DEFAULT '[]',
    best_for        TEXT,
    weight_raw      TEXT,                              -- as supplied: "350 g", "0.95 kg"
    weight_g        REAL,                              -- normalised by code
    language        TEXT NOT NULL DEFAULT 'en',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    -- ShopMate catalogue fields (also added to older DBs by database.migrate_schema())
    retail_price         REAL,                         -- MRP as supplied
    image_url            TEXT,
    source_product_id    TEXT,                         -- provenance
    data_source          TEXT,
    stock_is_simulated   INTEGER NOT NULL DEFAULT 0,   -- 1 = demo inventory, not live stock
    quality_flags_json   TEXT NOT NULL DEFAULT '[]',
    original_values_json TEXT NOT NULL DEFAULT '{}',
    embedding_text       TEXT,                         -- text used for the FAISS vector
    is_active            INTEGER NOT NULL DEFAULT 1    -- 0 = archived (kept only for order history)
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_brand    ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_price    ON products(price);

-- Keyword search index (BM25). Kept in sync by repository.upsert_products().
CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
    product_id UNINDEXED,
    name,
    brand,
    category,
    attributes_text,
    description,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- ---------------------------------------------------------------
-- Shoppers and orders
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id               TEXT PRIMARY KEY,        -- C-001
    name                      TEXT NOT NULL,
    city                      TEXT,
    preferred_language        TEXT NOT NULL DEFAULT 'en',
    budget_min                REAL,
    budget_max                REAL,
    preferred_brands_json     TEXT NOT NULL DEFAULT '[]',
    preferred_categories_json TEXT NOT NULL DEFAULT '[]',
    sizes_json                TEXT NOT NULL DEFAULT '{}',
    persona_note              TEXT,
    created_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,                      -- O-00001
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id  TEXT NOT NULL REFERENCES products(product_id),
    quantity    INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    price_paid  REAL,
    order_date  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);

-- ---------------------------------------------------------------
-- Conversation memory (one row per chat session, bound to ONE shopper)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    state_json  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------
-- Audit log (R10): one row per reply, replayable by trace_id
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    trace_id               TEXT PRIMARY KEY,
    session_id             TEXT NOT NULL,
    customer_id            TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    message                TEXT NOT NULL,
    language               TEXT,
    intent_json            TEXT NOT NULL DEFAULT '{}',
    retrieved_json         TEXT NOT NULL DEFAULT '[]',
    filters_json           TEXT NOT NULL DEFAULT '{}',
    tools_json             TEXT NOT NULL DEFAULT '[]',
    final_product_ids_json TEXT NOT NULL DEFAULT '[]',
    answer_type            TEXT NOT NULL,
    answer                 TEXT NOT NULL,
    follow_up_question     TEXT,
    response_json          TEXT NOT NULL DEFAULT '{}',
    latency_ms             REAL,
    llm_mode               TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
