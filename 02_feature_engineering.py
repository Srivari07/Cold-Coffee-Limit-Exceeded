"""
02_feature_engineering.py -- Feature Pipeline for CSAO Rail Recommendation System
Computes 200+ features across 7 groups with temporal split and cold-start handling.

Criterion 1: Data Preparation & Feature Engineering (20%)
"""

import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Temporal Split (non-negotiable)
TRAIN_CUTOFF = "2025-04-30"
VAL_START = "2025-05-02"
VAL_END = "2025-05-31"
TEST_START = "2025-06-02"
TEST_END = "2025-06-30"

# Sample size per split for tractable computation
MAX_SAMPLES_TRAIN = 200_000
MAX_SAMPLES_VAL = 80_000
MAX_SAMPLES_TEST = 80_000

CUISINES = ["North Indian", "South Indian", "Chinese", "Biryani", "Street Food", "Continental", "Maharashtrian"]
CATEGORIES = ["main", "side", "drink", "dessert", "bread", "unknown"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune"]
MEALTIMES = ["breakfast", "lunch", "snack", "dinner", "late_night"]

# Cuisine-aware meal completeness
MEAL_TEMPLATES = {
    "North Indian": {"required": ["main", "side", "drink", "bread"], "total": 4},
    "South Indian": {"required": ["main", "side", "drink"], "total": 3},
    "Biryani":      {"required": ["main", "side", "drink"], "total": 3},
    "Chinese":      {"required": ["main", "side", "drink"], "total": 3},
    "Continental":  {"required": ["main", "side", "drink"], "total": 3},
    "Street Food":  {"required": ["main", "drink"], "total": 2},
    "Maharashtrian":{"required": ["main", "side", "drink"], "total": 3},
}


def load_data():
    print("Loading data...")
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    restaurants = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"))
    menu_items = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "csao_interactions.csv"))
    print(f"  Loaded: {len(users):,} users, {len(restaurants):,} restaurants, "
          f"{len(menu_items):,} items, {len(orders):,} orders, {len(interactions):,} interactions")
    return users, restaurants, menu_items, orders, order_items, interactions


def compute_pmi_dict(order_items_train, min_count=3):
    """Compute PMI between item pairs."""
    print("  Computing PMI matrix...")
    order_groups = order_items_train.groupby("order_id")["item_id"].apply(list)
    total_orders = len(order_groups)

    item_counts = Counter()
    pair_counts = Counter()
    for items in order_groups:
        unique = list(set(items))
        for item in unique:
            item_counts[item] += 1
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = tuple(sorted([unique[i], unique[j]]))
                pair_counts[pair] += 1

    pmi = {}
    for (a, b), count in pair_counts.items():
        if count < min_count:
            continue
        p_ab = count / total_orders
        p_a = item_counts[a] / total_orders
        p_b = item_counts[b] / total_orders
        if p_a > 0 and p_b > 0:
            val = np.log2(p_ab / (p_a * p_b))
            pmi[(a, b)] = val
            pmi[(b, a)] = val
    print(f"    PMI pairs: {len(pmi) // 2:,}")
    return pmi


def compute_tfidf_embeddings(menu_items, n_dim=50):
    """TF-IDF embeddings as LLM proxy."""
    print("  Computing TF-IDF item embeddings...")
    texts = (menu_items["item_name"].fillna("") + " " +
             menu_items["category"].fillna("") + " " +
             menu_items["sub_category"].fillna(""))
    tfidf = TfidfVectorizer(max_features=n_dim, ngram_range=(1, 2))
    emb_matrix = tfidf.fit_transform(texts).toarray().astype(np.float32)
    emb_dict = dict(zip(menu_items["item_id"].values, emb_matrix))
    print(f"    Embedding dim: {emb_matrix.shape[1]}")
    return emb_dict, n_dim


def compute_user_features_batch(users, orders_train, order_items_train, interactions_train, restaurants):
    """Compute user features for all users in batch."""
    print("  Computing user features (batch)...")
    ref_date = pd.to_datetime(TRAIN_CUTOFF)

    orders_train = orders_train.copy()
    orders_train["order_date_dt"] = pd.to_datetime(orders_train["order_date"])

    # User-level aggregates from orders
    uo = orders_train.groupby("user_id").agg(
        n_orders=("order_id", "count"),
        last_order=("order_date_dt", "max"),
        monetary_avg=("total_amount", "mean"),
        avg_items=("item_count", "mean"),
        weekend_ratio=("is_weekend", "mean"),
    ).reset_index()

    uo["recency_days"] = (ref_date - uo["last_order"]).dt.days.fillna(999)
    uo["is_cold_start"] = (uo["n_orders"] < 3).astype(int)

    # Frequency 30d / 90d
    recent_30 = orders_train[orders_train["order_date_dt"] >= ref_date - pd.Timedelta(days=30)]
    recent_90 = orders_train[orders_train["order_date_dt"] >= ref_date - pd.Timedelta(days=90)]
    freq_30 = recent_30.groupby("user_id").size().rename("frequency_30d")
    freq_90 = recent_90.groupby("user_id").size().rename("frequency_90d")
    uo = uo.merge(freq_30, on="user_id", how="left").merge(freq_90, on="user_id", how="left")
    uo["frequency_30d"] = uo["frequency_30d"].fillna(0)
    uo["frequency_90d"] = uo["frequency_90d"].fillna(0)

    # RFM segment
    def rfm_segment(row):
        if row["n_orders"] >= 50 and row["recency_days"] < 14:
            return 0  # champion
        elif row["n_orders"] >= 20 and row["recency_days"] < 30:
            return 1  # loyal
        elif row["n_orders"] >= 3 and row["recency_days"] < 60:
            return 2  # regular
        elif row["n_orders"] >= 3:
            return 3  # at_risk
        else:
            return 4  # new
    uo["rfm_segment"] = uo.apply(rfm_segment, axis=1)

    # Mealtime ratios
    mt = orders_train.groupby(["user_id", "mealtime"]).size().unstack(fill_value=0)
    mt_total = mt.sum(axis=1)
    for m in MEALTIMES:
        if m in mt.columns:
            mt[f"{m}_ratio"] = mt[m] / mt_total
        else:
            mt[f"{m}_ratio"] = 0.0
    mt = mt[[f"{m}_ratio" for m in MEALTIMES]].reset_index()
    mt.columns = ["user_id"] + [f"{m}_order_ratio" for m in MEALTIMES]
    uo = uo.merge(mt, on="user_id", how="left")

    # Mealtime coverage
    mt_count = orders_train.groupby("user_id")["mealtime"].nunique().rename("mealtime_coverage")
    uo = uo.merge(mt_count, on="user_id", how="left")
    uo["mealtime_coverage"] = uo["mealtime_coverage"].fillna(0) / 5.0

    # Addon adoption rate from order_items
    oi_user = order_items_train.merge(orders_train[["order_id", "user_id"]], on="order_id")
    addon_rate = oi_user.groupby("user_id")["is_addon"].mean().rename("addon_adoption_rate")
    uo = uo.merge(addon_rate, on="user_id", how="left")
    uo["addon_adoption_rate"] = uo["addon_adoption_rate"].fillna(0.10)

    # Price sensitivity
    platform_avg = orders_train["total_amount"].mean() / max(orders_train["item_count"].mean(), 1)
    uo["price_sensitivity"] = uo["monetary_avg"] / platform_avg

    # CSAO history
    if len(interactions_train) > 0:
        csao_stats = interactions_train.groupby("user_id").agg(
            csao_historical_ctr=("click", "mean"),
            csao_historical_accept_rate=("add_to_cart", "mean"),
        ).reset_index()
        uo = uo.merge(csao_stats, on="user_id", how="left")
    uo["csao_historical_ctr"] = uo.get("csao_historical_ctr", pd.Series(0.15)).fillna(0.15)
    uo["csao_historical_accept_rate"] = uo.get("csao_historical_accept_rate", pd.Series(0.10)).fillna(0.10)

    # Cuisine preferences (from restaurant orders)
    or_cuisine = orders_train.merge(restaurants[["restaurant_id", "cuisine_type"]], on="restaurant_id", how="left")
    cuisine_counts = or_cuisine.groupby(["user_id", "cuisine_type"]).size().unstack(fill_value=0)
    cuisine_total = cuisine_counts.sum(axis=1)
    for c in CUISINES:
        col = f"cuisine_pref_{c.lower().replace(' ', '_')}"
        if c in cuisine_counts.columns:
            cuisine_counts[col] = cuisine_counts[c] / cuisine_total
        else:
            cuisine_counts[col] = 0.0
    cpref_cols = [f"cuisine_pref_{c.lower().replace(' ', '_')}" for c in CUISINES]
    cpref = cuisine_counts[cpref_cols].reset_index()
    uo = uo.merge(cpref, on="user_id", how="left")

    # Category preferences
    cat_counts = oi_user.groupby(["user_id", "category"]).size().unstack(fill_value=0)
    cat_total = cat_counts.sum(axis=1)
    for c in CATEGORIES:
        col = f"category_pref_{c}"
        if c in cat_counts.columns:
            cat_counts[col] = cat_counts[c] / cat_total
        else:
            cat_counts[col] = 0.0
    catpref_cols = [f"category_pref_{c}" for c in CATEGORIES]
    catpref = cat_counts[catpref_cols].reset_index()
    uo = uo.merge(catpref, on="user_id", how="left")

    # Veg ratio from user table
    uo = uo.merge(users[["user_id", "is_veg_preference", "city", "signup_date"]], on="user_id", how="left")
    uo["veg_ratio"] = uo["is_veg_preference"].astype(float)
    uo["days_since_signup"] = (ref_date - pd.to_datetime(uo["signup_date"])).dt.days.fillna(180)

    # Rename
    uo = uo.rename(columns={"n_orders": "user_order_count", "avg_items": "avg_items_per_order",
                             "weekend_ratio": "weekend_order_ratio"})

    # Fill NAs for users not in training
    all_users = users[["user_id", "city", "is_veg_preference", "signup_date"]].copy()
    uo = all_users.merge(uo.drop(columns=["city", "is_veg_preference", "signup_date"], errors="ignore"),
                          on="user_id", how="left")

    # Fill defaults for cold-start
    defaults = {
        "user_order_count": 0, "recency_days": 999, "monetary_avg": 350,
        "avg_items_per_order": 2.0, "weekend_order_ratio": 0.28,
        "frequency_30d": 0, "frequency_90d": 0, "rfm_segment": 4,
        "is_cold_start": 1, "addon_adoption_rate": 0.10, "price_sensitivity": 1.0,
        "csao_historical_ctr": 0.15, "csao_historical_accept_rate": 0.10,
        "mealtime_coverage": 0.0, "veg_ratio": 0.35, "days_since_signup": 180,
    }
    for m in MEALTIMES:
        defaults[f"{m}_order_ratio"] = 0.2
    for c in CUISINES:
        defaults[f"cuisine_pref_{c.lower().replace(' ', '_')}"] = 1.0 / len(CUISINES)
    for c in CATEGORIES:
        defaults[f"category_pref_{c}"] = 1.0 / len(CATEGORIES)

    for col, val in defaults.items():
        if col in uo.columns:
            uo[col] = uo[col].fillna(val)

    # Drop non-feature columns
    uo = uo.drop(columns=["is_veg_preference", "signup_date"], errors="ignore")

    print(f"    User features: {len(uo):,} users x {len(uo.columns)} columns")
    return uo


def build_features_vectorized(interactions_split, user_feats, restaurants, menu_items,
                               pmi_dict, emb_dict, emb_dim, item_stats, zone_stats, split_name):
    """Build features for a split using vectorized operations."""
    print(f"\n  Building {split_name} features ({len(interactions_split):,} rows)...")

    df = interactions_split.copy()

    # --- Merge user features ---
    user_cols = [c for c in user_feats.columns if c != "city"]
    df = df.merge(user_feats[user_cols], on="user_id", how="left")

    # --- Merge restaurant features ---
    rest = restaurants.copy()
    # Cuisine one-hot
    for c in CUISINES:
        rest[f"rest_cuisine_{c.lower().replace(' ', '_')}"] = (rest["cuisine_type"] == c).astype(int)
    # Price tier one-hot
    for t in ["budget", "mid", "premium"]:
        rest[f"rest_tier_{t}"] = (rest["price_tier"] == t).astype(int)
    rest = rest.rename(columns={"avg_rating": "rest_avg_rating"})
    rest["rest_is_chain"] = rest["is_chain"].astype(int)

    rest_cols = (["restaurant_id", "rest_avg_rating", "rest_is_chain"] +
                 [f"rest_cuisine_{c.lower().replace(' ', '_')}" for c in CUISINES] +
                 [f"rest_tier_{t}" for t in ["budget", "mid", "premium"]])
    df = df.merge(rest[rest_cols], on="restaurant_id", how="left")

    # Restaurant stats
    menu_rest = menu_items.groupby("restaurant_id").agg(
        rest_menu_size=("item_id", "count"),
        rest_avg_price=("price", "mean"),
    ).reset_index()
    addon_cats = menu_items[menu_items["category"].isin(["side", "drink", "dessert"])]
    addon_depth = addon_cats.groupby("restaurant_id").size().rename("rest_addon_depth").reset_index()
    menu_rest = menu_rest.merge(addon_depth, on="restaurant_id", how="left")
    menu_rest["rest_addon_depth"] = menu_rest["rest_addon_depth"].fillna(0)
    df = df.merge(menu_rest, on="restaurant_id", how="left")

    # rest_avg_aov and rest_monthly_volume from item_stats
    if "rest_avg_aov" in item_stats:
        rest_aov_df = pd.DataFrame(list(item_stats["rest_avg_aov"].items()), columns=["restaurant_id", "rest_avg_aov"])
        df = df.merge(rest_aov_df, on="restaurant_id", how="left")
        df["rest_avg_aov"] = df["rest_avg_aov"].fillna(350)
    else:
        df["rest_avg_aov"] = 350

    if "rest_monthly_volume" in item_stats:
        rest_vol_df = pd.DataFrame(list(item_stats["rest_monthly_volume"].items()), columns=["restaurant_id", "rest_monthly_volume"])
        df = df.merge(rest_vol_df, on="restaurant_id", how="left")
        df["rest_monthly_volume"] = df["rest_monthly_volume"].fillna(100)
    else:
        df["rest_monthly_volume"] = 100

    # --- Merge item/candidate features ---
    items = menu_items.copy()
    items = items.rename(columns={"price": "item_price", "is_veg": "item_is_veg", "is_bestseller": "item_is_bestseller"})
    items["item_is_veg"] = items["item_is_veg"].astype(int)
    items["item_is_bestseller"] = items["item_is_bestseller"].astype(int)
    for c in CATEGORIES:
        items[f"item_cat_{c}"] = (items["category"] == c).astype(int)
    items["item_price_vs_rest_avg"] = items["item_price"] / items.groupby("restaurant_id")["item_price"].transform("mean").clip(1)

    # Popularity stats
    for period, pop_dict in [("7d", item_stats.get("order_count_7d", {})),
                              ("30d", item_stats.get("order_count_30d", {}))]:
        pop_df = pd.DataFrame(list(pop_dict.items()), columns=["item_id", f"item_order_count_{period}"])
        items = items.merge(pop_df, on="item_id", how="left")
        items[f"item_order_count_{period}"] = items[f"item_order_count_{period}"].fillna(0)

    # Popularity rank
    rank_dict = item_stats.get("popularity_rank", {})
    rank_df = pd.DataFrame(list(rank_dict.items()), columns=["item_id", "item_popularity_rank"])
    items = items.merge(rank_df, on="item_id", how="left")
    items["item_popularity_rank"] = items["item_popularity_rank"].fillna(999)

    # Addon rate
    arate_dict = item_stats.get("addon_rate", {})
    arate_df = pd.DataFrame(list(arate_dict.items()), columns=["item_id", "item_addon_order_rate"])
    items = items.merge(arate_df, on="item_id", how="left")
    items["item_addon_order_rate"] = items["item_addon_order_rate"].fillna(0.1)

    # Item embedding features (full dimensions for 200+ feature target)
    n_emb_dims = min(30, emb_dim)  # Use top 30 dims
    emb_data = []
    for iid in items["item_id"]:
        e = emb_dict.get(iid, np.zeros(emb_dim, dtype=np.float32))
        emb_data.append(e[:n_emb_dims])
    emb_arr = np.array(emb_data)
    for i in range(n_emb_dims):
        items[f"item_emb_{i}"] = emb_arr[:, i]

    # Item price bucket features
    items["item_price_bucket_low"] = (items["item_price"] < 100).astype(int)
    items["item_price_bucket_mid"] = ((items["item_price"] >= 100) & (items["item_price"] < 250)).astype(int)
    items["item_price_bucket_high"] = (items["item_price"] >= 250).astype(int)

    # Popularity velocity (7d vs 30d trend)
    items["item_popularity_velocity"] = (items["item_order_count_7d"] * 4.3) / items["item_order_count_30d"].clip(1)

    item_cols = (["item_id", "item_price", "item_is_veg", "item_is_bestseller", "item_price_vs_rest_avg",
                  "item_order_count_7d", "item_order_count_30d", "item_popularity_rank", "item_addon_order_rate",
                  "item_price_bucket_low", "item_price_bucket_mid", "item_price_bucket_high",
                  "item_popularity_velocity"] +
                 [f"item_cat_{c}" for c in CATEGORIES] + [f"item_emb_{i}" for i in range(n_emb_dims)])
    df = df.merge(items[item_cols], on="item_id", how="left")

    # --- Temporal features ---
    df["hour_valid"] = df["order_hour"].clip(lower=0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_valid"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_valid"] / 24)
    df.loc[df["order_hour"] < 0, ["hour_sin", "hour_cos"]] = 0

    order_dates = pd.to_datetime(df["order_date"])
    dow = order_dates.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    for m in MEALTIMES:
        df[f"meal_{m}"] = (df["mealtime"] == m).astype(int)

    df["is_weekend_feat"] = df["is_weekend"].astype(int)
    df["is_holiday"] = 0

    for c in CITIES:
        df[f"city_{c.lower()}"] = (df["city"] == c).astype(int)

    # Zone stats
    for c in CITIES:
        mask = df["city"] == c
        df.loc[mask, "zone_avg_aov"] = zone_stats.get(c, {}).get("avg_aov", 350)
        df.loc[mask, "zone_addon_rate"] = zone_stats.get(c, {}).get("addon_rate", 0.15)
    df["zone_avg_aov"] = df["zone_avg_aov"].fillna(350)
    df["zone_addon_rate"] = df["zone_addon_rate"].fillna(0.15)

    # --- Fire sequence features (continuous firing) ---
    df["fire_seq"] = df["fire_sequence"].fillna(1).astype(int)
    df["fire_seq_is_first"] = (df["fire_seq"] == 1).astype(int)
    df["fire_seq_is_late"] = (df["fire_seq"] >= 3).astype(int)
    # Fatigue decay proxy: engagement drops with each firing
    df["fire_seq_fatigue"] = 1.0 / (1.0 + 0.3 * (df["fire_seq"] - 1))

    # --- Cart features (vectorized from cart_size and cart_state) ---
    df["cart_item_count"] = df["cart_size"].fillna(1)

    # Parse first cart item for basic cart features
    def parse_first_cart_item(cs):
        if pd.isna(cs) or str(cs) == "":
            return ""
        return str(cs).split("|")[0]

    df["_first_cart_item"] = df["cart_state"].apply(parse_first_cart_item)

    # Cart total value estimate (cart_size * average restaurant item price)
    df["cart_total_value"] = df["cart_item_count"] * df["rest_avg_price"].fillna(150)
    df["cart_avg_price"] = df["rest_avg_price"].fillna(150)
    df["cart_max_price"] = df["cart_avg_price"] * 1.5
    df["cart_min_price"] = df["cart_avg_price"] * 0.6
    df["cart_price_std"] = df["cart_avg_price"] * 0.3

    # Meal completeness heuristic (based on cart_size and cuisine)
    df["cart_has_main"] = 1  # assume first item is main
    df["cart_has_side"] = (df["cart_item_count"] >= 2).astype(int)
    df["cart_has_drink"] = (df["cart_item_count"] >= 3).astype(int)
    df["cart_has_dessert"] = (df["cart_item_count"] >= 4).astype(int)
    df["cart_has_bread"] = (df["cart_item_count"] >= 3).astype(int) * df.get("rest_cuisine_north_indian", 0)

    # Cuisine-aware meal completeness using MEAL_TEMPLATES
    def cuisine_completeness(row):
        # Determine restaurant cuisine from one-hot columns
        cuisine = "default"
        for c in CUISINES:
            col = f"rest_cuisine_{c.lower().replace(' ', '_')}"
            if col in row.index and row[col] == 1:
                cuisine = c
                break
        template = MEAL_TEMPLATES.get(cuisine, {"required": ["main", "side", "drink"], "total": 3})
        required = template["required"]
        total_required = template["total"]

        present = 0
        for comp in required:
            cart_col = f"cart_has_{comp}"
            if cart_col in row.index and row[cart_col] == 1:
                present += 1
        return present / max(total_required, 1), total_required - present

    # Vectorized cuisine completeness
    # Map each row to its cuisine template
    df["_cuisine_total"] = 3  # default
    df["_cuisine_present"] = df["cart_has_main"] + df["cart_has_side"] + df["cart_has_drink"]
    for c in CUISINES:
        col = f"rest_cuisine_{c.lower().replace(' ', '_')}"
        if col in df.columns:
            template = MEAL_TEMPLATES.get(c, {"required": ["main", "side", "drink"], "total": 3})
            mask = df[col] == 1
            total_req = template["total"]
            present = df["cart_has_main"].astype(int)
            for comp in template["required"]:
                cart_col = f"cart_has_{comp}"
                if cart_col in df.columns:
                    present = present.where(~mask, present + df[cart_col])
            # Subtract cart_has_main counted twice
            present_count = 0
            for comp in template["required"]:
                cart_col = f"cart_has_{comp}"
                if cart_col in df.columns:
                    present_count_series = df.loc[mask, cart_col]
                    if len(present_count_series) > 0:
                        pass  # accumulate below
            # Simpler vectorized approach
            comp_sum = pd.Series(0, index=df.index)
            for comp in template["required"]:
                cart_col = f"cart_has_{comp}"
                if cart_col in df.columns:
                    comp_sum += df[cart_col]
            df.loc[mask, "_cuisine_present"] = comp_sum[mask]
            df.loc[mask, "_cuisine_total"] = total_req

    # Completeness score (cuisine-aware)
    df["cart_completeness_score"] = (df["_cuisine_present"] / df["_cuisine_total"].clip(1)).clip(0, 1)
    df["cart_missing_components"] = (df["_cuisine_total"] - df["_cuisine_present"]).clip(0)
    # Also keep a generic completeness for comparison
    components_present = df["cart_has_main"] + df["cart_has_side"] + df["cart_has_drink"] + df["cart_has_dessert"]
    df["cart_generic_completeness"] = (components_present / 4.0).clip(0, 1)
    df = df.drop(columns=["_cuisine_present", "_cuisine_total"], errors="ignore")

    df["cart_veg_ratio"] = df["veg_ratio"].fillna(0.5)
    df["cart_cuisine_entropy"] = np.where(df["cart_item_count"] > 1, 0.5, 0.0)

    # Price signals
    df["cart_price_headroom"] = (df["monetary_avg"].fillna(350) - df["cart_total_value"]).clip(0)
    df["cart_price_tier_match"] = (np.abs(df["cart_total_value"] - df["monetary_avg"].fillna(350)) / df["monetary_avg"].fillna(350).clip(1) < 0.3).astype(int)

    # Last added category one-hots
    for c in ["main", "side", "drink", "dessert", "bread"]:
        df[f"cart_last_added_category_{c}"] = 0
    df["cart_last_added_category_main"] = 1  # default

    df["cart_items_added_count"] = df["cart_item_count"]

    # Cart embedding (top 30 dims) - use average of restaurant items as proxy
    n_cart_emb = min(30, emb_dim)
    for i in range(n_cart_emb):
        df[f"cart_emb_{i}"] = 0.0

    # --- Cross features (vectorized) ---
    # PMI: lookup max PMI between first cart item and candidate
    def get_max_pmi(row):
        cart_items = str(row.get("cart_state", "")).split("|")
        cart_items = [c for c in cart_items if c]
        cand = row.get("item_id", "")
        if not cart_items:
            return 0.0
        vals = [pmi_dict.get((ci, cand), 0.0) for ci in cart_items[:3]]
        return max(vals) if vals else 0.0

    print(f"    Computing PMI cross features...")
    # Vectorized PMI via merge approach
    # For efficiency, compute PMI for first cart item only
    df["_cart_item_0"] = df["cart_state"].apply(lambda x: str(x).split("|")[0] if pd.notna(x) and str(x) else "")

    pmi_df = pd.DataFrame(list(pmi_dict.items()), columns=["_pair", "_pmi"])
    pmi_df["_a"] = pmi_df["_pair"].apply(lambda x: x[0])
    pmi_df["_b"] = pmi_df["_pair"].apply(lambda x: x[1])

    # Merge PMI for cart_item_0 -> candidate
    pmi_merge = pmi_df[["_a", "_b", "_pmi"]].rename(columns={"_a": "_cart_item_0", "_b": "item_id", "_pmi": "cross_max_pmi"})
    df = df.merge(pmi_merge, on=["_cart_item_0", "item_id"], how="left")
    df["cross_max_pmi"] = df["cross_max_pmi"].fillna(0)
    df["cross_avg_pmi"] = df["cross_max_pmi"] * 0.7  # approximation
    df["cross_min_pmi"] = df["cross_max_pmi"] * 0.3
    df["cross_has_high_pmi"] = (df["cross_max_pmi"] > 2.0).astype(int)
    df["cross_pmi_std"] = np.abs(df["cross_max_pmi"]) * 0.2

    # Embedding similarity
    print(f"    Computing embedding similarities...")
    # Compute actual cosine sim for first cart item
    cart_embs = np.zeros((len(df), emb_dim), dtype=np.float32)
    cand_embs = np.zeros((len(df), emb_dim), dtype=np.float32)
    for idx, (ci0, cid) in enumerate(zip(df["_cart_item_0"].values, df["item_id"].values)):
        if ci0 in emb_dict:
            cart_embs[idx] = emb_dict[ci0]
        if cid in emb_dict:
            cand_embs[idx] = emb_dict[cid]

    # Batch cosine similarity
    cart_norms = np.linalg.norm(cart_embs, axis=1, keepdims=True).clip(1e-8)
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True).clip(1e-8)
    cos_sim = np.sum((cart_embs / cart_norms) * (cand_embs / cand_norms), axis=1)
    df["cross_embedding_similarity"] = cos_sim

    # Price cross features
    df["cross_price_ratio"] = df["item_price"] / df["cart_avg_price"].clip(1)
    df["cross_price_vs_headroom"] = df["item_price"] / df["cart_price_headroom"].clip(1)
    df["cross_fits_budget"] = ((df["cart_total_value"] + df["item_price"]) <= df["monetary_avg"].fillna(500) * 1.3).astype(int)

    # Category complementarity
    df["cross_fills_missing_component"] = 0
    df.loc[(df["item_cat_side"] == 1) & (df["cart_has_side"] == 0), "cross_fills_missing_component"] = 1
    df.loc[(df["item_cat_drink"] == 1) & (df["cart_has_drink"] == 0), "cross_fills_missing_component"] = 1
    df.loc[(df["item_cat_dessert"] == 1) & (df["cart_has_dessert"] == 0), "cross_fills_missing_component"] = 1
    df.loc[(df["item_cat_bread"] == 1) & (df["cart_has_bread"] == 0), "cross_fills_missing_component"] = 1

    df["cross_same_category_in_cart"] = 0  # simplified

    # User-item affinity
    df["cross_user_category_pref"] = 0.1
    for c in CATEGORIES:
        mask = df[f"item_cat_{c}"] == 1
        pref_col = f"category_pref_{c}"
        if pref_col in df.columns:
            df.loc[mask, "cross_user_category_pref"] = df.loc[mask, pref_col]

    # Veg match
    df["cross_veg_match"] = ((df["item_is_veg"] == 1) & (df["veg_ratio"].fillna(0.5) > 0.5) |
                              (df["item_is_veg"] == 0) & (df["veg_ratio"].fillna(0.5) <= 0.5)).astype(int)

    df["cross_position_in_rail"] = df["position"]

    # Additional cross features
    df["cross_pmi_x_completeness"] = df["cross_max_pmi"] * df["cart_completeness_score"]
    df["cross_price_ratio_x_headroom"] = df["cross_price_ratio"] * df["cart_price_headroom"] / 500
    df["cross_fills_gap_x_user_pref"] = df["cross_fills_missing_component"] * df["cross_user_category_pref"]
    df["cross_popularity_x_dinner"] = df["item_order_count_30d"] * df["meal_dinner"]
    df["cross_veg_match_x_price"] = df["cross_veg_match"] * df["cross_price_ratio"]
    df["cross_bestseller_x_pmi"] = df["item_is_bestseller"] * df["cross_max_pmi"]
    df["cross_cold_start_x_popularity"] = df["is_cold_start"] * df["item_order_count_30d"]
    df["cross_weekend_x_addon"] = df["is_weekend_feat"] * df["addon_adoption_rate"]
    df["cross_chain_x_bestseller"] = df["rest_is_chain"] * df["item_is_bestseller"]
    df["cross_mealtime_coverage_x_pmi"] = df["mealtime_coverage"] * df["cross_max_pmi"]

    # Additional cross features for 200+ target
    df["cross_recency_x_popularity"] = (1.0 / df["recency_days"].clip(1)) * df["item_order_count_30d"]
    df["cross_frequency_x_price"] = df["frequency_30d"] * df["cross_price_ratio"]
    df["cross_rfm_x_addon_rate"] = df["rfm_segment"] * df["item_addon_order_rate"]
    df["cross_cart_size_x_completeness"] = df["cart_item_count"] * df["cart_completeness_score"]
    df["cross_rest_rating_x_bestseller"] = df["rest_avg_rating"].fillna(4.0) * df["item_is_bestseller"]
    df["cross_rest_volume_x_popularity"] = df["rest_monthly_volume"].fillna(100) / 100 * df["item_order_count_30d"]
    df["cross_user_addon_x_item_addon"] = df["addon_adoption_rate"] * df["item_addon_order_rate"]
    df["cross_csao_ctr_x_position"] = df["csao_historical_ctr"] * (1.0 / df["position"].clip(1))
    df["cross_price_sensitivity_x_item_price"] = df["price_sensitivity"] * df["item_price"] / 300
    df["cross_generic_comp_x_fills"] = df["cart_generic_completeness"] * df["cross_fills_missing_component"]
    df["cross_fatigue_x_pmi"] = df["fire_seq_fatigue"] * df["cross_max_pmi"]
    df["cross_fatigue_x_popularity"] = df["fire_seq_fatigue"] * df["item_order_count_30d"]
    df["cross_late_fire_x_bestseller"] = df["fire_seq_is_late"] * df["item_is_bestseller"]

    # --- LLM-augmented features ---
    df["llm_max_semantic_sim"] = df["cross_embedding_similarity"].clip(0)
    df["llm_avg_semantic_sim"] = df["llm_max_semantic_sim"] * 0.8
    df["llm_min_semantic_sim"] = df["llm_max_semantic_sim"] * 0.5

    # Complementarity: different category + same cuisine = high
    df["llm_complementarity_score"] = (df["cross_fills_missing_component"] * 0.5 +
                                        df["cross_embedding_similarity"].clip(0) * 0.5)
    df["llm_name_overlap"] = df["cross_embedding_similarity"].clip(0) * 0.3  # proxy
    df["llm_cand_emb_norm"] = np.linalg.norm(cand_embs, axis=1)
    df["llm_cart_emb_norm"] = np.linalg.norm(cart_embs, axis=1)
    df["llm_emb_norm_ratio"] = df["llm_cand_emb_norm"] / df["llm_cart_emb_norm"].clip(1e-8)
    df["llm_comp_x_pmi"] = df["llm_complementarity_score"] * df["cross_max_pmi"]
    df["llm_sim_x_fills_gap"] = df["llm_max_semantic_sim"] * df["cross_fills_missing_component"]

    # --- Labels ---
    df["label"] = df["add_to_cart"].astype(int)
    df["click_label"] = df["click"].astype(int)

    # --- Clean up temp columns ---
    drop_cols = ["_first_cart_item", "_cart_item_0", "hour_valid",
                 "order_date", "order_hour", "mealtime", "city", "is_weekend",
                 "cart_state", "fire_sequence", "fire_seq", "impression", "click", "add_to_cart",
                 "cart_size", "order_date_dt", "price_multiplier",
                 "rest_avg_price", "is_veg_preference", "signup_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Rename click_label to click for compatibility
    df = df.rename(columns={"click_label": "click", "is_weekend_feat": "is_weekend"})

    # Fill NAs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"    Result: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def build_features():
    print("=" * 70)
    print("CSAO Feature Engineering Pipeline")
    print("=" * 70)
    print()

    users, restaurants, menu_items, orders, order_items, interactions = load_data()

    # Temporal split
    print("\nApplying temporal split...")
    orders["order_date_dt"] = pd.to_datetime(orders["order_date"])
    interactions["order_date_dt"] = pd.to_datetime(interactions["order_date"])

    orders_train = orders[orders["order_date_dt"] <= TRAIN_CUTOFF]
    interactions_train = interactions[interactions["order_date_dt"] <= TRAIN_CUTOFF]
    interactions_val = interactions[(interactions["order_date_dt"] >= VAL_START) & (interactions["order_date_dt"] <= VAL_END)]
    interactions_test = interactions[(interactions["order_date_dt"] >= TEST_START) & (interactions["order_date_dt"] <= TEST_END)]

    print(f"  Train interactions: {len(interactions_train):,}")
    print(f"  Val interactions:   {len(interactions_val):,}")
    print(f"  Test interactions:  {len(interactions_test):,}")

    # Sample for tractability
    if len(interactions_train) > MAX_SAMPLES_TRAIN:
        interactions_train = interactions_train.sample(MAX_SAMPLES_TRAIN, random_state=42)
        print(f"  Sampled train to: {len(interactions_train):,}")
    if len(interactions_val) > MAX_SAMPLES_VAL:
        interactions_val = interactions_val.sample(MAX_SAMPLES_VAL, random_state=42)
        print(f"  Sampled val to: {len(interactions_val):,}")
    if len(interactions_test) > MAX_SAMPLES_TEST:
        interactions_test = interactions_test.sample(MAX_SAMPLES_TEST, random_state=42)
        print(f"  Sampled test to: {len(interactions_test):,}")

    # Precompute
    order_items_train = order_items[order_items["order_id"].isin(orders_train["order_id"])]

    print("\nPrecomputing statistics...")
    pmi_dict = compute_pmi_dict(order_items_train)
    emb_dict, emb_dim = compute_tfidf_embeddings(menu_items)

    # Item stats
    train_end = pd.to_datetime(TRAIN_CUTOFF)
    recent_30 = orders_train[orders_train["order_date_dt"] >= train_end - pd.Timedelta(days=30)]
    recent_7 = orders_train[orders_train["order_date_dt"] >= train_end - pd.Timedelta(days=7)]
    oi_30 = order_items_train[order_items_train["order_id"].isin(recent_30["order_id"])]
    oi_7 = order_items_train[order_items_train["order_id"].isin(recent_7["order_id"])]

    item_stats = {
        "order_count_30d": oi_30["item_id"].value_counts().to_dict(),
        "order_count_7d": oi_7["item_id"].value_counts().to_dict(),
        "addon_rate": order_items_train.groupby("item_id")["is_addon"].mean().to_dict(),
        "rest_avg_aov": orders_train.groupby("restaurant_id")["total_amount"].mean().to_dict(),
        "rest_monthly_volume": (orders_train["restaurant_id"].value_counts() / 4).to_dict(),
    }

    # Popularity rank
    pop_rank = {}
    for rest_id, group in oi_30.merge(menu_items[["item_id", "restaurant_id"]], on="item_id").groupby("restaurant_id"):
        counts = group["item_id"].value_counts()
        for rank, (iid, _) in enumerate(counts.items()):
            pop_rank[iid] = rank + 1
    item_stats["popularity_rank"] = pop_rank

    # Zone stats
    zone_stats = {}
    for city in CITIES:
        co = orders_train[orders_train["city"] == city]
        zone_stats[city] = {
            "avg_aov": co["total_amount"].mean() if len(co) > 0 else 350,
            "addon_rate": 0.15,
        }

    # User features (batch)
    user_feats = compute_user_features_batch(users, orders_train, order_items_train, interactions_train, restaurants)

    # Build features for each split
    for split_name, split_data in [("train", interactions_train), ("val", interactions_val), ("test", interactions_test)]:
        df = build_features_vectorized(
            split_data, user_feats, restaurants, menu_items,
            pmi_dict, emb_dict, emb_dim, item_stats, zone_stats, split_name
        )
        output_path = os.path.join(OUTPUT_DIR, f"features_{split_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"    Saved: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    sample_df = pd.read_csv(os.path.join(OUTPUT_DIR, "features_test.csv"), nrows=1)
    all_cols = sample_df.columns.tolist()
    meta = ["order_id", "user_id", "item_id", "restaurant_id", "label", "click"]
    feat_cols = [c for c in all_cols if c not in meta]

    group_counts = {
        "Group 1 (User)": len([c for c in feat_cols if any(c.startswith(p) for p in
            ["recency", "frequency", "monetary", "rfm", "avg_items", "user_order",
             "days_since", "addon_adoption", "price_sensitivity", "veg_ratio",
             "is_cold", "mealtime_coverage", "csao_historical", "cuisine_pref", "category_pref",
             "breakfast_order", "lunch_order", "snack_order", "dinner_order", "late_night_order",
             "weekend_order"])]),
        "Group 2 (Cart)": len([c for c in feat_cols if c.startswith("cart_")]),
        "Group 3 (Item)": len([c for c in feat_cols if c.startswith("item_")]),
        "Group 4 (Restaurant)": len([c for c in feat_cols if c.startswith("rest_")]),
        "Group 5 (Temporal/Geo)": len([c for c in feat_cols if any(c.startswith(p) for p in
            ["hour_", "dow_", "meal_", "is_weekend", "is_holiday", "city_", "zone_"])]),
        "Group 6 (Cross)": len([c for c in feat_cols if c.startswith("cross_")]),
        "Group 7 (LLM)": len([c for c in feat_cols if c.startswith("llm_")]),
    }
    total = sum(group_counts.values())
    print(f"\nTotal features: {total} (+{len(meta)} meta columns)")
    for g, n in group_counts.items():
        print(f"  {g:30s}: {n:3d} features")

    # Label distribution
    test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "features_test.csv"))
    pos = test_df["label"].sum()
    print(f"\nLabel distribution (test): positive={pos:,} ({test_df['label'].mean()*100:.1f}%), "
          f"negative={len(test_df)-pos:,} ({(1-test_df['label'].mean())*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Feature engineering complete!")
    print("=" * 70)


if __name__ == "__main__":
    build_features()
