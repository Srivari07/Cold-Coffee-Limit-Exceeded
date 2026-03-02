"""
Feature pre-computation and real-time feature engineering for CSAO backend.

At startup: caches per-entity feature vectors from features_train.csv.
At serving time: builds the full feature vector for each (user, item, context) triple.
"""

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("csao-backend")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Meal completeness templates (same as 02_feature_engineering.py)
MEAL_TEMPLATES: dict[str, dict[str, Any]] = {
    "North Indian": {"required": ["main", "side", "drink", "bread"], "total": 4},
    "South Indian": {"required": ["main", "side", "drink"], "total": 3},
    "Biryani": {"required": ["main", "side", "drink"], "total": 3},
    "Chinese": {"required": ["main", "side", "drink"], "total": 3},
    "Continental": {"required": ["main", "side", "drink"], "total": 3},
    "Street Food": {"required": ["main", "drink"], "total": 2},
    "Maharashtrian": {"required": ["main", "side", "drink"], "total": 3},
}

# Cold-start user defaults (from 02_feature_engineering.py lines 233-246)
COLD_START_USER: dict[str, float] = {
    "user_order_count": 0, "last_order": 0, "monetary_avg": 350,
    "avg_items_per_order": 2.0, "weekend_order_ratio": 0.28,
    "recency_days": 999, "is_cold_start": 1, "frequency_30d": 0,
    "frequency_90d": 0, "rfm_segment": 4, "breakfast_order_ratio": 0.2,
    "lunch_order_ratio": 0.2, "snack_order_ratio": 0.2,
    "dinner_order_ratio": 0.2, "late_night_order_ratio": 0.2,
    "mealtime_coverage": 0.0, "addon_adoption_rate": 0.10,
    "price_sensitivity": 1.0, "csao_historical_ctr": 0.15,
    "csao_historical_accept_rate": 0.10,
    "cuisine_pref_north_indian": 1 / 7, "cuisine_pref_south_indian": 1 / 7,
    "cuisine_pref_chinese": 1 / 7, "cuisine_pref_biryani": 1 / 7,
    "cuisine_pref_street_food": 1 / 7, "cuisine_pref_continental": 1 / 7,
    "cuisine_pref_maharashtrian": 1 / 7,
    "category_pref_main": 1 / 6, "category_pref_side": 1 / 6,
    "category_pref_drink": 1 / 6, "category_pref_dessert": 1 / 6,
    "category_pref_bread": 1 / 6, "category_pref_unknown": 1 / 6,
    "veg_ratio": 0.35, "days_since_signup": 180,
}

# Column groups used for pre-computing per-entity features
USER_COLS = [
    "user_order_count", "last_order", "monetary_avg", "avg_items_per_order",
    "weekend_order_ratio", "recency_days", "is_cold_start", "frequency_30d",
    "frequency_90d", "rfm_segment", "breakfast_order_ratio", "lunch_order_ratio",
    "snack_order_ratio", "dinner_order_ratio", "late_night_order_ratio",
    "mealtime_coverage", "addon_adoption_rate", "price_sensitivity",
    "csao_historical_ctr", "csao_historical_accept_rate",
    "cuisine_pref_north_indian", "cuisine_pref_south_indian",
    "cuisine_pref_chinese", "cuisine_pref_biryani",
    "cuisine_pref_street_food", "cuisine_pref_continental",
    "cuisine_pref_maharashtrian",
    "category_pref_main", "category_pref_side", "category_pref_drink",
    "category_pref_dessert", "category_pref_bread", "category_pref_unknown",
    "veg_ratio", "days_since_signup",
]

RESTAURANT_COLS = [
    "rest_avg_rating", "rest_is_chain",
    "rest_cuisine_north_indian", "rest_cuisine_south_indian",
    "rest_cuisine_chinese", "rest_cuisine_biryani",
    "rest_cuisine_street_food", "rest_cuisine_continental",
    "rest_cuisine_maharashtrian",
    "rest_tier_budget", "rest_tier_mid", "rest_tier_premium",
    "rest_menu_size", "rest_addon_depth", "rest_avg_aov", "rest_monthly_volume",
]

ITEM_COLS = [
    "item_price", "item_is_veg", "item_is_bestseller",
    "item_price_vs_rest_avg", "item_order_count_7d", "item_order_count_30d",
    "item_popularity_rank", "item_addon_order_rate",
    "item_price_bucket_low", "item_price_bucket_mid", "item_price_bucket_high",
    "item_popularity_velocity",
    "item_cat_main", "item_cat_side", "item_cat_drink",
    "item_cat_dessert", "item_cat_bread", "item_cat_unknown",
] + [f"item_emb_{i}" for i in range(30)]

ZONE_STATS_DEFAULT = {"zone_avg_aov": 350.0, "zone_addon_rate": 0.15}


class FeatureStore:
    """Pre-computed feature caches loaded at startup."""

    def __init__(self) -> None:
        self.user_features: dict[str, dict[str, float]] = {}
        self.item_features: dict[str, dict[str, float]] = {}
        self.restaurant_features: dict[str, dict[str, float]] = {}
        self.zone_stats: dict[str, dict[str, float]] = {}
        self.item_embeddings: dict[str, np.ndarray] = {}  # item_id -> 30-dim TF-IDF

    def load(self) -> None:
        """Load and cache per-entity features from features_train.csv."""
        start = time.time()
        features_path = OUTPUT_DIR / "features_train.csv"

        if not features_path.exists():
            log.warning("features_train.csv not found — feature store will be empty")
            return

        log.info("Loading features_train.csv for pre-computation...")
        df = pd.read_csv(features_path)
        log.info(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

        # --- User features: group by user_id, take first (same user = same features) ---
        avail_user_cols = [c for c in USER_COLS if c in df.columns]
        if avail_user_cols:
            user_df = df.groupby("user_id")[avail_user_cols].first()
            for uid, row in user_df.iterrows():
                self.user_features[uid] = row.fillna(0).to_dict()
            log.info(f"  Cached {len(self.user_features)} user feature vectors")

        # --- Restaurant features: group by restaurant_id ---
        avail_rest_cols = [c for c in RESTAURANT_COLS if c in df.columns]
        if avail_rest_cols:
            rest_df = df.groupby("restaurant_id")[avail_rest_cols].first()
            for rid, row in rest_df.iterrows():
                self.restaurant_features[rid] = row.fillna(0).to_dict()
            log.info(f"  Cached {len(self.restaurant_features)} restaurant feature vectors")

        # --- Item features: group by item_id ---
        avail_item_cols = [c for c in ITEM_COLS if c in df.columns]
        if avail_item_cols:
            item_df = df.groupby("item_id")[avail_item_cols].first()
            emb_cols = [c for c in avail_item_cols if c.startswith("item_emb_")]
            for iid, row in item_df.iterrows():
                feats = row.fillna(0).to_dict()
                self.item_features[iid] = feats
                if emb_cols:
                    self.item_embeddings[iid] = np.array([feats[c] for c in emb_cols],
                                                         dtype=np.float32)
            log.info(f"  Cached {len(self.item_features)} item feature vectors "
                     f"({len(self.item_embeddings)} with embeddings)")

        # --- Zone stats: per city ---
        city_cols = [c for c in df.columns if c.startswith("city_")]
        if city_cols and "zone_avg_aov" in df.columns:
            for city_col in city_cols:
                city_name = city_col.replace("city_", "")
                city_rows = df[df[city_col] == 1]
                if len(city_rows) > 0:
                    self.zone_stats[city_name] = {
                        "zone_avg_aov": float(city_rows["zone_avg_aov"].mean()),
                        "zone_addon_rate": float(city_rows.get("zone_addon_rate",
                                                               pd.Series([0.15])).mean()),
                    }
            log.info(f"  Cached zone stats for {len(self.zone_stats)} cities")

        elapsed = time.time() - start
        log.info(f"  Feature store loaded in {elapsed:.1f}s")

    def get_user_features(
        self,
        user_id: str,
        city: str | None = None,
        city_defaults: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Get user features, falling back to city-specific or global cold-start defaults."""
        if user_id in self.user_features:
            return self.user_features[user_id]
        # City-specific cold-start
        if city and city_defaults and city in city_defaults:
            return dict(city_defaults[city])
        # Global fallback
        return dict(COLD_START_USER)

    def get_item_features(self, item_id: str) -> dict[str, float]:
        """Get item features with zero-fill defaults."""
        return self.item_features.get(item_id, {c: 0.0 for c in ITEM_COLS})

    def get_restaurant_features(self, restaurant_id: str) -> dict[str, float]:
        """Get restaurant features with zero-fill defaults."""
        return self.restaurant_features.get(restaurant_id, {c: 0.0 for c in RESTAURANT_COLS})

    def get_item_embedding(self, item_id: str) -> np.ndarray:
        """Get 30-dim TF-IDF embedding for an item."""
        return self.item_embeddings.get(item_id, np.zeros(30, dtype=np.float32))


def build_temporal_features(meal_period: str, city: str) -> dict[str, float]:
    """Compute temporal features from the current request context."""
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()

    features: dict[str, float] = {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "meal_breakfast": 1.0 if meal_period == "breakfast" else 0.0,
        "meal_lunch": 1.0 if meal_period == "lunch" else 0.0,
        "meal_snack": 1.0 if meal_period == "snack" else 0.0,
        "meal_dinner": 1.0 if meal_period == "dinner" else 0.0,
        "meal_late_night": 1.0 if meal_period == "late_night" else 0.0,
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "is_holiday": 0.0,
    }

    # City one-hot
    city_lower = city.lower()
    for c in ["mumbai", "delhi", "bangalore", "hyderabad", "pune"]:
        features[f"city_{c}"] = 1.0 if city_lower == c else 0.0

    return features


def build_cart_features(
    cart_items: list[dict[str, Any]],
    user_feats: dict[str, float],
    cuisine: str,
    item_embeddings: dict[str, np.ndarray],
    meal_templates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute cart features from the current cart state."""
    n_items = max(len(cart_items), 1)
    cart_categories = set()
    cart_prices = []
    cart_emb_sum = np.zeros(30, dtype=np.float32)

    for ci in cart_items:
        cart_categories.add(ci.get("category", "main"))
        cart_prices.append(ci.get("price", 150.0))
        emb = item_embeddings.get(ci.get("item_id", ""), np.zeros(30, dtype=np.float32))
        cart_emb_sum += emb

    avg_price = np.mean(cart_prices) if cart_prices else 150.0
    total_value = sum(cart_prices)

    # Meal completeness (use learned templates if provided, else hardcoded)
    _templates = meal_templates if meal_templates is not None else MEAL_TEMPLATES
    template = _templates.get(cuisine, {"required": ["main", "side", "drink"], "total": 3})
    required = template["required"]
    present = len(set(required) & cart_categories)
    completeness = present / template["total"] if template["total"] > 0 else 0.0
    missing = template["total"] - present

    # Generic completeness
    generic_cats = {"main", "side", "drink", "dessert"}
    generic_present = len(generic_cats & cart_categories)
    generic_comp = generic_present / 4.0

    # Price headroom
    monetary_avg = user_feats.get("monetary_avg", 350.0)
    headroom = max(monetary_avg - total_value, 0.0)
    tier_match = 1.0 if abs(total_value - monetary_avg) < monetary_avg * 0.3 else 0.0

    features: dict[str, float] = {
        "cart_item_count": float(n_items),
        "cart_total_value": total_value,
        "cart_avg_price": avg_price,
        "cart_max_price": avg_price * 1.5 if cart_prices else 150.0,
        "cart_min_price": avg_price * 0.6 if cart_prices else 90.0,
        "cart_price_std": avg_price * 0.3 if len(cart_prices) > 1 else 0.0,
        "cart_has_main": 1.0 if "main" in cart_categories else 0.0,
        "cart_has_side": 1.0 if "side" in cart_categories else 0.0,
        "cart_has_drink": 1.0 if "drink" in cart_categories else 0.0,
        "cart_has_dessert": 1.0 if "dessert" in cart_categories else 0.0,
        "cart_has_bread": 1.0 if "bread" in cart_categories else 0.0,
        "cart_completeness_score": completeness,
        "cart_missing_components": float(missing),
        "cart_generic_completeness": generic_comp,
        "cart_veg_ratio": user_feats.get("veg_ratio", 0.35),
        "cart_cuisine_entropy": 0.5 if n_items > 1 else 0.0,
        "cart_price_headroom": headroom,
        "cart_price_tier_match": tier_match,
        "cart_items_added_count": float(n_items),
    }

    # Last added category one-hot (use last cart item's category)
    last_cat = cart_items[-1].get("category", "main") if cart_items else "main"
    for cat in ["main", "side", "drink", "dessert", "bread"]:
        features[f"cart_last_added_category_{cat}"] = 1.0 if last_cat == cat else 0.0

    # Cart embeddings (averaged)
    if n_items > 0:
        cart_emb_avg = cart_emb_sum / n_items
    else:
        cart_emb_avg = np.zeros(30, dtype=np.float32)
    for i in range(30):
        features[f"cart_emb_{i}"] = float(cart_emb_avg[i])

    return features


def build_fire_seq_features(position: int = 1) -> dict[str, float]:
    """Compute fire sequence/fatigue features."""
    return {
        "fire_seq_is_first": 1.0 if position == 1 else 0.0,
        "fire_seq_is_late": 1.0 if position >= 3 else 0.0,
        "fire_seq_fatigue": 1.0 / (1.0 + 0.3 * (position - 1)),
    }


def build_cross_features(
    user_feats: dict[str, float],
    item_feats: dict[str, float],
    rest_feats: dict[str, float],
    cart_feats: dict[str, float],
    temporal_feats: dict[str, float],
    fire_feats: dict[str, float],
    pmi_scores: dict[str, float],
    cart_item_ids: list[str],
    candidate_item_id: str,
    candidate_embedding: np.ndarray,
    cart_embedding: np.ndarray,
    position_in_rail: int = 1,
) -> dict[str, float]:
    """Compute cross/interaction features between cart context and candidate item."""
    # PMI features
    pmi_values = []
    for cid in cart_item_ids:
        key = f"{cid}|{candidate_item_id}"
        pmi = pmi_scores.get(key, 0.0)
        if pmi > 0:
            pmi_values.append(pmi)

    max_pmi = max(pmi_values) if pmi_values else 0.0
    avg_pmi = max_pmi * 0.7
    min_pmi = max_pmi * 0.3

    # Embedding similarity
    cand_norm = np.linalg.norm(candidate_embedding)
    cart_norm = np.linalg.norm(cart_embedding)
    if cand_norm > 1e-8 and cart_norm > 1e-8:
        emb_sim = float(np.dot(candidate_embedding / cand_norm, cart_embedding / cart_norm))
    else:
        emb_sim = 0.0

    # Price features
    cart_avg_price = cart_feats.get("cart_avg_price", 150.0)
    item_price = item_feats.get("item_price", 100.0)
    price_ratio = item_price / max(cart_avg_price, 1.0)
    headroom = cart_feats.get("cart_price_headroom", 200.0)
    price_vs_headroom = item_price / max(headroom, 1.0)

    cart_total = cart_feats.get("cart_total_value", 0.0)
    monetary_avg = user_feats.get("monetary_avg", 350.0)
    fits_budget = 1.0 if (cart_total + item_price) <= monetary_avg * 1.3 else 0.0

    # Complementarity
    cart_categories = set()
    if cart_feats.get("cart_has_main", 0): cart_categories.add("main")
    if cart_feats.get("cart_has_side", 0): cart_categories.add("side")
    if cart_feats.get("cart_has_drink", 0): cart_categories.add("drink")
    if cart_feats.get("cart_has_dessert", 0): cart_categories.add("dessert")
    if cart_feats.get("cart_has_bread", 0): cart_categories.add("bread")

    # Determine candidate category
    cand_cat = "unknown"
    for cat in ["main", "side", "drink", "dessert", "bread"]:
        if item_feats.get(f"item_cat_{cat}", 0) > 0.5:
            cand_cat = cat
            break

    fills_missing = 1.0 if cand_cat not in cart_categories and cand_cat != "unknown" else 0.0

    # User-item affinity
    cat_pref = user_feats.get(f"category_pref_{cand_cat}", 1 / 6)
    item_is_veg = item_feats.get("item_is_veg", 0)
    user_veg_ratio = user_feats.get("veg_ratio", 0.35)
    veg_match = 1.0 if (item_is_veg > 0.5 and user_veg_ratio > 0.5) or \
                        (item_is_veg < 0.5 and user_veg_ratio <= 0.5) else 0.0

    completeness = cart_feats.get("cart_completeness_score", 0.0)
    item_count_30d = item_feats.get("item_order_count_30d", 0.0)
    item_is_bestseller = item_feats.get("item_is_bestseller", 0.0)
    is_cold_start = user_feats.get("is_cold_start", 0.0)
    is_weekend = temporal_feats.get("is_weekend", 0.0)
    addon_rate = user_feats.get("addon_adoption_rate", 0.10)
    rest_is_chain = rest_feats.get("rest_is_chain", 0.0)
    mealtime_cov = user_feats.get("mealtime_coverage", 0.0)
    recency_days = max(user_feats.get("recency_days", 999.0), 1.0)
    freq_30d = user_feats.get("frequency_30d", 0.0)
    rfm_seg = user_feats.get("rfm_segment", 4.0)
    cart_item_count = cart_feats.get("cart_item_count", 1.0)
    rest_rating = rest_feats.get("rest_avg_rating", 3.5)
    rest_volume = rest_feats.get("rest_monthly_volume", 100.0)
    item_addon_rate = item_feats.get("item_addon_order_rate", 0.1)
    csao_ctr = user_feats.get("csao_historical_ctr", 0.15)
    price_sens = user_feats.get("price_sensitivity", 1.0)
    generic_comp = cart_feats.get("cart_generic_completeness", 0.0)
    fatigue = fire_feats.get("fire_seq_fatigue", 1.0)
    is_late = fire_feats.get("fire_seq_is_late", 0.0)
    meal_dinner = temporal_feats.get("meal_dinner", 0.0)

    features: dict[str, float] = {
        # Base cross features
        "cross_max_pmi": max_pmi,
        "cross_avg_pmi": avg_pmi,
        "cross_min_pmi": min_pmi,
        "cross_has_high_pmi": 1.0 if max_pmi > 2.0 else 0.0,
        "cross_pmi_std": abs(max_pmi) * 0.2,
        "cross_embedding_similarity": emb_sim,
        "cross_price_ratio": price_ratio,
        "cross_price_vs_headroom": min(price_vs_headroom, 10.0),
        "cross_fits_budget": fits_budget,
        "cross_fills_missing_component": fills_missing,
        "cross_same_category_in_cart": 0.0,
        "cross_user_category_pref": cat_pref,
        "cross_veg_match": veg_match,
        "cross_position_in_rail": float(position_in_rail),
        # Interaction features
        "cross_pmi_x_completeness": max_pmi * completeness,
        "cross_price_ratio_x_headroom": price_ratio * headroom / 500.0,
        "cross_fills_gap_x_user_pref": fills_missing * cat_pref,
        "cross_popularity_x_dinner": item_count_30d * meal_dinner,
        "cross_veg_match_x_price": veg_match * price_ratio,
        "cross_bestseller_x_pmi": item_is_bestseller * max_pmi,
        "cross_cold_start_x_popularity": is_cold_start * item_count_30d,
        "cross_weekend_x_addon": is_weekend * addon_rate,
        "cross_chain_x_bestseller": rest_is_chain * item_is_bestseller,
        "cross_mealtime_coverage_x_pmi": mealtime_cov * max_pmi,
        "cross_recency_x_popularity": (1.0 / recency_days) * item_count_30d,
        "cross_frequency_x_price": freq_30d * price_ratio,
        "cross_rfm_x_addon_rate": rfm_seg * item_addon_rate,
        "cross_cart_size_x_completeness": cart_item_count * completeness,
        "cross_rest_rating_x_bestseller": rest_rating * item_is_bestseller,
        "cross_rest_volume_x_popularity": (rest_volume / 100.0) * item_count_30d,
        "cross_user_addon_x_item_addon": addon_rate * item_addon_rate,
        "cross_csao_ctr_x_position": csao_ctr * (1.0 / max(position_in_rail, 1)),
        "cross_price_sensitivity_x_item_price": price_sens * item_price / 300.0,
        "cross_generic_comp_x_fills": generic_comp * fills_missing,
        "cross_fatigue_x_pmi": fatigue * max_pmi,
        "cross_fatigue_x_popularity": fatigue * item_count_30d,
        "cross_late_fire_x_bestseller": is_late * item_is_bestseller,
    }

    return features


def build_llm_features(
    emb_sim: float,
    fills_missing: float,
    candidate_embedding: np.ndarray,
    cart_embedding: np.ndarray,
    max_pmi: float,
) -> dict[str, float]:
    """Compute LLM/semantic features (offline proxy using TF-IDF)."""
    max_sim = max(emb_sim, 0.0)
    complementarity = fills_missing * 0.5 + max_sim * 0.5
    cand_norm = float(np.linalg.norm(candidate_embedding))
    cart_norm = float(np.linalg.norm(cart_embedding))

    return {
        "llm_max_semantic_sim": max_sim,
        "llm_avg_semantic_sim": max_sim * 0.8,
        "llm_min_semantic_sim": max_sim * 0.5,
        "llm_complementarity_score": complementarity,
        "llm_name_overlap": emb_sim * 0.3,
        "llm_cand_emb_norm": cand_norm,
        "llm_cart_emb_norm": cart_norm,
        "llm_emb_norm_ratio": cand_norm / max(cart_norm, 1e-8),
        "llm_comp_x_pmi": complementarity * max_pmi,
        "llm_sim_x_fills_gap": max_sim * fills_missing,
    }


def build_full_feature_vector(
    user_id: str,
    candidate_item_id: str,
    restaurant_id: str,
    cart_items: list[dict[str, Any]],
    meal_period: str,
    city: str,
    feature_store: FeatureStore,
    pmi_scores: dict[str, float],
    position_in_rail: int = 1,
    meal_templates: dict[str, dict[str, Any]] | None = None,
    city_defaults: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Build the complete feature vector for a single candidate item.

    Assembles all 223 features from static lookups + real-time computation.
    """
    # Static lookups
    user_feats = feature_store.get_user_features(user_id, city=city, city_defaults=city_defaults)
    item_feats = feature_store.get_item_features(candidate_item_id)
    rest_feats = feature_store.get_restaurant_features(restaurant_id)

    # Embeddings
    cand_emb = feature_store.get_item_embedding(candidate_item_id)
    cart_item_ids = [ci.get("item_id", "") for ci in cart_items]

    # Cart embedding (average of cart item embeddings)
    cart_emb = np.zeros(30, dtype=np.float32)
    for cid in cart_item_ids:
        cart_emb += feature_store.get_item_embedding(cid)
    if len(cart_item_ids) > 0:
        cart_emb /= len(cart_item_ids)

    # Determine cuisine from restaurant features
    cuisine = "North Indian"
    cuisine_map = {
        "rest_cuisine_north_indian": "North Indian",
        "rest_cuisine_south_indian": "South Indian",
        "rest_cuisine_chinese": "Chinese",
        "rest_cuisine_biryani": "Biryani",
        "rest_cuisine_street_food": "Street Food",
        "rest_cuisine_continental": "Continental",
        "rest_cuisine_maharashtrian": "Maharashtrian",
    }
    for col, name in cuisine_map.items():
        if rest_feats.get(col, 0) > 0.5:
            cuisine = name
            break

    # Compute feature groups
    temporal_feats = build_temporal_features(meal_period, city)
    cart_feats = build_cart_features(cart_items, user_feats, cuisine,
                                     feature_store.item_embeddings,
                                     meal_templates=meal_templates)
    fire_feats = build_fire_seq_features(position_in_rail)
    cross_feats = build_cross_features(
        user_feats, item_feats, rest_feats, cart_feats, temporal_feats,
        fire_feats, pmi_scores, cart_item_ids, candidate_item_id,
        cand_emb, cart_emb, position_in_rail,
    )

    # LLM features
    emb_sim = cross_feats["cross_embedding_similarity"]
    fills_missing = cross_feats["cross_fills_missing_component"]
    max_pmi = cross_feats["cross_max_pmi"]
    llm_feats = build_llm_features(emb_sim, fills_missing, cand_emb, cart_emb, max_pmi)

    # Zone stats
    zone = feature_store.zone_stats.get(city.lower(), ZONE_STATS_DEFAULT)

    # Assemble full vector
    vector: dict[str, float] = {}
    vector.update(user_feats)
    vector.update(rest_feats)
    vector.update(item_feats)
    vector.update(temporal_feats)
    vector.update(zone)
    vector.update(cart_feats)
    vector.update(fire_feats)
    vector.update(cross_feats)
    vector.update(llm_feats)

    # Position feature (also in meta, needs to be numeric for some models)
    vector["position"] = float(position_in_rail)

    return vector


def build_feature_matrix(
    feature_vectors: list[dict[str, float]],
    feature_cols: list[str],
) -> np.ndarray:
    """Convert list of feature dicts to a numpy matrix aligned to feature_cols order."""
    n = len(feature_vectors)
    m = len(feature_cols)
    matrix = np.zeros((n, m), dtype=np.float32)
    for i, vec in enumerate(feature_vectors):
        for j, col in enumerate(feature_cols):
            matrix[i, j] = vec.get(col, 0.0)
    return matrix
