"""
CSAO Demo Backend — FastAPI server with real ML-powered add-on recommendations.

Pipeline: Two-Tower + FAISS → LightGBM L1 → SASRec → DCN-v2 L2 → MMR
Loads trained model artifacts from ../outputs/ at startup.
Runs on http://localhost:8000
"""

import csv
import json
import logging
import math
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("csao-backend")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Restaurant name generator (restaurants.csv has no name column)
# ---------------------------------------------------------------------------
CUISINE_NAMES: dict[str, list[str]] = {
    "North Indian": [
        "Punjabi Dhaba", "Tandoori Nights", "Spice Route", "Royal Kitchen",
        "Mughal Darbar", "Butter Story", "Delhi Highway", "Amritsari Haveli",
        "Nawab's Kitchen", "Saffron House", "Roti & Curry", "Punjab Grill",
        "Moti Mahal", "Frontier Grill", "Dum Pukht", "Dilli 6",
        "Karim's Kitchen", "Gulati's", "Chor Bizarre", "Handi House",
    ],
    "South Indian": [
        "Dosa Plaza", "Madras Cafe", "Udupi Grand", "Idli Factory",
        "Saravana Bhavan", "Meghana Foods", "Vidyarthi Bhavan", "Namma Cafe",
        "Filter Coffee House", "Andhra Mess", "Chettinad Palace", "MTR Express",
        "Rasa Kitchen", "Thali House", "Banana Leaf", "Mylapore Café",
        "Temple Kitchen", "Coorg Corner", "Malabar Junction", "Nilgiri Grill",
    ],
    "Biryani": [
        "Paradise Biryani", "Bawarchi", "Shah Ghouse", "Biryani Blues",
        "Behrouz Biryani", "Meghana Biryani", "Hyderabad House", "Pista House",
        "Cafe Bahar", "Lucky Biryani", "Biryani Pot", "Biryani Mahal",
        "Dum Biryani House", "Kolkata Biryani", "Zaiqa Biryani", "Biryaniwala",
        "Nawabi Biryani", "Shadab", "Sarvi Corner", "Royal Biryani",
    ],
    "Chinese": [
        "Wok Express", "Dragon Bowl", "Noodle Bar", "Chung Wah",
        "Mainland China", "Wonton House", "Golden Dragon", "Szechwan Court",
        "Hakka Noodles", "Bamboo Garden", "Stir Crazy", "Oriental Kitchen",
        "Ming's Wok", "Panda Express", "Red Lantern", "Dim Sum House",
        "Lucky Chopsticks", "Jade Garden", "Chopstick Express", "Peking Palace",
    ],
    "Street Food": [
        "Chaat Corner", "Mumbai Pav Bhaji", "Golgappa Station", "Samosa House",
        "Street Bites", "Vada Pav King", "Bhel Puri Stall", "Kachori Express",
        "Dabeli House", "Frankie's", "Tikki Junction", "Chat Chaat",
        "Dahi Bhalla Wala", "Jalebi House", "Kulfi Center", "Pani Puri Stop",
        "Ragda Patis", "Aloo Tikki Hub", "Pyaaz Kachori", "Roll Maal",
    ],
    "Maharashtrian": [
        "Misal Pav House", "Puran Poli Hub", "Kolhapuri Kitchen", "Aamhi Maharashtrian",
        "Bhajipalya", "Zunka Bhakar", "Kombdi Vade", "Konkan Katta",
        "Malvani Aroma", "Thalipeeth House", "Pitla Corner", "Modak Mandir",
        "Shivaji Kitchen", "Mastani Corner", "Tambda Rassa", "Pandhra Rassa House",
        "Bharli Vangi", "Sol Kadhi Stop", "Ukdiche Modak", "Pav Bhaji Centre",
    ],
    "Continental": [
        "Bistro 101", "The Daily Bread", "Olive Garden", "Rustic Table",
        "La Piazza", "Cafe Europa", "Pasta Street", "Steak House",
        "Grill & Chill", "Baker's Basket", "The Blue Door", "Green Leaf Cafe",
        "Brick Oven", "Sunrise Diner", "The Patio", "Urban Plate",
        "Fork & Knife", "Artisan Kitchen", "The Salad Bar", "Crepe Station",
    ],
}

CITY_SUFFIXES = {
    "Mumbai": ["Andheri", "Bandra", "Dadar", "Juhu", "Powai", "Thane", "Worli", "Colaba"],
    "Delhi": ["Connaught Place", "Karol Bagh", "Lajpat Nagar", "Hauz Khas", "Saket", "Rajouri"],
    "Bangalore": ["Indiranagar", "Koramangala", "Whitefield", "JP Nagar", "HSR Layout", "MG Road"],
    "Hyderabad": ["Jubilee Hills", "Banjara Hills", "Madhapur", "Gachibowli", "Ameerpet", "Secunderabad"],
    "Pune": ["Koregaon Park", "Viman Nagar", "Hinjewadi", "Kothrud", "FC Road", "Aundh"],
}

CUISINE_EMOJI: dict[str, str] = {
    "North Indian": "\U0001f35b",
    "South Indian": "\U0001f958",
    "Biryani": "\U0001f35a",
    "Chinese": "\U0001f35c",
    "Street Food": "\U0001f37f",
    "Maharashtrian": "\U0001fad5",
    "Continental": "\U0001f35d",
}

MEAL_TEMPLATES: dict[str, list[str]] = {
    "North Indian": ["main", "bread", "side", "drink"],
    "South Indian": ["main", "side", "drink", "dessert"],
    "Biryani": ["main", "side", "drink", "dessert"],
    "Chinese": ["main", "side", "drink"],
    "Street Food": ["main", "side", "drink"],
    "Maharashtrian": ["main", "bread", "side", "drink"],
    "Continental": ["main", "side", "drink", "dessert"],
}


def generate_restaurant_name(rid: str, cuisine: str, city: str, rng: np.random.Generator) -> str:
    """Generate a plausible restaurant name from cuisine + city."""
    names = CUISINE_NAMES.get(cuisine, CUISINE_NAMES["North Indian"])
    suffixes = CITY_SUFFIXES.get(city, ["Central"])
    idx = int(rid.replace("R", "")) % len(names)
    suffix_idx = int(rid.replace("R", "")) % len(suffixes)
    name = names[idx]
    if int(rid.replace("R", "")) >= len(names):
        name = f"{name} - {suffixes[suffix_idx]}"
    return name


def estimate_delivery_time(price_tier: str, rng: np.random.Generator) -> str:
    """Generate estimated delivery time based on price tier."""
    base = {"budget": 20, "mid": 25, "premium": 35}.get(price_tier, 25)
    variance = rng.integers(-5, 6)
    low = max(15, base + variance)
    high = low + rng.integers(5, 11)
    return f"{low}-{high} min"


# ---------------------------------------------------------------------------
# ML Model Manager — loads all trained model artifacts
# ---------------------------------------------------------------------------
class MLModelManager:
    """Loads and manages all trained ML model artifacts."""

    def __init__(self) -> None:
        self.models_loaded = False
        self.models_used: list[str] = []

        # Two-Tower
        self.two_tower_model = None
        self.faiss_index = None
        self.tt_item_embeddings: np.ndarray | None = None
        self.tt_item_ids: np.ndarray | None = None
        self.tt_item_id_to_idx: dict[str, int] = {}
        self.tt_config: dict[str, Any] = {}

        # LightGBM
        self.lgbm_model = None
        self.l1_feature_cols: list[str] = []

        # SASRec
        self.sasrec_model = None
        self.sasrec_item_id_map: dict[str, int] = {}
        self.sasrec_meta: dict[str, Any] = {}

        # DCN-v2
        self.dcnv2_model = None
        self.dcnv2_mean: np.ndarray | None = None
        self.dcnv2_std: np.ndarray | None = None
        self.l2_feature_cols: list[str] = []

        # SHAP importance (for model-driven reason generation)
        self.shap_importance: dict[str, float] = {}

    def load(self) -> None:
        """Load all available model artifacts from outputs/."""
        log.info("Loading ML model artifacts...")
        self.models_used = []

        self._load_two_tower()
        self._load_lightgbm()
        self._load_sasrec()
        self._load_dcnv2()
        self._load_shap_importance()

        if self.models_used:
            self.models_loaded = True
            log.info(f"ML models loaded: {', '.join(self.models_used)}")
        else:
            log.warning("No ML models loaded — using PMI+heuristic fallback")

    def _load_two_tower(self) -> None:
        """Load Two-Tower model + FAISS index."""
        try:
            import torch
            import faiss
            from models import TwoTowerModel

            config_path = OUTPUT_DIR / "two_tower_config.json"
            model_path = OUTPUT_DIR / "two_tower_model.pt"
            faiss_path = OUTPUT_DIR / "two_tower_faiss.index"
            emb_path = OUTPUT_DIR / "two_tower_item_embeddings.npy"
            ids_path = OUTPUT_DIR / "two_tower_item_ids.npy"

            if not all(p.exists() for p in [config_path, model_path, faiss_path, emb_path, ids_path]):
                log.warning("Two-Tower artifacts missing — skipping")
                return

            with open(config_path) as f:
                self.tt_config = json.load(f)

            self.two_tower_model = TwoTowerModel(
                user_dim=self.tt_config["user_dim"],
                cart_dim=self.tt_config["cart_dim"],
                context_dim=self.tt_config["context_dim"],
                item_dim=self.tt_config["item_dim"],
                hidden_dim=self.tt_config["hidden_dim"],
                seq_dim=self.tt_config.get("seq_dim", 0),
            )
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.two_tower_model.load_state_dict(state)
            self.two_tower_model.eval()

            self.faiss_index = faiss.read_index(str(faiss_path))
            self.tt_item_embeddings = np.load(emb_path)
            self.tt_item_ids = np.load(ids_path, allow_pickle=True)
            self.tt_item_id_to_idx = {str(iid): i for i, iid in enumerate(self.tt_item_ids)}

            log.info(f"  Two-Tower loaded: {self.faiss_index.ntotal} items, "
                     f"dim={self.tt_item_embeddings.shape[1]}")
            self.models_used.append("Two-Tower+FAISS")

        except Exception as e:
            log.warning(f"Two-Tower load failed: {e}")

    def _load_lightgbm(self) -> None:
        """Load LightGBM L1 model."""
        try:
            import lightgbm as lgb

            model_path = OUTPUT_DIR / "lgbm_l1_model.txt"
            cols_path = OUTPUT_DIR / "l1_feature_cols.json"

            if not model_path.exists() or not cols_path.exists():
                log.warning("LightGBM artifacts missing — skipping")
                return

            self.lgbm_model = lgb.Booster(model_file=str(model_path))
            with open(cols_path) as f:
                self.l1_feature_cols = json.load(f)

            log.info(f"  LightGBM L1 loaded: {len(self.l1_feature_cols)} features")
            self.models_used.append("LightGBM-L1")

        except Exception as e:
            log.warning(f"LightGBM load failed: {e}")

    def _load_sasrec(self) -> None:
        """Load SASRec sequential model."""
        try:
            import torch
            from models import SASRecModel

            model_path = OUTPUT_DIR / "sasrec_model.pt"
            meta_path = OUTPUT_DIR / "sasrec_meta.json"

            if not model_path.exists() or not meta_path.exists():
                log.warning("SASRec artifacts missing — skipping")
                return

            with open(meta_path) as f:
                self.sasrec_meta = json.load(f)

            self.sasrec_item_id_map = self.sasrec_meta["item_id_map"]

            self.sasrec_model = SASRecModel(
                num_items=self.sasrec_meta["num_items"],
                emb_dim=self.sasrec_meta["emb_dim"],
                n_heads=self.sasrec_meta["n_heads"],
                n_layers=self.sasrec_meta["n_layers"],
                max_seq_len=self.sasrec_meta["max_seq_len"],
            )
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.sasrec_model.load_state_dict(state)
            self.sasrec_model.eval()

            log.info(f"  SASRec loaded: vocab={self.sasrec_meta['num_items']}, "
                     f"emb_dim={self.sasrec_meta['emb_dim']}")
            self.models_used.append("SASRec")

        except Exception as e:
            log.warning(f"SASRec load failed: {e}")

    def _load_dcnv2(self) -> None:
        """Load DCN-v2 L2 model."""
        try:
            import torch
            from models import DCNv2Model

            model_path = OUTPUT_DIR / "dcnv2_model.pt"
            mean_path = OUTPUT_DIR / "dcnv2_norm_mean.npy"
            std_path = OUTPUT_DIR / "dcnv2_norm_std.npy"
            cols_path = OUTPUT_DIR / "l2_feature_cols.json"

            if not model_path.exists() or not mean_path.exists() or not std_path.exists():
                log.warning("DCN-v2 artifacts missing — skipping")
                return

            self.dcnv2_mean = np.load(mean_path)
            self.dcnv2_std = np.load(std_path)
            input_dim = len(self.dcnv2_mean)

            self.dcnv2_model = DCNv2Model(
                input_dim=input_dim,
                num_cross_layers=3,
                deep_dims=[256, 128, 64],
                dropout=0.1,
            )
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.dcnv2_model.load_state_dict(state)
            self.dcnv2_model.eval()

            if cols_path.exists():
                with open(cols_path) as f:
                    self.l2_feature_cols = json.load(f)
            else:
                self.l2_feature_cols = []

            log.info(f"  DCN-v2 loaded: input_dim={input_dim}, "
                     f"L2 features={len(self.l2_feature_cols)}")
            self.models_used.append("DCN-v2")

        except Exception as e:
            log.warning(f"DCN-v2 load failed: {e}")

    def _load_shap_importance(self) -> None:
        """Load SHAP feature importances for model-driven reason generation."""
        try:
            fi_path = OUTPUT_DIR / "feature_importance.json"
            if not fi_path.exists():
                log.warning("feature_importance.json missing — reasons will use fallback")
                return
            with open(fi_path) as f:
                data = json.load(f)
            shap_list = data.get("shap_analysis", {}).get("top_20_shap", [])
            self.shap_importance = {
                entry["feature"]: entry["mean_abs_shap"] for entry in shap_list
            }
            log.info(f"  SHAP importance loaded: {len(self.shap_importance)} features")
        except Exception as e:
            log.warning(f"SHAP importance load failed: {e}")

    def get_two_tower_scores(
        self,
        user_feats: dict[str, float],
        cart_items: list[dict[str, Any]],
        candidate_item_ids: list[str],
        feature_store: Any,
        seq_emb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute Two-Tower scores for candidate items via query-item dot product."""
        if self.two_tower_model is None:
            return {}

        import torch
        import torch.nn as nn

        try:
            tt_config = self.tt_config
            user_cols = tt_config.get("user_cols", [])
            cart_cols = tt_config.get("cart_cols", [])
            context_cols = tt_config.get("context_cols", [])
            seq_dim = tt_config.get("seq_dim", 0)

            # Build query feature vectors
            user_arr = np.array([[user_feats.get(c, 0.0) for c in user_cols]], dtype=np.float32)
            cart_arr = np.array([[user_feats.get(c, 0.0) for c in cart_cols]], dtype=np.float32)
            ctx_arr = np.array([[user_feats.get(c, 0.0) for c in context_cols]], dtype=np.float32)

            # Get query embedding (with sequential context from SASRec)
            with torch.no_grad():
                u_t = torch.tensor(user_arr)
                c_t = torch.tensor(cart_arr)
                ctx_t = torch.tensor(ctx_arr)

                # Pass SASRec sequential embedding if available
                seq_t = None
                if seq_dim > 0 and seq_emb is not None:
                    seq_arr = np.array([seq_emb[:seq_dim]], dtype=np.float32)
                    seq_t = torch.tensor(seq_arr)

                query_emb = self.two_tower_model.query_tower(u_t, c_t, ctx_t, seq_t)
                query_emb = nn.functional.normalize(query_emb, p=2, dim=1)
                query_vec = query_emb.numpy()[0]  # (64,)

            # Score each candidate by dot product with pre-computed item embedding
            scores: dict[str, float] = {}
            for item_id in candidate_item_ids:
                idx = self.tt_item_id_to_idx.get(item_id)
                if idx is not None:
                    item_emb = self.tt_item_embeddings[idx]  # already L2 normalized
                    scores[item_id] = float(np.dot(query_vec, item_emb))
                else:
                    scores[item_id] = 0.0

            return scores

        except Exception as e:
            log.warning(f"Two-Tower scoring failed: {e}")
            return {}

    def get_lgbm_scores(
        self,
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """Score candidates with LightGBM L1."""
        if self.lgbm_model is None:
            return np.zeros(feature_matrix.shape[0])

        try:
            return self.lgbm_model.predict(feature_matrix)
        except Exception as e:
            log.warning(f"LightGBM scoring failed: {e}")
            return np.zeros(feature_matrix.shape[0])

    def get_sasrec_embeddings(
        self,
        cart_item_ids: list[str],
    ) -> np.ndarray:
        """Get sequential context embedding from SASRec for the cart sequence."""
        if self.sasrec_model is None:
            return np.zeros(16, dtype=np.float32)

        import torch

        try:
            max_seq = self.sasrec_meta.get("max_seq_len", 20)

            # Map item IDs to integer indices
            mapped = [self.sasrec_item_id_map.get(iid, 0) for iid in cart_item_ids]

            # Pad/truncate
            if len(mapped) > max_seq:
                mapped = mapped[-max_seq:]
            else:
                mapped = [0] * (max_seq - len(mapped)) + mapped

            seq_tensor = torch.tensor([mapped], dtype=torch.long)

            with torch.no_grad():
                _, context_emb = self.sasrec_model(seq_tensor)
                emb = context_emb.numpy()[0]  # (64,)

            # Return first 16 dims (as used in training)
            return emb[:16].astype(np.float32)

        except Exception as e:
            log.warning(f"SASRec embedding failed: {e}")
            return np.zeros(16, dtype=np.float32)

    def get_dcnv2_scores(
        self,
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """Score candidates with DCN-v2 L2."""
        if self.dcnv2_model is None:
            return np.zeros(feature_matrix.shape[0])

        import torch

        try:
            # Normalize using training mean/std (clamp std to avoid division by zero)
            safe_std = np.maximum(self.dcnv2_std, 1e-8)
            normed = (feature_matrix - self.dcnv2_mean) / safe_std
            # Replace any remaining NaN/inf with 0
            normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
            feat_tensor = torch.tensor(normed.astype(np.float32))

            with torch.no_grad():
                logits = self.dcnv2_model(feat_tensor)
                scores = torch.sigmoid(logits).numpy()

            # Replace NaN scores with 0
            scores = np.nan_to_num(scores, nan=0.0)
            return scores

        except Exception as e:
            log.warning(f"DCN-v2 scoring failed: {e}")
            return np.zeros(feature_matrix.shape[0])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class DataStore:
    """In-memory data store loaded from CSVs at startup."""

    def __init__(self) -> None:
        self.restaurants: list[dict[str, Any]] = []
        self.restaurant_map: dict[str, dict[str, Any]] = {}
        self.menu_by_restaurant: dict[str, list[dict[str, Any]]] = {}
        self.users: list[dict[str, Any]] = []
        self.user_map: dict[str, dict[str, Any]] = {}
        self.orders: list[dict[str, Any]] = []
        self.pmi_scores: dict[str, float] = {}
        self.item_popularity: dict[str, float] = {}
        self.item_map: dict[str, dict[str, Any]] = {}
        self.learned_meal_templates: dict[str, dict[str, Any]] = {}
        self.city_cold_start_defaults: dict[str, dict[str, float]] = {}
        self.stats: dict[str, Any] = {}

    def load(self) -> None:
        """Load all data from CSVs and pre-compute PMI."""
        start = time.time()
        rng = np.random.default_rng(42)

        # --- Restaurants ---
        log.info("Loading restaurants...")
        rdf = pd.read_csv(DATA_DIR / "restaurants.csv")
        for rec in rdf.to_dict("records"):
            rid = rec["restaurant_id"]
            cuisine = rec["cuisine_type"]
            city = rec["city"]
            name = generate_restaurant_name(rid, cuisine, city, rng)
            r = {
                "restaurant_id": rid,
                "name": name,
                "city": city,
                "cuisine_type": cuisine,
                "price_tier": rec["price_tier"],
                "avg_rating": round(float(rec["avg_rating"]), 1),
                "is_chain": bool(rec["is_chain"]),
                "delivery_time": estimate_delivery_time(rec["price_tier"], rng),
                "emoji": CUISINE_EMOJI.get(cuisine, "\U0001f37d\ufe0f"),
            }
            self.restaurants.append(r)
            self.restaurant_map[rid] = r

        # --- Menu Items ---
        log.info("Loading menu items...")
        mdf = pd.read_csv(DATA_DIR / "menu_items.csv")
        for rec in mdf.to_dict("records"):
            cat = rec["category"] if rec["category"] != "unknown" else "side"
            item = {
                "item_id": rec["item_id"],
                "restaurant_id": rec["restaurant_id"],
                "item_name": rec["item_name"],
                "category": cat,
                "price": int(rec["price"]),
                "is_veg": bool(rec["is_veg"]),
                "is_bestseller": bool(rec["is_bestseller"]),
            }
            self.item_map[item["item_id"]] = item
            self.menu_by_restaurant.setdefault(rec["restaurant_id"], []).append(item)

        for r in self.restaurants:
            r["item_count"] = len(self.menu_by_restaurant.get(r["restaurant_id"], []))

        # --- Users ---
        log.info("Loading users...")
        udf = pd.read_csv(DATA_DIR / "users.csv")
        for rec in udf.to_dict("records"):
            u = {
                "user_id": rec["user_id"],
                "city": rec["city"],
                "user_type": rec["user_type"],
                "is_veg": bool(rec["is_veg_preference"]),
                "avg_budget": round(float(rec["avg_budget"]), 0),
            }
            self.users.append(u)
            self.user_map[u["user_id"]] = u

        # --- Orders (sample for speed) ---
        log.info("Loading orders...")
        odf = pd.read_csv(DATA_DIR / "orders.csv", nrows=50000)
        self.orders = odf.to_dict("records")

        # --- Pre-compute PMI from order_items ---
        log.info("Computing PMI scores (this may take a moment)...")
        self._compute_pmi()

        # --- Item popularity ---
        log.info("Computing item popularity...")
        oidf = pd.read_csv(DATA_DIR / "order_items.csv", usecols=["item_id", "quantity"])
        pop = oidf.groupby("item_id")["quantity"].sum()
        max_pop = pop.max()
        self.item_popularity = (pop / max_pop).to_dict()

        # --- Stats ---
        self.stats = {
            "total_restaurants": len(self.restaurants),
            "total_users": len(self.users),
            "total_orders": len(self.orders),
            "total_menu_items": len(self.item_map),
            "cities": sorted(rdf["city"].unique().tolist()),
            "cuisines": sorted(rdf["cuisine_type"].unique().tolist()),
        }

        # --- Learned meal templates ---
        log.info("Learning meal templates from order data...")
        self.learned_meal_templates = self._learn_meal_templates()
        log.info(f"Learned meal templates for {len(self.learned_meal_templates)} cuisines")

        # --- City-specific cold-start defaults ---
        log.info("Computing city-specific cold-start defaults...")
        self.city_cold_start_defaults = self._compute_city_defaults()
        log.info(f"Computed cold-start defaults for {len(self.city_cold_start_defaults)} cities")

        elapsed = time.time() - start
        log.info(f"Data loaded in {elapsed:.1f}s — {len(self.restaurants)} restaurants, "
                 f"{len(self.item_map)} items, {len(self.user_map)} users, "
                 f"{len(self.pmi_scores)} PMI pairs")

    def _compute_pmi(self) -> None:
        """Compute PMI from order_items co-occurrence (vectorized for speed)."""
        oidf = pd.read_csv(DATA_DIR / "order_items.csv", usecols=["order_id", "item_id"])
        oidf = oidf.drop_duplicates(subset=["order_id", "item_id"])

        total_orders = oidf["order_id"].nunique()
        if total_orders == 0:
            return

        item_freq = oidf.groupby("item_id")["order_id"].nunique()

        order_sizes = oidf.groupby("order_id").size()
        valid_orders = order_sizes[order_sizes <= 8].index
        oidf_small = oidf[oidf["order_id"].isin(valid_orders)]

        log.info(f"Computing co-occurrences from {len(valid_orders)} orders...")
        merged = oidf_small.merge(oidf_small, on="order_id", suffixes=("_a", "_b"))
        pairs = merged[merged["item_id_a"] < merged["item_id_b"]]

        pair_counts = pairs.groupby(["item_id_a", "item_id_b"]).size().reset_index(name="count")
        pair_counts = pair_counts[pair_counts["count"] >= 3]

        for _, row in pair_counts.iterrows():
            a, b, count = row["item_id_a"], row["item_id_b"], row["count"]
            p_ab = count / total_orders
            p_a = item_freq.get(a, 0) / total_orders
            p_b = item_freq.get(b, 0) / total_orders
            if p_a > 0 and p_b > 0:
                pmi_val = math.log(p_ab / (p_a * p_b))
                if pmi_val > 0:
                    self.pmi_scores[f"{a}|{b}"] = round(pmi_val, 4)
                    self.pmi_scores[f"{b}|{a}"] = round(pmi_val, 4)

        log.info(f"Computed {len(self.pmi_scores)} positive PMI pairs")

    def _learn_meal_templates(self) -> dict[str, dict[str, Any]]:
        """Learn meal completeness templates from order data.

        Analyzes multi-item orders (2+ items) to discover which categories
        commonly co-occur per cuisine. Categories in >= 30% of multi-item
        orders are marked as 'required'. Falls back to hardcoded templates
        for cuisines with < 50 qualifying orders.
        """
        from features import MEAL_TEMPLATES as HARDCODED_TEMPLATES

        try:
            oi = pd.read_csv(DATA_DIR / "order_items.csv", usecols=["order_id", "category"])
            od = pd.read_csv(DATA_DIR / "orders.csv",
                             usecols=["order_id", "restaurant_id", "item_count"])
            rdf = pd.DataFrame(self.restaurants)

            # Only analyze multi-item orders (2+) — single-item orders
            # don't tell us about meal composition patterns
            multi_item_orders = od[od["item_count"] >= 2][["order_id", "restaurant_id"]]

            # Join order_items → orders → restaurants to get cuisine per order item
            merged = oi.merge(multi_item_orders, on="order_id", how="inner")
            merged = merged.merge(
                rdf[["restaurant_id", "cuisine_type"]], on="restaurant_id", how="inner",
            )

            # Exclude 'unknown' category
            merged = merged[merged["category"] != "unknown"]

            # For each (cuisine, order_id), collect distinct categories
            order_cats = (
                merged.groupby(["cuisine_type", "order_id"])["category"]
                .apply(lambda x: set(x))
                .reset_index(name="cats")
            )

            templates: dict[str, dict[str, Any]] = {}
            for cuisine, group in order_cats.groupby("cuisine_type"):
                n_orders = len(group)
                if n_orders < 50:
                    # Insufficient data — use hardcoded fallback
                    if cuisine in HARDCODED_TEMPLATES:
                        templates[cuisine] = dict(HARDCODED_TEMPLATES[cuisine])
                    continue

                # Count how many orders contain each category
                cat_counts: dict[str, int] = defaultdict(int)
                for cats in group["cats"]:
                    for cat in cats:
                        cat_counts[cat] += 1

                # Required = appears in >= 30% of multi-item orders
                required = sorted(
                    cat for cat, count in cat_counts.items()
                    if count / n_orders >= 0.30
                )
                if not required:
                    required = ["main"]  # Safety: at least one component

                templates[cuisine] = {"required": required, "total": len(required)}
                log.info(f"  {cuisine}: {required} (from {n_orders} orders)")

            # Backfill any hardcoded cuisines not in data
            for cuisine, tmpl in HARDCODED_TEMPLATES.items():
                if cuisine not in templates:
                    templates[cuisine] = dict(tmpl)

            return templates

        except Exception as e:
            log.warning(f"Failed to learn meal templates: {e}. Using hardcoded defaults.")
            from features import MEAL_TEMPLATES as HARDCODED_TEMPLATES
            return dict(HARDCODED_TEMPLATES)

    def _compute_city_defaults(self) -> dict[str, dict[str, float]]:
        """Compute per-city cold-start user defaults from users + orders data.

        For each city, aggregates user and order statistics to build
        city-specific cold-start profiles. Cities with < 100 orders
        are skipped (global default will be used).
        """
        from features import COLD_START_USER

        try:
            udf = pd.DataFrame(self.users)
            odf = pd.read_csv(DATA_DIR / "orders.csv")
            rdf = pd.DataFrame(self.restaurants)

            # Join orders → restaurants for cuisine info
            odf_cuisine = odf.merge(
                rdf[["restaurant_id", "cuisine_type"]], on="restaurant_id", how="left",
            )

            city_defaults: dict[str, dict[str, float]] = {}

            for city in udf["city"].unique():
                city_users = udf[udf["city"] == city]
                city_orders = odf[odf["city"] == city]
                city_orders_cuisine = odf_cuisine[odf_cuisine["city"] == city]

                if len(city_orders) < 100:
                    continue

                defaults = dict(COLD_START_USER)

                # --- From users ---
                if "is_veg" in city_users.columns:
                    defaults["veg_ratio"] = round(float(city_users["is_veg"].mean()), 3)
                if "avg_budget" in city_users.columns:
                    avg_budget = float(city_users["avg_budget"].mean())
                    defaults["monetary_avg"] = round(avg_budget, 1)
                    # Price sensitivity: normalized against global default (350)
                    defaults["price_sensitivity"] = round(350.0 / max(avg_budget, 1.0), 3)

                # --- From orders ---
                if "total_amount" in city_orders.columns:
                    defaults["monetary_avg"] = round(float(city_orders["total_amount"].mean()), 1)
                if "item_count" in city_orders.columns:
                    defaults["avg_items_per_order"] = round(float(city_orders["item_count"].mean()), 2)
                if "is_weekend" in city_orders.columns:
                    defaults["weekend_order_ratio"] = round(float(city_orders["is_weekend"].mean()), 3)

                # Mealtime fractions
                if "mealtime" in city_orders.columns:
                    mt_counts = city_orders["mealtime"].value_counts(normalize=True)
                    for mt in ["breakfast", "lunch", "snack", "dinner", "late_night"]:
                        defaults[f"{mt}_order_ratio"] = round(float(mt_counts.get(mt, 0.0)), 3)

                # Cuisine preferences
                if "cuisine_type" in city_orders_cuisine.columns:
                    cuisine_counts = city_orders_cuisine["cuisine_type"].value_counts(normalize=True)
                    cuisine_key_map = {
                        "North Indian": "cuisine_pref_north_indian",
                        "South Indian": "cuisine_pref_south_indian",
                        "Chinese": "cuisine_pref_chinese",
                        "Biryani": "cuisine_pref_biryani",
                        "Street Food": "cuisine_pref_street_food",
                        "Continental": "cuisine_pref_continental",
                        "Maharashtrian": "cuisine_pref_maharashtrian",
                    }
                    for cuisine_name, key in cuisine_key_map.items():
                        defaults[key] = round(float(cuisine_counts.get(cuisine_name, 0.0)), 4)

                city_defaults[city] = defaults
                log.info(f"  {city}: budget={defaults['monetary_avg']}, "
                         f"veg={defaults['veg_ratio']}, "
                         f"top_cuisine={max(((k, v) for k, v in defaults.items() if k.startswith('cuisine_pref_')), key=lambda x: x[1])[0].replace('cuisine_pref_', '')}")

            return city_defaults

        except Exception as e:
            log.warning(f"Failed to compute city defaults: {e}. Using global cold-start.")
            return {}

    def get_meal_template(self, cuisine: str) -> dict[str, Any]:
        """Get meal template for a cuisine, with hardcoded fallback."""
        return self.learned_meal_templates.get(
            cuisine,
            {"required": ["main", "side", "drink"], "total": 3},
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="CSAO Demo API", description="Cart Super Add-On Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserSessionTracker:
    """In-memory per-user session state for fatigue prevention.

    Tracks frequency capping, impression decay, and session novelty
    to avoid showing the same recommendations repeatedly.
    """

    def __init__(self) -> None:
        self.freq_cap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.impression_penalty: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.prev_shown: dict[str, set[str]] = defaultdict(set)
        self.fire_count: dict[str, int] = defaultdict(int)
        self.last_active: dict[str, float] = {}

    def _maybe_reset_daily(self, user_id: str) -> None:
        """Reset fatigue state if last activity was on a different day."""
        now = time.time()
        last = self.last_active.get(user_id, 0)
        if last == 0 or date.fromtimestamp(last) != date.fromtimestamp(now):
            self.freq_cap[user_id].clear()
            self.impression_penalty[user_id].clear()
            self.prev_shown[user_id].clear()
            self.fire_count[user_id] = 0
        self.last_active[user_id] = now

    def get_session_state(self, user_id: str) -> dict[str, Any]:
        """Return current fatigue state for this user."""
        self._maybe_reset_daily(user_id)
        return {
            "freq_cap": dict(self.freq_cap[user_id]),
            "impression_penalty": dict(self.impression_penalty[user_id]),
            "prev_session_items": set(self.prev_shown[user_id]),
            "fire_count": self.fire_count[user_id],
        }

    def record_impressions(self, user_id: str, shown_item_ids: list[str]) -> None:
        """Called after each /api/recommendations — update freq_cap + prev_shown."""
        self._maybe_reset_daily(user_id)
        for item_id in shown_item_ids:
            self.freq_cap[user_id][item_id] += 1
        self.prev_shown[user_id] = set(shown_item_ids)
        self.fire_count[user_id] += 1

    def record_feedback(self, user_id: str, shown_ids: list[str], accepted_ids: set[str]) -> None:
        """Called on /api/orders — apply impression decay for shown-but-not-accepted."""
        for item_id in shown_ids:
            if item_id not in accepted_ids:
                self.impression_penalty[user_id][item_id] += 0.05

    def reset_session(self, user_id: str) -> None:
        """Reset on restaurant switch or explicit session end."""
        self.prev_shown[user_id].clear()
        self.fire_count[user_id] = 0


store = DataStore()
ml_manager = MLModelManager()
session_tracker = UserSessionTracker()

# Lazy import to avoid import errors if not installed
feature_store = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load data + ML models + feature store at server startup."""
    global feature_store

    store.load()

    # Load ML models
    ml_manager.load()

    # Load feature store for real-time feature engineering
    try:
        from features import FeatureStore
        feature_store = FeatureStore()
        feature_store.load()
    except Exception as e:
        log.warning(f"Feature store load failed: {e}")
        feature_store = None


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class CartItemModel(BaseModel):
    item_id: str
    name: str
    price: float
    category: str


class RecommendationRequest(BaseModel):
    user_id: str = "U000000"
    restaurant_id: str
    cart_items: list[CartItemModel]
    meal_period: str = "dinner"
    city: str = "Mumbai"


class RecShownItem(BaseModel):
    item_id: str
    position: int
    score: float


class OrderRequest(BaseModel):
    user_id: str
    restaurant_id: str
    items: list[dict[str, Any]]
    total: float
    meal_period: str = "dinner"
    city: str = "Mumbai"
    recs_shown: list[RecShownItem] = []
    recs_accepted: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/restaurants")
def get_restaurants(
    city: str | None = None,
    cuisine: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """List restaurants with optional filtering."""
    results = store.restaurants
    if city:
        results = [r for r in results if r["city"] == city]
    if cuisine:
        results = [r for r in results if r["cuisine_type"] == cuisine]
    if search:
        q = search.lower()
        results = [r for r in results if q in r["name"].lower() or q in r["cuisine_type"].lower()]
    results = sorted(results, key=lambda r: r["avg_rating"], reverse=True)
    return results


@app.get("/api/restaurants/{restaurant_id}")
def get_restaurant(restaurant_id: str) -> dict[str, Any]:
    """Get restaurant detail with full menu grouped by category."""
    r = store.restaurant_map.get(restaurant_id)
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    menu = store.menu_by_restaurant.get(restaurant_id, [])
    grouped: dict[str, list[dict]] = {}
    for item in menu:
        cat = item["category"]
        grouped.setdefault(cat, []).append(item)

    cat_order = ["main", "bread", "side", "drink", "dessert"]
    sorted_menu: dict[str, list[dict]] = {}
    for cat in cat_order:
        if cat in grouped:
            sorted_menu[cat] = sorted(grouped[cat], key=lambda x: -x["is_bestseller"])
    for cat in grouped:
        if cat not in sorted_menu:
            sorted_menu[cat] = grouped[cat]

    return {**r, "menu": sorted_menu}


@app.get("/api/users/{user_id}")
def get_user(user_id: str) -> dict[str, Any]:
    """Get user profile."""
    u = store.user_map.get(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u


# Feature-to-reason mapping for model-driven explanations.
# Each entry: (feature_name, reason_template, is_binary, is_item_specific).
# Templates can use {best_cart_item} placeholder for PMI-based reasons.
# Item-specific features (vary per candidate) are preferred over context
# features (same for all candidates in a request) to avoid identical reasons.
_REASON_FEATURES_ITEM: list[tuple[str, str, bool]] = [
    # These vary across candidates — preferred
    ("cross_max_pmi", "Goes well with {best_cart_item}", False),
    ("cross_fills_missing_component", "Completes your meal", True),
    ("cross_embedding_similarity", "Pairs well with your cart", False),
    ("item_is_bestseller", "Bestseller here", True),
    ("cross_fills_gap_x_user_pref", "Matches your taste", False),
    ("cross_bestseller_x_pmi", "Top pick for your combo", False),
    ("cross_veg_match", "Matches your diet", True),
]

_REASON_FEATURES_CONTEXT: list[tuple[str, str, bool]] = [
    # These are similar across candidates — used as fallback
    ("cross_fits_budget", "Fits your budget", True),
    ("cross_price_vs_headroom", "Great value add", False),
    ("cross_user_category_pref", "Matches your taste", False),
    ("csao_historical_accept_rate", "Based on your history", False),
]


def _generate_reason_ml(
    feature_vec: dict[str, float],
    shap_importance: dict[str, float],
    cart_items: list[CartItemModel],
    pmi_scores: dict[str, float],
    item_id: str,
) -> str:
    """Generate a reason driven by SHAP importance × feature value.

    Prefers item-specific features (that differentiate candidates) over
    context features (that are the same for all items in a request).
    """
    # Find best cart item by PMI (needed for PMI-based reason template)
    best_cart_name = ""
    best_pmi = 0.0
    for ci in cart_items:
        pmi = pmi_scores.get(f"{ci.item_id}|{item_id}", 0.0)
        if pmi > best_pmi:
            best_pmi = pmi
            best_cart_name = ci.name

    def _best_from(features: list[tuple[str, str, bool]]) -> tuple[str, float]:
        best = ("", -1.0)
        for feat_name, template, is_binary in features:
            value = feature_vec.get(feat_name, 0.0)
            if is_binary and value < 0.5:
                continue
            if not is_binary and value <= 0.0:
                continue
            importance = shap_importance.get(feat_name, 1e-4)
            contribution = value * importance
            if contribution > best[1]:
                best = (template, contribution)
        return best

    # Try item-specific features first
    reason, score = _best_from(_REASON_FEATURES_ITEM)
    if score <= 0:
        # Fall back to context features
        reason, score = _best_from(_REASON_FEATURES_CONTEXT)
    if not reason:
        reason = "Recommended for you"

    return reason.format(best_cart_item=best_cart_name or "your cart")


def _generate_reason(
    item: dict[str, Any],
    cart_items: list[CartItemModel],
    pmi_scores: dict[str, float],
    missing_components: list[str],
    item_popularity: dict[str, float],
) -> str:
    """Heuristic reason generation (used only by the fallback path)."""
    iid = item["item_id"]

    best_pmi = 0.0
    best_cart_name = ""
    for ci in cart_items:
        pmi = pmi_scores.get(f"{ci.item_id}|{iid}", 0.0)
        if pmi > best_pmi:
            best_pmi = pmi
            best_cart_name = ci.name

    if best_pmi > 0 and best_cart_name:
        return f"Goes well with {best_cart_name}"
    if item["category"] in missing_components:
        return "Completes your meal"
    if item.get("is_bestseller", False):
        return "Bestseller here"
    return "Recommended for you"


def _heuristic_recommendations(
    req: RecommendationRequest,
    menu: list[dict[str, Any]],
    user: dict[str, Any],
    cuisine: str,
    missing_components: list[str],
) -> list[dict[str, Any]]:
    """Fallback: PMI + meal completeness + popularity heuristics (no ML)."""
    cart_ids = {item.item_id for item in req.cart_items}
    is_veg = user.get("is_veg", False)

    candidates: list[dict[str, Any]] = []
    for item in menu:
        iid = item["item_id"]
        if iid in cart_ids:
            continue
        if is_veg and not item["is_veg"]:
            continue

        score = 0.0
        pmi_total = 0.0
        for cart_item in req.cart_items:
            key = f"{cart_item.item_id}|{iid}"
            pmi_total += store.pmi_scores.get(key, 0.0)
        if pmi_total > 0:
            score += min(pmi_total * 0.3, 0.6)

        if item["category"] in missing_components:
            score += 0.25

        pop = store.item_popularity.get(iid, 0.0)
        score += pop * 0.15
        if item["is_bestseller"]:
            score += 0.05
        if score < 0.01:
            score = pop * 0.1 + 0.05

        reason = _generate_reason(item, req.cart_items, store.pmi_scores,
                                  missing_components, store.item_popularity)

        candidates.append({
            "item_id": iid, "item_name": item["item_name"],
            "price": item["price"], "category": item["category"],
            "is_veg": item["is_veg"], "score": round(score, 4),
            "reason": reason,
        })

    candidates.sort(key=lambda x: -x["score"])

    # Diversity filter + fatigue prevention
    fatigue_state = session_tracker.get_session_state(req.user_id)
    fc = fatigue_state["freq_cap"]
    final: list[dict[str, Any]] = []
    cat_counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        # Frequency capping
        if fc.get(c["item_id"], 0) >= 3:
            continue
        cat = c["category"]
        if cat_counts[cat] >= 2:
            continue
        final.append(c)
        cat_counts[cat] += 1
        if len(final) >= 10:
            break

    if final:
        max_score = max(c["score"] for c in final)
        if max_score > 0:
            for c in final:
                c["score"] = round(c["score"] / max_score, 2)

    return final[:10]


def _lgbm_only_fallback(
    req: RecommendationRequest,
    candidate_items: list[dict[str, Any]],
    cuisine: str,
    missing_components: list[str],
    pipeline_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tier 2 fallback: LightGBM L1 scoring + MMR (no Two-Tower/SASRec/DCN-v2)."""
    from features import build_full_feature_vector, build_feature_matrix
    from models import mmr_rerank

    cart_item_dicts = [
        {"item_id": ci.item_id, "name": ci.name, "price": ci.price, "category": ci.category}
        for ci in req.cart_items
    ]

    # Build feature vectors (without Two-Tower score or SASRec embeddings)
    feature_vectors: list[dict[str, float]] = []
    for pos, item in enumerate(candidate_items, 1):
        vec = build_full_feature_vector(
            user_id=req.user_id,
            candidate_item_id=item["item_id"],
            restaurant_id=req.restaurant_id,
            cart_items=cart_item_dicts,
            meal_period=req.meal_period,
            city=req.city,
            feature_store=feature_store,
            pmi_scores=store.pmi_scores,
            position_in_rail=pos,
            meal_templates=store.learned_meal_templates,
            city_defaults=store.city_cold_start_defaults,
        )
        feature_vectors.append(vec)

    l1_matrix = build_feature_matrix(feature_vectors, ml_manager.l1_feature_cols)
    l1_scores = ml_manager.get_lgbm_scores(l1_matrix)
    pipeline_info["models_used"] = ["LightGBM-L1"]
    pipeline_info["l1_scored"] = len(candidate_items)

    top_indices = np.argsort(-l1_scores)[:30]

    scored: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices):
        item = candidate_items[idx]
        cand_emb = feature_store.get_item_embedding(item["item_id"])
        scored.append({
            "item_id": item["item_id"],
            "item_name": item["item_name"],
            "price": item["price"],
            "category": item["category"],
            "is_veg": item["is_veg"],
            "score": float(l1_scores[idx]),
            "embedding": cand_emb,
            "reason": _generate_reason_ml(
                feature_vectors[idx], ml_manager.shap_importance,
                req.cart_items, store.pmi_scores, item["item_id"],
            ),
        })
    # Missing-component boost (same logic as full cascade)
    if missing_components:
        missing_set = set(missing_components)
        sv = [c["score"] for c in scored]
        sr = max(sv) - min(sv) if len(sv) > 1 else 1.0
        bm = max(sr * 0.3, 0.01)
        for c in scored:
            if c["category"] in missing_set:
                item_boost = bm
                best_pmi = 0.0
                for ci in req.cart_items:
                    pmi_val = store.pmi_scores.get(f"{ci.item_id}|{c['item_id']}", 0.0)
                    best_pmi = max(best_pmi, pmi_val)
                if best_pmi > 0:
                    item_boost *= 0.5 + min(best_pmi / 10.0, 1.0)
                c["score"] += item_boost

    scored.sort(key=lambda x: -x["score"])

    embeddings = np.array([c.pop("embedding") for c in scored], dtype=np.float32)
    fatigue_state = session_tracker.get_session_state(req.user_id)
    final = mmr_rerank(
        scored, embeddings=embeddings, lambda_param=0.7, top_k=8, max_per_category=2,
        freq_cap=fatigue_state["freq_cap"],
        impression_penalty=fatigue_state["impression_penalty"],
        prev_session_items=fatigue_state["prev_session_items"] if fatigue_state["fire_count"] > 0 else None,
    )
    pipeline_info["models_used"].append("MMR")

    # Normalize
    if final:
        max_s = max(c["score"] for c in final)
        min_s = min(c["score"] for c in final)
        rng = max_s - min_s
        if rng > 1e-9:
            for c in final:
                c["score"] = round((c["score"] - min_s) / rng, 2)
    return final


def _pmi_lookup_fallback(
    req: RecommendationRequest,
    menu: list[dict[str, Any]],
    missing_components: list[str],
) -> list[dict[str, Any]]:
    """Tier 3 fallback: pure PMI co-occurrence scoring (no ML models)."""
    cart_ids = {item.item_id for item in req.cart_items}
    fc = session_tracker.get_session_state(req.user_id)["freq_cap"]
    candidates: list[dict[str, Any]] = []

    for item in menu:
        iid = item["item_id"]
        if iid in cart_ids:
            continue
        if fc.get(iid, 0) >= 3:
            continue

        pmi_total = 0.0
        best_pmi = 0.0
        best_name = ""
        for ci in req.cart_items:
            pmi = store.pmi_scores.get(f"{ci.item_id}|{iid}", 0.0)
            pmi_total += pmi
            if pmi > best_pmi:
                best_pmi = pmi
                best_name = ci.name

        if pmi_total <= 0:
            continue  # PMI-only: skip items with no co-occurrence signal

        reason = f"Goes well with {best_name}" if best_name else "Frequently ordered together"
        candidates.append({
            "item_id": iid, "item_name": item["item_name"],
            "price": item["price"], "category": item["category"],
            "is_veg": item["is_veg"], "score": round(pmi_total, 4),
            "reason": reason,
        })

    candidates.sort(key=lambda x: -x["score"])

    # Normalize
    if candidates:
        max_s = candidates[0]["score"]
        if max_s > 0:
            for c in candidates:
                c["score"] = round(c["score"] / max_s, 2)

    return candidates[:10]


def _popularity_fallback(
    req: RecommendationRequest,
    menu: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tier 4 fallback: pure item popularity (absolute last resort)."""
    cart_ids = {item.item_id for item in req.cart_items}
    fc = session_tracker.get_session_state(req.user_id)["freq_cap"]
    candidates: list[dict[str, Any]] = []

    for item in menu:
        iid = item["item_id"]
        if iid in cart_ids:
            continue
        if fc.get(iid, 0) >= 3:
            continue

        pop = store.item_popularity.get(iid, 0.0)
        candidates.append({
            "item_id": iid, "item_name": item["item_name"],
            "price": item["price"], "category": item["category"],
            "is_veg": item["is_veg"],
            "score": round(pop, 4),
            "reason": "Popular at this restaurant",
        })

    candidates.sort(key=lambda x: -x["score"])

    if candidates:
        max_s = candidates[0]["score"]
        if max_s > 0:
            for c in candidates:
                c["score"] = round(c["score"] / max_s, 2)

    return candidates[:10]


@app.post("/api/recommendations")
def get_recommendations(req: RecommendationRequest) -> dict[str, Any]:
    """Generate CSAO recommendations — full ML cascade with fallback chain.

    Pipeline: Two-Tower → LightGBM L1 → SASRec → DCN-v2 L2 → MMR
    Fallback: full_pipeline → lgbm_only → pmi_lookup → popularity
    """
    start_time = time.time()

    menu = store.menu_by_restaurant.get(req.restaurant_id, [])
    if not menu:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    user = store.user_map.get(req.user_id, {"is_veg": False})
    is_veg = user.get("is_veg", False)
    cart_ids = {item.item_id for item in req.cart_items}
    cart_categories = {item.category for item in req.cart_items}

    restaurant = store.restaurant_map.get(req.restaurant_id, {})
    cuisine = restaurant.get("cuisine_type", "North Indian")
    template = store.get_meal_template(cuisine)
    required = template["required"]
    missing_components = [c for c in required if c not in cart_categories]

    # Cart completeness
    filled = len([c for c in required if c in cart_categories])
    completeness = round(filled / template["total"], 2) if template["total"] > 0 else 0.0

    # Filter candidate items (exclude cart items, apply veg filter)
    candidate_items = []
    for item in menu:
        if item["item_id"] in cart_ids:
            continue
        if is_veg and not item["is_veg"]:
            continue
        candidate_items.append(item)

    if not candidate_items:
        latency_ms = round((time.time() - start_time) * 1000, 1)
        return {
            "recommendations": [],
            "cart_completeness": completeness,
            "missing_components": missing_components,
            "latency_ms": latency_ms,
            "pipeline": {"models_used": [], "fallback": "no_candidates"},
        }

    # --- Fallback chain: full_pipeline → lgbm_only → pmi_lookup → popularity ---
    pipeline_info: dict[str, Any] = {
        "candidates_retrieved": len(candidate_items),
        "models_used": [],
    }

    def _make_response(recs: list[dict[str, Any]]) -> dict[str, Any]:
        latency_ms = round((time.time() - start_time) * 1000, 1)
        pipeline_info["final_after_mmr"] = len(recs)

        # Track impressions for fatigue prevention
        shown_ids = [r["item_id"] for r in recs[:10]]
        session_tracker.record_impressions(req.user_id, shown_ids)

        return {
            "recommendations": recs[:10],
            "cart_completeness": completeness,
            "missing_components": missing_components,
            "latency_ms": latency_ms,
            "pipeline": pipeline_info,
        }

    # Tier 1: Full ML cascade (SASRec → Two-Tower → LightGBM → DCN-v2 → MMR)
    if ml_manager.models_loaded and feature_store is not None:
        try:
            final_recs = _ml_cascade(
                req, candidate_items, user, cuisine, missing_components,
                pipeline_info,
            )
            if final_recs:
                return _make_response(final_recs)
        except Exception as e:
            log.warning(f"Full ML cascade failed: {e}")
            pipeline_info["error"] = str(e)

    # Tier 2: LightGBM-only (skip Two-Tower/SASRec/DCN-v2, use L1 features + MMR)
    if ml_manager.lgbm_model is not None and feature_store is not None:
        try:
            final_recs = _lgbm_only_fallback(
                req, candidate_items, cuisine, missing_components, pipeline_info,
            )
            if final_recs:
                pipeline_info["fallback"] = "lgbm_only"
                return _make_response(final_recs)
        except Exception as e:
            log.warning(f"LightGBM-only fallback failed: {e}")

    # Tier 3: PMI lookup (no ML models, pure co-occurrence scoring)
    pmi_recs = _pmi_lookup_fallback(req, menu, missing_components)
    if pmi_recs:
        pipeline_info["models_used"] = ["PMI-Lookup"]
        pipeline_info["fallback"] = "pmi_lookup"
        return _make_response(pmi_recs)

    # Tier 4: Popularity (absolute last resort)
    pop_recs = _popularity_fallback(req, menu)
    pipeline_info["models_used"] = ["Popularity"]
    pipeline_info["fallback"] = "popularity"
    return _make_response(pop_recs)


def _ml_cascade(
    req: RecommendationRequest,
    candidate_items: list[dict[str, Any]],
    user: dict[str, Any],
    cuisine: str,
    missing_components: list[str],
    pipeline_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the full ML cascade: SASRec → Two-Tower → LightGBM L1 → DCN-v2 → MMR."""
    from features import build_full_feature_vector, build_feature_matrix
    from models import mmr_rerank

    cart_item_dicts = [
        {"item_id": ci.item_id, "name": ci.name, "price": ci.price, "category": ci.category}
        for ci in req.cart_items
    ]
    cart_item_ids = [ci.item_id for ci in req.cart_items]
    candidate_ids = [item["item_id"] for item in candidate_items]

    # ─── Step 1: SASRec sequential context (RUNS FIRST) ─────────────
    # SASRec encodes the cart item sequence into a 16-dim context embedding
    # that enriches the Query Tower for sequence-aware retrieval.
    seq_emb = np.zeros(16, dtype=np.float32)
    if ml_manager.sasrec_model is not None:
        seq_emb = ml_manager.get_sasrec_embeddings(cart_item_ids)
        pipeline_info["models_used"].append("SASRec")
        log.info(f"  SASRec context embedding computed (16-dim)")

    # ─── Step 2: Two-Tower scoring (with sequential context) ─────────
    tt_scores: dict[str, float] = {}
    if ml_manager.two_tower_model is not None:
        user_feats = feature_store.get_user_features(
            req.user_id, city=req.city, city_defaults=store.city_cold_start_defaults,
        )
        tt_scores = ml_manager.get_two_tower_scores(
            user_feats, cart_item_dicts, candidate_ids, feature_store,
            seq_emb=seq_emb,
        )
        if tt_scores:
            pipeline_info["models_used"].append("Two-Tower+FAISS")
            log.info(f"  Two-Tower scored {len(tt_scores)} candidates (seq-aware)")

    # ─── Step 3: Build feature vectors + LightGBM L1 ────────────────
    feature_vectors: list[dict[str, float]] = []
    for pos, item in enumerate(candidate_items, 1):
        vec = build_full_feature_vector(
            user_id=req.user_id,
            candidate_item_id=item["item_id"],
            restaurant_id=req.restaurant_id,
            cart_items=cart_item_dicts,
            meal_period=req.meal_period,
            city=req.city,
            feature_store=feature_store,
            pmi_scores=store.pmi_scores,
            position_in_rail=pos,
            meal_templates=store.learned_meal_templates,
            city_defaults=store.city_cold_start_defaults,
        )
        # Add Two-Tower score as feature
        vec["two_tower_score"] = tt_scores.get(item["item_id"], 0.0)
        # Add sequential embeddings to ALL candidates (not just top-30)
        for d in range(16):
            vec[f"sequential_emb_{d}"] = float(seq_emb[d])
        feature_vectors.append(vec)

    # LightGBM L1 scoring (now has two_tower_score + sequential_emb)
    l1_scores = np.zeros(len(candidate_items))
    if ml_manager.lgbm_model is not None and ml_manager.l1_feature_cols:
        l1_matrix = build_feature_matrix(feature_vectors, ml_manager.l1_feature_cols)
        l1_scores = ml_manager.get_lgbm_scores(l1_matrix)
        pipeline_info["models_used"].append("LightGBM-L1")
        log.info(f"  LightGBM L1 scored {len(l1_scores)} candidates")

    # Add l1_score to feature vectors
    for i, score in enumerate(l1_scores):
        feature_vectors[i]["l1_score"] = float(score)

    # Select top 30 by L1 score
    top_30_indices = np.argsort(-l1_scores)[:30]
    pipeline_info["l1_scored"] = len(candidate_items)
    pipeline_info["l1_top30"] = len(top_30_indices)

    # ─── Step 4: DCN-v2 L2 scoring on top 30 ────────────────────────
    l2_scores = np.zeros(len(candidate_items))
    if ml_manager.dcnv2_model is not None and ml_manager.l2_feature_cols:
        # Build L2 feature matrix for top 30 only
        top30_vectors = [feature_vectors[i] for i in top_30_indices]
        l2_matrix = build_feature_matrix(top30_vectors, ml_manager.l2_feature_cols)
        top30_l2_scores = ml_manager.get_dcnv2_scores(l2_matrix)

        for rank, idx in enumerate(top_30_indices):
            l2_scores[idx] = top30_l2_scores[rank]

        pipeline_info["models_used"].append("DCN-v2")
        pipeline_info["l2_scored"] = len(top_30_indices)
        log.info(f"  DCN-v2 L2 scored {len(top_30_indices)} candidates")
    else:
        # Fallback: use L1 scores as L2
        l2_scores = l1_scores.copy()

    # ─── Step 5: ML cascade tiebreaker + MMR re-ranking ─────────────
    # Use L2 (DCN-v2) as primary score. When L2 produces no variance
    # (common for small candidate sets), fall back to L1 (LightGBM)
    # which already learned completeness signals (cross_fills_missing_component,
    # cart_completeness_score, etc.) through its feature set.
    top30_l2 = np.array([l2_scores[i] for i in top_30_indices])
    l2_range = top30_l2.max() - top30_l2.min() if len(top30_l2) > 0 else 0.0

    if l2_range < 1e-9:
        # L2 flat — fall back to L1 scores (which incorporate completeness features)
        ranking_scores = np.array([l1_scores[i] for i in top_30_indices])
        log.info("  L2 scores flat — using L1 scores for ranking")
    else:
        ranking_scores = top30_l2

    # Build candidate list for MMR
    scored_candidates: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_30_indices):
        item = candidate_items[idx]
        cand_emb = feature_store.get_item_embedding(item["item_id"])

        scored_candidates.append({
            "item_id": item["item_id"],
            "item_name": item["item_name"],
            "price": item["price"],
            "category": item["category"],
            "is_veg": item["is_veg"],
            "score": float(np.nan_to_num(ranking_scores[rank], nan=0.0)),
            "embedding": cand_emb,
            "reason": _generate_reason_ml(
                feature_vectors[idx], ml_manager.shap_importance,
                req.cart_items, store.pmi_scores, item["item_id"],
            ),
        })

    # ─── Missing-component boost ──────────────────────────────────────
    # The ML model's top features are user-level (same for all candidates),
    # so item-level signals like PMI and fills-missing-component get drowned
    # out. This boost ensures items that complete the meal get a meaningful
    # score lift proportional to their PMI with the cart.
    if missing_components:
        missing_set = set(missing_components)
        score_values = [c["score"] for c in scored_candidates]
        score_range = max(score_values) - min(score_values) if len(score_values) > 1 else 1.0
        boost_magnitude = max(score_range * 0.3, 0.01)  # 30% of score range

        for c in scored_candidates:
            if c["category"] in missing_set:
                # Base boost for filling a gap
                item_boost = boost_magnitude
                # Scale by PMI: items with stronger cart co-purchase get more boost
                best_pmi = 0.0
                for ci in req.cart_items:
                    key = f"{ci.item_id}|{c['item_id']}"
                    pmi_val = store.pmi_scores.get(key, 0.0)
                    best_pmi = max(best_pmi, pmi_val)
                if best_pmi > 0:
                    # PMI typically ranges 2-10; normalize to 0.5-1.5x multiplier
                    pmi_mult = 0.5 + min(best_pmi / 10.0, 1.0)
                    item_boost *= pmi_mult
                c["score"] += item_boost

    # Sort by score before MMR
    scored_candidates.sort(key=lambda x: -x["score"])

    # Build embedding matrix for MMR similarity
    embeddings = np.array([c.pop("embedding") for c in scored_candidates], dtype=np.float32)

    # Get fatigue state for this user
    fatigue_state = session_tracker.get_session_state(req.user_id)
    fire_count = fatigue_state["fire_count"]

    final_recs = mmr_rerank(
        scored_candidates,
        embeddings=embeddings,
        lambda_param=0.7,
        top_k=8,
        max_per_category=2,
        freq_cap=fatigue_state["freq_cap"],
        impression_penalty=fatigue_state["impression_penalty"],
        prev_session_items=fatigue_state["prev_session_items"] if fire_count > 0 else None,
    )

    pipeline_info["models_used"].append("MMR")
    log.info(f"  Fatigue: fire={fire_count}, freq_capped={sum(1 for v in fatigue_state['freq_cap'].values() if v >= 3)}, "
             f"penalized={sum(1 for v in fatigue_state['impression_penalty'].values() if v > 0)}")

    # Normalize scores to 0-1 for display.
    # Use position-based scoring (MMR selection order IS the ranking)
    # so scores always decrease down the rail. Position 1 = 1.0, last = 0.5.
    if final_recs:
        n = len(final_recs)
        for i, c in enumerate(final_recs):
            # Linear decay from 1.0 (position 0) to 0.5 (last position)
            c["score"] = round(1.0 - 0.5 * (i / max(n - 1, 1)), 2)

    return final_recs


@app.post("/api/orders")
def create_order(req: OrderRequest) -> dict[str, Any]:
    """Place a new order and persist to training CSVs."""
    now = datetime.now()
    order_id = f"O{len(store.orders):07d}"

    # --- 1. Append to orders.csv ---
    is_weekend = now.weekday() >= 5
    order_row = {
        "order_id": order_id,
        "user_id": req.user_id,
        "restaurant_id": req.restaurant_id,
        "order_date": now.strftime("%Y-%m-%d"),
        "order_hour": now.hour,
        "order_minute": now.minute,
        "mealtime": req.meal_period,
        "city": req.city,
        "total_amount": req.total,
        "item_count": sum(it.get("quantity", 1) for it in req.items),
        "is_weekend": is_weekend,
    }
    orders_path = DATA_DIR / "orders.csv"
    with open(orders_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(order_row.keys()))
        writer.writerow(order_row)

    # --- 2. Append to order_items.csv ---
    recs_accepted_set = set(req.recs_accepted)
    order_items_path = DATA_DIR / "order_items.csv"
    with open(order_items_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["order_id", "item_id", "item_name", "category", "price",
                      "quantity", "is_addon", "position_in_cart"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for pos, it in enumerate(req.items, start=1):
            is_addon = (it.get("source") == "recommendation") or (it["item_id"] in recs_accepted_set)
            writer.writerow({
                "order_id": order_id,
                "item_id": it["item_id"],
                "item_name": it.get("name", ""),
                "category": it.get("category", ""),
                "price": it.get("price", 0),
                "quantity": it.get("quantity", 1),
                "is_addon": is_addon,
                "position_in_cart": pos,
            })

    # --- 3. Append to csao_interactions.csv ---
    # Each recommendation shown is an interaction row.
    # add_to_cart = True if item_id is in recs_accepted.
    interactions_path = DATA_DIR / "csao_interactions.csv"
    # Get current max interaction_id
    existing_count = 0
    try:
        with open(interactions_path, "r", encoding="utf-8") as f:
            for _ in f:
                existing_count += 1
        existing_count -= 1  # subtract header
    except FileNotFoundError:
        existing_count = 0

    cart_state = "|".join(it["item_id"] for it in req.items if it.get("source") != "recommendation")
    cart_size = sum(1 for it in req.items if it.get("source") != "recommendation")

    if req.recs_shown:
        with open(interactions_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["interaction_id", "order_id", "user_id", "restaurant_id",
                          "item_id", "position", "impression", "click", "add_to_cart",
                          "cart_state", "cart_size", "mealtime", "city", "is_weekend",
                          "fire_sequence", "order_date", "order_hour"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for i, rec in enumerate(req.recs_shown):
                int_id = f"INT{existing_count + i:08d}"
                accepted = rec.item_id in recs_accepted_set
                writer.writerow({
                    "interaction_id": int_id,
                    "order_id": order_id,
                    "user_id": req.user_id,
                    "restaurant_id": req.restaurant_id,
                    "item_id": rec.item_id,
                    "position": rec.position,
                    "impression": True,
                    "click": accepted,
                    "add_to_cart": accepted,
                    "cart_state": cart_state,
                    "cart_size": cart_size,
                    "mealtime": req.meal_period,
                    "city": req.city,
                    "is_weekend": is_weekend,
                    "fire_sequence": session_tracker.fire_count.get(req.user_id, 1),
                    "order_date": now.strftime("%Y-%m-%d"),
                    "order_hour": now.hour,
                })

    # Keep in-memory store updated
    store.orders.append(order_row)

    # Update fatigue state with accept/reject feedback
    if req.recs_shown:
        shown_ids = [r.item_id for r in req.recs_shown]
        accepted_set = set(req.recs_accepted)
        session_tracker.record_feedback(req.user_id, shown_ids, accepted_set)

    log.info(f"Order {order_id} persisted: {len(req.items)} items, "
             f"{len(req.recs_shown)} recs shown, {len(req.recs_accepted)} accepted")

    return {"order_id": order_id, "status": "confirmed", "message": "Order placed successfully!"}


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    """Summary stats for the dashboard."""
    return store.stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
