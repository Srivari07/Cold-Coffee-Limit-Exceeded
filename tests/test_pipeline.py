"""
End-to-end validation tests for the CSAO Rail Recommendation System pipeline.

Validates all outputs from the 7-module pipeline without re-running it.
Run with: pytest tests/test_pipeline.py -v

Tests organized by module:
    TestDataGeneration       - 01_data_generator.py outputs
    TestFeatureEngineering   - 02_feature_engineering.py outputs
    TestBaselineModels       - 03_baseline_models.py outputs
    TestModelTraining        - 04_model_training.py outputs
    TestEvaluation           - 05_evaluation.py outputs
    TestSystemDesign         - 06_system_design.py outputs
    TestBusinessImpact       - 07_business_impact.py outputs
    TestDocumentation        - docs/ completeness
    TestCrossModuleConsistency - cross-module alignment
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths — tests/ is one level below project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(name: str) -> dict[str, Any]:
    with open(OUTPUTS_DIR / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (loaded once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def users_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "users.csv")


@pytest.fixture(scope="module")
def restaurants_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "restaurants.csv")


@pytest.fixture(scope="module")
def menu_items_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "menu_items.csv")


@pytest.fixture(scope="module")
def orders_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "orders.csv")


@pytest.fixture(scope="module")
def order_items_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "order_items.csv")


@pytest.fixture(scope="module")
def interactions_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "csao_interactions.csv")


@pytest.fixture(scope="module")
def features_train_meta() -> tuple[int, list[str]]:
    """Return (row_count, column_names) without loading full data."""
    df = pd.read_csv(OUTPUTS_DIR / "features_train.csv", nrows=0)
    nrows = sum(1 for _ in open(OUTPUTS_DIR / "features_train.csv")) - 1
    return nrows, list(df.columns)


@pytest.fixture(scope="module")
def features_val_meta() -> tuple[int, list[str]]:
    df = pd.read_csv(OUTPUTS_DIR / "features_val.csv", nrows=0)
    nrows = sum(1 for _ in open(OUTPUTS_DIR / "features_val.csv")) - 1
    return nrows, list(df.columns)


@pytest.fixture(scope="module")
def features_test_meta() -> tuple[int, list[str]]:
    df = pd.read_csv(OUTPUTS_DIR / "features_test.csv", nrows=0)
    nrows = sum(1 for _ in open(OUTPUTS_DIR / "features_test.csv")) - 1
    return nrows, list(df.columns)


@pytest.fixture(scope="module")
def features_test_df() -> pd.DataFrame:
    """Full test features for label/value checks (80K rows — acceptable)."""
    return pd.read_csv(OUTPUTS_DIR / "features_test.csv")


@pytest.fixture(scope="module")
def baseline_results() -> dict:
    return _load_json("baseline_results.json")


@pytest.fixture(scope="module")
def model_results() -> dict:
    return _load_json("model_results.json")


@pytest.fixture(scope="module")
def evaluation_report() -> dict:
    return _load_json("evaluation_report.json")


@pytest.fixture(scope="module")
def feature_importance() -> dict:
    return _load_json("feature_importance.json")


@pytest.fixture(scope="module")
def latency_benchmark() -> dict:
    return _load_json("latency_benchmark.json")


@pytest.fixture(scope="module")
def ab_test_design() -> dict:
    return _load_json("ab_test_design.json")


@pytest.fixture(scope="module")
def cold_start_analysis() -> dict:
    return _load_json("cold_start_analysis.json")


@pytest.fixture(scope="module")
def monitoring_report() -> dict:
    return _load_json("monitoring_report.json")


@pytest.fixture(scope="module")
def final_report() -> dict:
    return _load_json("final_submission_report.json")


# ===================================================================
# 1. DATA GENERATION (01_data_generator.py)
# ===================================================================

class TestDataGeneration:
    """Validate synthetic data produced by 01_data_generator.py."""

    DATA_FILES = [
        "users.csv", "restaurants.csv", "menu_items.csv",
        "orders.csv", "order_items.csv", "csao_interactions.csv",
    ]

    @pytest.mark.parametrize("filename", DATA_FILES)
    def test_data_files_exist(self, filename: str) -> None:
        assert (DATA_DIR / filename).exists(), f"{filename} missing from data/"

    # -- users --
    def test_users_row_count(self, users_df: pd.DataFrame) -> None:
        assert len(users_df) == 50_000

    def test_users_columns(self, users_df: pd.DataFrame) -> None:
        expected = {"user_id", "city", "user_type", "signup_date",
                    "is_veg_preference", "dinner_only", "avg_budget"}
        assert expected == set(users_df.columns)

    def test_users_city_values(self, users_df: pd.DataFrame) -> None:
        valid = {"Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune"}
        assert set(users_df["city"].unique()) == valid

    def test_users_type_values(self, users_df: pd.DataFrame) -> None:
        valid = {"new", "occasional", "regular", "power"}
        assert set(users_df["user_type"].unique()) == valid

    # -- restaurants --
    def test_restaurants_row_count(self, restaurants_df: pd.DataFrame) -> None:
        assert len(restaurants_df) == 2_000

    def test_restaurants_cuisine_values(self, restaurants_df: pd.DataFrame) -> None:
        valid = {"North Indian", "South Indian", "Chinese", "Biryani",
                 "Street Food", "Continental", "Maharashtrian"}
        assert set(restaurants_df["cuisine_type"].unique()) == valid

    def test_restaurants_tier_values(self, restaurants_df: pd.DataFrame) -> None:
        valid = {"budget", "mid", "premium"}
        assert set(restaurants_df["price_tier"].unique()) == valid

    def test_restaurants_rating_range(self, restaurants_df: pd.DataFrame) -> None:
        assert restaurants_df["avg_rating"].between(2.0, 5.0).all()

    # -- menu items --
    def test_menu_items_row_count(self, menu_items_df: pd.DataFrame) -> None:
        assert 35_000 <= len(menu_items_df) <= 45_000

    def test_menu_items_categories(self, menu_items_df: pd.DataFrame) -> None:
        valid = {"main", "side", "drink", "dessert", "bread", "unknown"}
        assert set(menu_items_df["category"].unique()) == valid

    def test_menu_items_prices_positive(self, menu_items_df: pd.DataFrame) -> None:
        assert (menu_items_df["price"] > 0).all()

    def test_menu_items_unknown_category_ratio(self, menu_items_df: pd.DataFrame) -> None:
        unknown_pct = (menu_items_df["category"] == "unknown").mean()
        assert 0.03 <= unknown_pct <= 0.08, f"Unknown category {unknown_pct:.1%}, expected ~5%"

    # -- orders --
    def test_orders_row_count(self, orders_df: pd.DataFrame) -> None:
        assert 450_000 <= len(orders_df) <= 550_000

    def test_orders_date_range(self, orders_df: pd.DataFrame) -> None:
        dates = pd.to_datetime(orders_df["order_date"])
        assert dates.min() >= pd.Timestamp("2025-01-01")
        assert dates.max() <= pd.Timestamp("2025-06-30")

    def test_orders_mealtime_values(self, orders_df: pd.DataFrame) -> None:
        valid = {"breakfast", "lunch", "snack", "dinner", "late_night"}
        actual = set(orders_df["mealtime"].dropna().unique())
        assert actual == valid

    def test_orders_missing_timestamps(self, orders_df: pd.DataFrame) -> None:
        missing_pct = (orders_df["order_hour"] == -1).mean()
        assert 0.05 <= missing_pct <= 0.15, f"Missing timestamps {missing_pct:.1%}, expected ~10%"

    # -- order items --
    def test_order_items_row_count(self, order_items_df: pd.DataFrame) -> None:
        assert len(order_items_df) > 1_000_000

    def test_order_items_quantity_positive(self, order_items_df: pd.DataFrame) -> None:
        assert (order_items_df["quantity"] >= 1).all()

    # -- interactions --
    def test_interactions_row_count(self, interactions_df: pd.DataFrame) -> None:
        assert len(interactions_df) == 2_000_000

    def test_interactions_position_range(self, interactions_df: pd.DataFrame) -> None:
        assert interactions_df["position"].between(1, 10).all()

    def test_interactions_fire_sequence_range(self, interactions_df: pd.DataFrame) -> None:
        assert interactions_df["fire_sequence"].between(1, 5).all()

    def test_interactions_position_bias(self, interactions_df: pd.DataFrame) -> None:
        """CTR at position 1 should be higher than CTR at position 10."""
        ctr_by_pos = interactions_df.groupby("position")["click"].mean()
        assert ctr_by_pos[1] > ctr_by_pos[10], "No position bias detected"

    def test_cold_start_ratio(self, users_df: pd.DataFrame, orders_df: pd.DataFrame) -> None:
        order_counts = orders_df.groupby("user_id").size()
        n_cold = (order_counts < 3).sum() + (len(users_df) - len(order_counts))
        cold_pct = n_cold / len(users_df)
        assert 0.15 <= cold_pct <= 0.45, f"Cold-start ratio {cold_pct:.1%}, expected ~25-40%"

    def test_referential_integrity_orders(
        self, orders_df: pd.DataFrame, users_df: pd.DataFrame, restaurants_df: pd.DataFrame
    ) -> None:
        assert orders_df["user_id"].isin(users_df["user_id"]).all()
        assert orders_df["restaurant_id"].isin(restaurants_df["restaurant_id"]).all()


# ===================================================================
# 2. FEATURE ENGINEERING (02_feature_engineering.py)
# ===================================================================

class TestFeatureEngineering:
    """Validate feature CSVs from 02_feature_engineering.py."""

    FEATURE_FILES = ["features_train.csv", "features_val.csv", "features_test.csv"]

    @pytest.mark.parametrize("filename", FEATURE_FILES)
    def test_feature_files_exist(self, filename: str) -> None:
        assert (OUTPUTS_DIR / filename).exists()

    def test_train_row_count(self, features_train_meta: tuple) -> None:
        nrows, _ = features_train_meta
        assert nrows == 200_000

    def test_val_row_count(self, features_val_meta: tuple) -> None:
        nrows, _ = features_val_meta
        assert nrows == 80_000

    def test_test_row_count(self, features_test_meta: tuple) -> None:
        nrows, _ = features_test_meta
        assert nrows == 80_000

    def test_column_count(self, features_train_meta: tuple) -> None:
        _, cols = features_train_meta
        assert len(cols) == 229

    def test_all_splits_same_columns(
        self, features_train_meta: tuple, features_val_meta: tuple, features_test_meta: tuple
    ) -> None:
        _, train_cols = features_train_meta
        _, val_cols = features_val_meta
        _, test_cols = features_test_meta
        assert train_cols == val_cols == test_cols

    def test_meta_columns_present(self, features_train_meta: tuple) -> None:
        _, cols = features_train_meta
        for c in ["interaction_id", "order_id", "user_id", "restaurant_id", "item_id", "position"]:
            assert c in cols, f"Meta column {c} missing"

    def test_label_columns_present(self, features_train_meta: tuple) -> None:
        _, cols = features_train_meta
        assert "label" in cols
        assert "click" in cols

    def test_label_no_nan(self, features_test_df: pd.DataFrame) -> None:
        assert features_test_df["label"].notna().all()

    def test_label_binary(self, features_test_df: pd.DataFrame) -> None:
        assert set(features_test_df["label"].unique()).issubset({0, 1, 0.0, 1.0})

    def test_label_positive_rate(self, features_test_df: pd.DataFrame) -> None:
        pos_rate = features_test_df["label"].mean()
        assert 0.05 <= pos_rate <= 0.20, f"Positive rate {pos_rate:.1%}, expected ~10%"

    def test_embedding_columns_present(self, features_train_meta: tuple) -> None:
        _, cols = features_train_meta
        for prefix in ["item_emb_", "cart_emb_"]:
            emb_cols = [c for c in cols if c.startswith(prefix)]
            assert len(emb_cols) >= 20, f"Expected >=20 {prefix} columns, got {len(emb_cols)}"

    def test_feature_groups_present(self, features_train_meta: tuple) -> None:
        _, cols = features_train_meta
        groups = {
            "user": ["user_order_count", "monetary_avg", "recency_days"],
            "cart": ["cart_item_count", "cart_total_value"],
            "item": ["item_price", "item_is_veg"],
            "restaurant": ["rest_avg_rating", "rest_menu_size"],
            "temporal": ["hour_sin", "hour_cos"],
            "cross": ["cross_max_pmi", "cross_embedding_similarity"],
            "llm": ["llm_complementarity_score", "llm_max_semantic_sim"],
        }
        for group, expected_cols in groups.items():
            for c in expected_cols:
                assert c in cols, f"Feature group '{group}' missing column '{c}'"

    def test_no_inf_values(self, features_test_df: pd.DataFrame) -> None:
        numeric = features_test_df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric.values).sum()
        assert inf_counts == 0, f"Found {inf_counts} inf values in test features"


# ===================================================================
# 3. BASELINE MODELS (03_baseline_models.py)
# ===================================================================

class TestBaselineModels:
    """Validate baseline results from 03_baseline_models.py."""

    BASELINES = ["random", "popularity", "cooccurrence", "pmi"]
    METRICS = ["AUC", "NDCG@5", "NDCG@10", "Precision@5", "HR@5", "HR@10", "Recall@10", "MRR"]

    def test_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "baseline_results.json").exists()

    def test_all_baselines_present(self, baseline_results: dict) -> None:
        for b in self.BASELINES:
            assert b in baseline_results, f"Baseline '{b}' missing"

    @pytest.mark.parametrize("baseline", BASELINES)
    def test_baseline_has_all_metrics(self, baseline_results: dict, baseline: str) -> None:
        for m in self.METRICS:
            assert m in baseline_results[baseline], f"{baseline} missing metric '{m}'"

    @pytest.mark.parametrize("baseline", BASELINES)
    def test_metrics_in_valid_range(self, baseline_results: dict, baseline: str) -> None:
        for m in self.METRICS:
            val = baseline_results[baseline][m]
            assert 0.0 <= val <= 1.0, f"{baseline}.{m}={val} out of [0,1]"

    def test_baseline_auc_near_random(self, baseline_results: dict) -> None:
        for b in self.BASELINES:
            auc = baseline_results[b]["AUC"]
            assert 0.40 <= auc <= 0.60, f"{b} AUC={auc}, expected near 0.5"

    def test_pmi_beats_random(self, baseline_results: dict) -> None:
        assert baseline_results["pmi"]["NDCG@5"] > baseline_results["random"]["NDCG@5"]


# ===================================================================
# 4. MODEL TRAINING (04_model_training.py)
# ===================================================================

class TestModelTraining:
    """Validate model results and artifacts from 04_model_training.py."""

    STAGES = [
        "two_tower_retrieval",
        "l1_lightgbm_ips_debiased",
        "l2_dcnv2_pytorch",
        "full_pipeline_l1_l2_mmr",
    ]
    STAGE_METRICS = ["AUC", "NDCG@5", "Precision@5", "HR@5", "Recall@10", "MRR", "Catalog_Coverage"]

    def test_model_results_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "model_results.json").exists()

    def test_all_stages_present(self, model_results: dict) -> None:
        for s in self.STAGES:
            assert s in model_results, f"Stage '{s}' missing"

    @pytest.mark.parametrize("stage", STAGES)
    def test_stage_has_all_metrics(self, model_results: dict, stage: str) -> None:
        for m in self.STAGE_METRICS:
            assert m in model_results[stage], f"{stage} missing metric '{m}'"

    @pytest.mark.parametrize("stage", STAGES)
    def test_stage_metrics_in_range(self, model_results: dict, stage: str) -> None:
        for m in self.STAGE_METRICS:
            val = model_results[stage][m]
            assert 0.0 <= val <= 1.0, f"{stage}.{m}={val} out of [0,1]"

    def test_l1_auc_above_baseline(self, model_results: dict, baseline_results: dict) -> None:
        l1_auc = model_results["l1_lightgbm_ips_debiased"]["AUC"]
        best_bl = max(baseline_results[b]["AUC"] for b in baseline_results)
        assert l1_auc > best_bl, f"L1 AUC {l1_auc} <= best baseline {best_bl}"

    def test_l2_auc_ge_l1(self, model_results: dict) -> None:
        l1 = model_results["l1_lightgbm_ips_debiased"]["AUC"]
        l2 = model_results["l2_dcnv2_pytorch"]["AUC"]
        assert l2 >= l1 - 0.01, f"L2 AUC {l2} unexpectedly lower than L1 {l1}"

    def test_full_pipeline_ndcg_beats_baselines(
        self, model_results: dict, baseline_results: dict
    ) -> None:
        fp = model_results["full_pipeline_l1_l2_mmr"]["NDCG@5"]
        best_bl = max(baseline_results[b]["NDCG@5"] for b in baseline_results)
        assert fp > best_bl, f"Full pipeline NDCG@5 {fp} <= best baseline {best_bl}"

    def test_model_details_present(self, model_results: dict) -> None:
        assert "model_details" in model_results
        details = model_results["model_details"]
        for gap in ["gap_10_ips", "gap_1_two_tower", "l1", "gap_3_sasrec", "gap_2_dcnv2", "gap_5_mmr"]:
            assert gap in details, f"model_details missing '{gap}'"

    def test_ai_edge_techniques(self, model_results: dict) -> None:
        assert "ai_edge" in model_results
        implemented = model_results["ai_edge"]["implemented"]
        assert len(implemented) >= 10, f"Only {len(implemented)} AI edge techniques, expected >=10"

    # -- Model artifacts --
    PT_FILES = ["two_tower_model.pt", "sasrec_model.pt", "dcnv2_model.pt"]

    @pytest.mark.parametrize("filename", PT_FILES)
    def test_pytorch_model_exists(self, filename: str) -> None:
        assert (OUTPUTS_DIR / filename).exists()

    @pytest.mark.parametrize("filename", PT_FILES)
    def test_pytorch_model_loadable(self, filename: str) -> None:
        import torch
        state = torch.load(OUTPUTS_DIR / filename, map_location="cpu", weights_only=False)
        assert state is not None

    def test_faiss_index_exists(self) -> None:
        assert (OUTPUTS_DIR / "two_tower_faiss.index").exists()

    def test_faiss_index_loadable(self) -> None:
        import faiss
        index = faiss.read_index(str(OUTPUTS_DIR / "two_tower_faiss.index"))
        assert index.d == 64, f"FAISS dim={index.d}, expected 64"
        assert index.ntotal > 30_000, f"FAISS has {index.ntotal} vectors, expected >30K"

    def test_item_embeddings(self) -> None:
        embs = np.load(OUTPUTS_DIR / "two_tower_item_embeddings.npy")
        ids = np.load(OUTPUTS_DIR / "two_tower_item_ids.npy", allow_pickle=True)
        assert embs.shape[1] == 64
        assert embs.shape[0] == len(ids), "Embedding/ID count mismatch"
        assert not np.isnan(embs).any(), "NaN in item embeddings"

    def test_dcnv2_normalization_arrays(self) -> None:
        mean = np.load(OUTPUTS_DIR / "dcnv2_norm_mean.npy")
        std = np.load(OUTPUTS_DIR / "dcnv2_norm_std.npy")
        assert mean.shape == std.shape
        assert (std >= 0).all(), "Negative std values"


# ===================================================================
# 5. EVALUATION (05_evaluation.py)
# ===================================================================

class TestEvaluation:
    """Validate evaluation outputs from 05_evaluation.py."""

    EVAL_FILES = ["evaluation_report.json", "segment_analysis.json", "feature_importance.json"]

    @pytest.mark.parametrize("filename", EVAL_FILES)
    def test_eval_files_exist(self, filename: str) -> None:
        assert (OUTPUTS_DIR / filename).exists()

    # -- System comparison --
    EXPECTED_SYSTEMS = ["random", "popularity", "cooccurrence", "pmi",
                        "LightGBM_L1", "L1_L2_DCNv2", "Full_Pipeline"]

    def test_system_comparison_has_all_systems(self, evaluation_report: dict) -> None:
        systems = evaluation_report["system_comparison"]
        for s in self.EXPECTED_SYSTEMS:
            assert s in systems, f"System '{s}' missing from comparison"

    # -- Segment analysis --
    SEGMENT_DIMS = {
        "user_type": 4,
        "meal_period": 5,
        "city": 5,
        "cart_size": 4,
        "restaurant_tier": 3,
        "cuisine_type": 7,
    }

    def test_segment_dimensions_present(self, evaluation_report: dict) -> None:
        segments = evaluation_report["segment_analysis"]
        for dim in self.SEGMENT_DIMS:
            assert dim in segments, f"Segment dimension '{dim}' missing"

    @pytest.mark.parametrize("dim,expected_count", list(SEGMENT_DIMS.items()))
    def test_segment_count(self, evaluation_report: dict, dim: str, expected_count: int) -> None:
        segments = evaluation_report["segment_analysis"][dim]
        assert len(segments) == expected_count, (
            f"{dim}: {len(segments)} segments, expected {expected_count}"
        )

    def test_segment_sample_counts_positive(self, evaluation_report: dict) -> None:
        for dim, segs in evaluation_report["segment_analysis"].items():
            for seg_name, seg_data in segs.items():
                assert seg_data.get("sample_count", 0) > 0, (
                    f"{dim}/{seg_name} has sample_count=0"
                )

    # -- Feature importance --
    def test_top_20_features(self, feature_importance: dict) -> None:
        assert len(feature_importance["top_20"]) == 20

    def test_group_importance_sums_to_100(self, feature_importance: dict) -> None:
        total = sum(feature_importance["group_importance"].values())
        assert 95.0 <= total <= 105.0, f"Group importance sums to {total}, expected ~100"

    def test_shap_analysis_present(self, feature_importance: dict) -> None:
        shap = feature_importance["shap_analysis"]
        assert "top_20_shap" in shap
        assert len(shap["top_20_shap"]) == 20
        assert shap["method"] == "TreeExplainer"

    def test_worst_segments_count(self, evaluation_report: dict) -> None:
        assert len(evaluation_report["worst_segments"]) == 10

    def test_pareto_frontier_count(self, evaluation_report: dict) -> None:
        assert len(evaluation_report["pareto_frontier"]) == 7


# ===================================================================
# 6. SYSTEM DESIGN (06_system_design.py)
# ===================================================================

class TestSystemDesign:
    """Validate system design outputs from 06_system_design.py."""

    def test_latency_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "latency_benchmark.json").exists()

    def test_monitoring_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "monitoring_report.json").exists()

    def test_cold_start_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "cold_start_analysis.json").exists()

    LATENCY_COMPONENTS = [
        "api_gateway", "feature_retrieval", "candidate_generation",
        "l1_ranking_lgbm", "l2_ranking_dcn", "mmr_postprocessing",
        "serialization", "network_overhead",
    ]

    def test_latency_components_present(self, latency_benchmark: dict) -> None:
        stats = latency_benchmark["latency_simulation"]["component_stats"]
        for c in self.LATENCY_COMPONENTS:
            assert c in stats, f"Latency component '{c}' missing"

    def test_p95_under_200ms_sequential(self, latency_benchmark: dict) -> None:
        p95 = latency_benchmark["latency_simulation"]["sequential"]["p95"]
        assert p95 < 200, f"Sequential P95={p95}ms exceeds 200ms SLA"

    def test_p95_under_200ms_parallel(self, latency_benchmark: dict) -> None:
        p95 = latency_benchmark["latency_simulation"]["parallel_optimized"]["p95"]
        assert p95 < 200, f"Parallel P95={p95}ms exceeds 200ms SLA"

    def test_parallel_faster_than_sequential(self, latency_benchmark: dict) -> None:
        sim = latency_benchmark["latency_simulation"]
        assert sim["parallel_optimized"]["p95"] < sim["sequential"]["p95"]

    def test_fallback_chain(self, latency_benchmark: dict) -> None:
        chain = latency_benchmark["fallback_chain"]
        assert len(chain) == 4, f"Fallback chain has {len(chain)} levels, expected 4"

    def test_scalability_section(self, latency_benchmark: dict) -> None:
        s = latency_benchmark["scalability"]
        assert s["peak_target_rps"] == 10_000
        assert s["instances_with_headroom"] >= s["instances_needed"]

    # -- Monitoring --
    MONITOR_DIMS = [
        "1_prediction_quality", "2_feature_drift", "3_score_distribution",
        "4_business_kpis", "5_system_health", "6_coverage",
    ]

    def test_monitoring_dimensions(self, monitoring_report: dict) -> None:
        dims = monitoring_report["monitoring_dimensions"]
        for d in self.MONITOR_DIMS:
            assert d in dims, f"Monitoring dimension '{d}' missing"

    def test_psi_features_monitored(self, monitoring_report: dict) -> None:
        psi = monitoring_report["psi_results"]
        assert len(psi) >= 15, f"Only {len(psi)} PSI features, expected >=15"

    # -- Cold start --
    def test_thompson_sampling_beats_popularity(self, cold_start_analysis: dict) -> None:
        ts = cold_start_analysis["thompson_sampling"]["accept_rate"]
        pop = cold_start_analysis["popularity_baseline"]["accept_rate"]
        assert ts > pop, f"Thompson {ts} <= Popularity {pop}"

    def test_cuisine_transfer_coverage(self, cold_start_analysis: dict) -> None:
        cuisines = cold_start_analysis["cuisine_transfer"]
        assert len(cuisines) == 7


# ===================================================================
# 7. BUSINESS IMPACT (07_business_impact.py)
# ===================================================================

class TestBusinessImpact:
    """Validate business impact outputs from 07_business_impact.py."""

    def test_ab_test_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "ab_test_design.json").exists()

    def test_final_report_file_exists(self) -> None:
        assert (OUTPUTS_DIR / "final_submission_report.json").exists()

    # -- A/B test design --
    def test_ab_test_unit(self, ab_test_design: dict) -> None:
        assert ab_test_design["unit"] == "user_id"

    def test_ab_test_allocations_sum_to_one(self, ab_test_design: dict) -> None:
        total = ab_test_design["control"]["allocation"] + ab_test_design["treatment"]["allocation"]
        assert abs(total - 1.0) < 0.001

    def test_ab_test_primary_metrics(self, ab_test_design: dict) -> None:
        expected = {"aov_lift", "addon_acceptance_rate", "items_per_order"}
        assert set(ab_test_design["primary_metrics"].keys()) == expected

    def test_ab_test_guardrail_metrics(self, ab_test_design: dict) -> None:
        expected = {"cart_to_order_rate", "order_completion_time",
                    "csao_dismiss_rate", "app_crash_rate"}
        assert set(ab_test_design["guardrail_metrics"].keys()) == expected

    def test_ab_test_power_analysis(self, ab_test_design: dict) -> None:
        pa = ab_test_design["power_analysis"]
        assert pa["alpha"] == 0.05
        assert pa["power"] == 0.8

    # -- Final report --
    REPORT_SECTIONS = [
        "1_problem_formulation", "2_data_generation", "3_feature_pipeline",
        "4_baseline_results", "5_model_results", "6_segment_analysis",
        "7_feature_importance", "8_latency_benchmark", "9_ab_test_design",
        "10_business_impact", "11_cold_start_strategy", "12_llm_integration",
        "13_tradeoffs_and_limitations", "14_phased_deployment",
    ]

    def test_final_report_sections(self, final_report: dict) -> None:
        for section in self.REPORT_SECTIONS:
            assert section in final_report, f"Final report missing section '{section}'"

    def test_final_report_deployment_phases(self, final_report: dict) -> None:
        deploy = final_report["14_phased_deployment"]
        assert "phases" in deploy
        assert len(deploy["phases"]) == 3, "Expected 3 deployment phases"


# ===================================================================
# 8. DOCUMENTATION (docs/)
# ===================================================================

class TestDocumentation:
    """Validate documentation completeness."""

    DOC_FILES = [
        "feature_dictionary.md",
        "system_architecture.md",
        "problem_formulation.md",
        "streaming_pipeline.md",
        "retraining_strategy.md",
        "deployment_playbook.md",
    ]

    @pytest.mark.parametrize("filename", DOC_FILES)
    def test_doc_file_exists(self, filename: str) -> None:
        assert (DOCS_DIR / filename).exists(), f"docs/{filename} missing"

    @pytest.mark.parametrize("filename", DOC_FILES)
    def test_doc_file_non_empty(self, filename: str) -> None:
        size = (DOCS_DIR / filename).stat().st_size
        assert size > 100, f"docs/{filename} is too small ({size} bytes)"


# ===================================================================
# 9. CROSS-MODULE CONSISTENCY
# ===================================================================

class TestCrossModuleConsistency:
    """Validate consistency across different pipeline outputs."""

    def test_baseline_metrics_match_evaluation_report(
        self, baseline_results: dict, evaluation_report: dict
    ) -> None:
        """Baseline metrics in evaluation_report should match baseline_results.json."""
        for baseline in ["random", "popularity", "cooccurrence", "pmi"]:
            eval_sys = evaluation_report["system_comparison"].get(baseline, {})
            orig = baseline_results[baseline]
            if eval_sys and "AUC" in eval_sys:
                assert abs(eval_sys["AUC"] - orig["AUC"]) < 0.01, (
                    f"{baseline} AUC mismatch: eval={eval_sys['AUC']}, baseline={orig['AUC']}"
                )

    def test_full_pipeline_metrics_consistency(
        self, model_results: dict, evaluation_report: dict
    ) -> None:
        """Full pipeline NDCG@5 should be close across reports."""
        mr = model_results["full_pipeline_l1_l2_mmr"]["NDCG@5"]
        er = evaluation_report["system_comparison"].get("Full_Pipeline", {}).get("NDCG@5")
        if er is not None:
            assert abs(mr - er) < 0.02, f"NDCG@5 mismatch: model={mr}, eval={er}"

    def test_faiss_index_matches_menu_items(self, menu_items_df: pd.DataFrame) -> None:
        """FAISS index item count should be close to unique menu items."""
        import faiss
        index = faiss.read_index(str(OUTPUTS_DIR / "two_tower_faiss.index"))
        n_menu = len(menu_items_df)
        assert abs(index.ntotal - n_menu) / n_menu < 0.1, (
            f"FAISS={index.ntotal} vs menu_items={n_menu}"
        )

    def test_feature_importance_aligns_with_features(
        self, feature_importance: dict, features_train_meta: tuple
    ) -> None:
        """Top-20 features should all exist in the feature CSV or be runtime-generated."""
        _, cols = features_train_meta
        # Features generated at runtime by models (not in feature CSV)
        runtime_features = {"two_tower_score"} | {f"sequential_emb_{i}" for i in range(16)}
        for entry in feature_importance["top_20"]:
            feat = entry["feature"]
            assert feat in cols or feat in runtime_features, (
                f"Feature importance lists '{feat}' not in feature CSV or runtime features"
            )

    def test_final_report_baseline_consistency(
        self, final_report: dict, baseline_results: dict
    ) -> None:
        """Final report baseline section should reference same baselines."""
        report_baselines = set(final_report["4_baseline_results"].keys())
        orig_baselines = set(baseline_results.keys())
        assert report_baselines == orig_baselines
