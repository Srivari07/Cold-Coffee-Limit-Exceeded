"""
03_baseline_models.py — Baseline Models for CSAO Rail Recommendation System
Implements 4 baselines: Random, Popularity, Co-occurrence, PMI

Criterion 2 & 6: Ideation & Problem Formulation (15%) + Business Impact (15%)
"""

import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, ndcg_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CUTOFF = "2025-04-30"
TEST_START = "2025-06-02"
TEST_END = "2025-06-30"


# ─── Metric Computation ─────────────────────────────────────────────────────────

def compute_metrics(y_true_groups, y_score_groups, k=5):
    """Compute ranking metrics per group (order), then average."""
    aucs, ndcgs_5, ndcgs_10, prec_5, hr_5, hr_10, recall_10, mrrs = [], [], [], [], [], [], [], []
    all_recommended = set()
    all_items = set()

    for y_true, y_score in zip(y_true_groups, y_score_groups):
        if len(y_true) < 2:
            continue

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # AUC
        if len(set(y_true)) > 1:
            try:
                aucs.append(roc_auc_score(y_true, y_score))
            except:
                pass

        # Rank by score
        ranked_idx = np.argsort(-y_score)
        ranked_labels = y_true[ranked_idx]

        # NDCG@5, NDCG@10
        if len(y_true) >= 5:
            ndcgs_5.append(ndcg_score([y_true], [y_score], k=5))
        if len(y_true) >= 10:
            ndcgs_10.append(ndcg_score([y_true], [y_score], k=10))

        # Precision@5
        top5 = ranked_labels[:5]
        prec_5.append(top5.sum() / min(5, len(top5)))

        # Hit Rate@5: at least one relevant in top 5
        hr_5.append(1 if top5.sum() > 0 else 0)

        # Hit Rate@10
        top10 = ranked_labels[:min(10, len(ranked_labels))]
        hr_10.append(1 if top10.sum() > 0 else 0)

        # Recall@10
        total_relevant = y_true.sum()
        if total_relevant > 0:
            recall_10.append(top10.sum() / total_relevant)

        # MRR
        for rank, label in enumerate(ranked_labels):
            if label == 1:
                mrrs.append(1.0 / (rank + 1))
                break
        else:
            mrrs.append(0)

        # Track for coverage
        all_items.update(range(len(y_true)))
        all_recommended.update(ranked_idx[:5].tolist())

    metrics = {
        "AUC": round(np.mean(aucs), 4) if aucs else 0.5,
        "NDCG@5": round(np.mean(ndcgs_5), 4) if ndcgs_5 else 0,
        "NDCG@10": round(np.mean(ndcgs_10), 4) if ndcgs_10 else 0,
        "Precision@5": round(np.mean(prec_5), 4) if prec_5 else 0,
        "HR@5": round(np.mean(hr_5), 4) if hr_5 else 0,
        "HR@10": round(np.mean(hr_10), 4) if hr_10 else 0,
        "Recall@10": round(np.mean(recall_10), 4) if recall_10 else 0,
        "MRR": round(np.mean(mrrs), 4) if mrrs else 0,
        "Catalog_Coverage": round(len(all_recommended) / max(len(all_items), 1), 4),
    }
    return metrics


# ─── Load Data ───────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))
    menu_items = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "csao_interactions.csv"))

    orders["order_date_dt"] = pd.to_datetime(orders["order_date"])

    # Split
    orders_train = orders[orders["order_date_dt"] <= TRAIN_CUTOFF]
    orders_test = orders[(orders["order_date_dt"] >= TEST_START) & (orders["order_date_dt"] <= TEST_END)]

    interactions_test = interactions[interactions["order_date"].between(TEST_START, TEST_END)]
    order_items_train = order_items[order_items["order_id"].isin(orders_train["order_id"])]

    print(f"  Train orders: {len(orders_train):,}")
    print(f"  Test interactions: {len(interactions_test):,}")
    return orders_train, orders_test, order_items_train, interactions_test, menu_items


# ─── Baseline 1: Random ─────────────────────────────────────────────────────────

def baseline_random(interactions_test):
    """Random scores for each candidate."""
    print("\n  Running Baseline 1: Random...")
    groups_true = []
    groups_score = []

    for order_id, group in interactions_test.groupby("order_id"):
        y_true = group["add_to_cart"].astype(int).values
        y_score = np.random.random(len(y_true))
        groups_true.append(y_true)
        groups_score.append(y_score)

    metrics = compute_metrics(groups_true, groups_score)
    print(f"    AUC={metrics['AUC']:.4f}, NDCG@5={metrics['NDCG@5']:.4f}, HR@5={metrics['HR@5']:.4f}")
    return metrics


# ─── Baseline 2: Popularity ─────────────────────────────────────────────────────

def baseline_popularity(interactions_test, order_items_train, orders_train, menu_items):
    """Rank by popularity (order count in last 30 days of training)."""
    print("\n  Running Baseline 2: Popularity...")

    # Compute item popularity by restaurant + mealtime
    train_end = pd.to_datetime(TRAIN_CUTOFF)
    recent = orders_train[pd.to_datetime(orders_train["order_date"]) >= train_end - pd.Timedelta(days=30)]
    recent_items = order_items_train[order_items_train["order_id"].isin(recent["order_id"])]

    # Merge mealtime info
    recent_items_m = recent_items.merge(recent[["order_id", "restaurant_id", "mealtime"]], on="order_id")
    pop_scores = recent_items_m.groupby(["restaurant_id", "item_id"]).size().reset_index(name="pop_count")
    pop_dict = {}
    for _, row in pop_scores.iterrows():
        pop_dict[(row["restaurant_id"], row["item_id"])] = row["pop_count"]

    # Global popularity as fallback
    global_pop = recent_items["item_id"].value_counts().to_dict()

    groups_true = []
    groups_score = []

    for order_id, group in interactions_test.groupby("order_id"):
        y_true = group["add_to_cart"].astype(int).values
        rest_id = group["restaurant_id"].iloc[0]
        scores = []
        for _, row in group.iterrows():
            score = pop_dict.get((rest_id, row["item_id"]), global_pop.get(row["item_id"], 0))
            scores.append(score)
        groups_true.append(y_true)
        groups_score.append(np.array(scores, dtype=float))

    metrics = compute_metrics(groups_true, groups_score)
    print(f"    AUC={metrics['AUC']:.4f}, NDCG@5={metrics['NDCG@5']:.4f}, HR@5={metrics['HR@5']:.4f}")
    return metrics


# ─── Baseline 3: Co-occurrence ───────────────────────────────────────────────────

def baseline_cooccurrence(interactions_test, order_items_train):
    """Rank by co-occurrence count with cart items."""
    print("\n  Running Baseline 3: Co-occurrence...")

    # Build co-occurrence matrix from training
    cooc = defaultdict(lambda: defaultdict(int))
    for order_id, group in order_items_train.groupby("order_id"):
        items = group["item_id"].unique().tolist()
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    cooc[items[i]][items[j]] += 1

    groups_true = []
    groups_score = []

    for order_id, group in interactions_test.groupby("order_id"):
        y_true = group["add_to_cart"].astype(int).values

        # Get cart items from cart_state
        cart_state = str(group["cart_state"].iloc[0])
        cart_items = [c for c in cart_state.split("|") if c]

        scores = []
        for _, row in group.iterrows():
            cand_id = row["item_id"]
            score = sum(cooc[cart_item].get(cand_id, 0) for cart_item in cart_items)
            scores.append(score)

        groups_true.append(y_true)
        groups_score.append(np.array(scores, dtype=float))

    metrics = compute_metrics(groups_true, groups_score)
    print(f"    AUC={metrics['AUC']:.4f}, NDCG@5={metrics['NDCG@5']:.4f}, HR@5={metrics['HR@5']:.4f}")
    return metrics


# ─── Baseline 4: PMI ────────────────────────────────────────────────────────────

def baseline_pmi(interactions_test, order_items_train):
    """Rank by PMI score with cart items."""
    print("\n  Running Baseline 4: PMI...")

    # Compute PMI
    order_groups = order_items_train.groupby("order_id")["item_id"].apply(list)
    total_orders = len(order_groups)

    item_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for items in order_groups:
        unique = list(set(items))
        for item in unique:
            item_counts[item] += 1
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = tuple(sorted([unique[i], unique[j]]))
                pair_counts[pair] += 1

    pmi_scores = {}
    for (a, b), count in pair_counts.items():
        if count < 3:
            continue
        p_ab = count / total_orders
        p_a = item_counts[a] / total_orders
        p_b = item_counts[b] / total_orders
        if p_a > 0 and p_b > 0:
            pmi = np.log2(p_ab / (p_a * p_b))
            pmi_scores[(a, b)] = pmi
            pmi_scores[(b, a)] = pmi

    groups_true = []
    groups_score = []

    for order_id, group in interactions_test.groupby("order_id"):
        y_true = group["add_to_cart"].astype(int).values

        cart_state = str(group["cart_state"].iloc[0])
        cart_items = [c for c in cart_state.split("|") if c]

        scores = []
        for _, row in group.iterrows():
            cand_id = row["item_id"]
            pmi_vals = [pmi_scores.get((cart_item, cand_id), 0) for cart_item in cart_items]
            score = max(pmi_vals) if pmi_vals else 0
            scores.append(score)

        groups_true.append(y_true)
        groups_score.append(np.array(scores, dtype=float))

    metrics = compute_metrics(groups_true, groups_score)
    print(f"    AUC={metrics['AUC']:.4f}, NDCG@5={metrics['NDCG@5']:.4f}, HR@5={metrics['HR@5']:.4f}")
    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CSAO Baseline Models")
    print("=" * 70)

    orders_train, orders_test, order_items_train, interactions_test, menu_items = load_data()

    results = {}

    # Run all baselines
    results["random"] = baseline_random(interactions_test)
    results["popularity"] = baseline_popularity(interactions_test, order_items_train, orders_train, menu_items)
    results["cooccurrence"] = baseline_cooccurrence(interactions_test, order_items_train)
    results["pmi"] = baseline_pmi(interactions_test, order_items_train)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Baseline':<20} {'AUC':>8} {'NDCG@5':>8} {'NDCG@10':>8} {'P@5':>8} {'HR@5':>8} {'HR@10':>8} {'MRR':>8} {'Coverage':>10}"
    print(header)
    print("-" * 90)
    for name, metrics in results.items():
        row = (f"{name:<20} {metrics['AUC']:>8.4f} {metrics['NDCG@5']:>8.4f} "
               f"{metrics['NDCG@10']:>8.4f} {metrics['Precision@5']:>8.4f} "
               f"{metrics['HR@5']:>8.4f} {metrics['HR@10']:>8.4f} "
               f"{metrics['MRR']:>8.4f} {metrics['Catalog_Coverage']:>10.4f}")
        print(row)
    print("=" * 90)

    print("\nBaseline evaluation complete!")


if __name__ == "__main__":
    main()
