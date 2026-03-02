# CSAO Rail Recommendation System

Cart Super Add-On (CSAO) rail recommendation system for a food delivery platform. When a customer adds items to their cart, a horizontal rail shows complementary add-on suggestions that update in real-time as the cart changes.

## Setup

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, ~4GB RAM. No GPU needed — everything runs on CPU.

## How to Run

Run scripts sequentially — each depends on the previous:

```bash
python 01_data_generator.py          # Generates data/ (6 CSVs)
python 02_feature_engineering.py     # Generates outputs/features_*.csv
python 03_baseline_models.py         # Generates outputs/baseline_results.json
python 04_model_training.py          # Trains models, generates outputs/model_results.json
python 05_evaluation.py              # Generates evaluation + segment reports
python 06_system_design.py           # Generates latency benchmarks + architecture docs
python 07_business_impact.py         # Generates A/B test design + final report
```

Total runtime: ~15 minutes on a modern laptop.

## Project Structure

```
├── 01_data_generator.py         # 50K users, 2K restaurants, 500K orders, 2M interactions
├── 02_feature_engineering.py    # 200+ features across 7 groups with temporal split
├── 03_baseline_models.py        # Random, Popularity, Co-occurrence, PMI baselines
├── 04_model_training.py         # LightGBM L1 + DCN-v2 L2 + MMR diversity reranking
├── 05_evaluation.py             # Metrics + segment analysis across 6 dimensions
├── 06_system_design.py          # Latency simulation + scalability + architecture
├── 07_business_impact.py        # A/B test design + business projections + final report
├── data/                        # Generated CSV files
├── outputs/                     # JSON reports + feature files
└── docs/                        # Architecture + problem formulation
```

## Key Results

| System                    | AUC  | NDCG@5 | HR@5 | P@5  | Latency |
|---------------------------|------|--------|------|------|---------|
| Random                    | ~0.50| ~0.08  | ~0.12| ~0.05| 1ms     |
| Popularity                | ~0.58| ~0.15  | ~0.22| ~0.10| 3ms     |
| PMI                       | ~0.65| ~0.22  | ~0.30| ~0.15| 3ms     |
| LightGBM L1               | ~0.75| ~0.32  | ~0.38| ~0.19| 20ms    |
| Full Pipeline (L1+L2+MMR) | ~0.78| ~0.36  | ~0.43| ~0.22| 120ms   |

**Projected business impact**: +20-24% AOV lift, 22% add-on acceptance rate, P95 latency <200ms.

## Evaluation Criteria

| # | Criterion                          | Weight |
|---|-----------------------------------|--------|
| 1 | Data Preparation & Feature Eng.    | 20%    |
| 2 | Ideation & Problem Formulation     | 15%    |
| 3 | Model Architecture & AI Edge       | 20%    |
| 4 | Evaluation & Fine-Tuning           | 15%    |
| 5 | System Design & Production Ready   | 15%    |
| 6 | Business Impact & A/B Testing      | 15%    |
