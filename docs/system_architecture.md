# CSAO Rail Recommendation System — System Architecture

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CSAO RECOMMENDATION PIPELINE                     │
│                                                                         │
│  Client App (Cart Update Event)                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                      │
│  │ API Gateway   │ ~8ms                                                │
│  │ (Rate Limit,  │                                                     │
│  │  Auth, Route) │                                                     │
│  └──────┬───────┘                                                      │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────┐                          │
│  │        PARALLEL EXECUTION BLOCK          │                          │
│  │  ┌──────────────┐  ┌──────────────────┐  │                          │
│  │  │  Feature      │  │  Candidate       │  │                          │
│  │  │  Retrieval    │  │  Generation      │  │                          │
│  │  │  (Redis)      │  │  (Redis/Filter)  │  │                          │
│  │  │  ~18ms        │  │  ~10ms           │  │                          │
│  │  └──────┬───────┘  └──────┬───────────┘  │                          │
│  │         │                  │              │                          │
│  │         └────────┬─────────┘              │                          │
│  └──────────────────┼───────────────────────┘                          │
│                     │  max(18, 10) = ~18ms                              │
│                     ▼                                                   │
│  ┌──────────────────────────────────────┐                              │
│  │  L1 Ranking (LightGBM)              │                              │
│  │  200 candidates → top 30            │                              │
│  │  ~22ms                              │                              │
│  └──────────────┬──────────────────────┘                              │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────────────────────────────┐                              │
│  │  L2 Ranking (DCN-v2 / Cross Model)  │                              │
│  │  30 candidates → scored             │                              │
│  │  ~30ms                              │                              │
│  └──────────────┬──────────────────────┘                              │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────────────────────────────┐                              │
│  │  MMR Post-Processing                │                              │
│  │  Diversity re-ranking → top 8       │                              │
│  │  ~6ms                               │                              │
│  └──────────────┬──────────────────────┘                              │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────┐                                                      │
│  │ Serialization │ ~3ms                                                │
│  │ + Response    │                                                     │
│  └──────────────┘                                                      │
│                                                                         │
│  Total E2E (parallel): ~117.3ms P95                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Latency Budget

| Component              | Mean (ms) | P95 (ms) | Notes                           |
|------------------------|-----------|----------|---------------------------------|
| API Gateway            | 8         | 12       | Rate limiting, auth, routing    |
| Feature Retrieval      | 18        | 28       | Redis lookup (user+cart+item)   |
| Candidate Generation   | 10        | 18       | Pre-computed Redis + filtering  |
| L1 Ranking (LightGBM)  | 22        | 35       | Score 200 candidates            |
| L2 Ranking (DCN-v2)    | 30        | 45       | Re-rank top 30 from L1          |
| MMR Post-Processing    | 6         | 12       | Diversity re-ranking            |
| Serialization          | 3         | 5        | JSON serialization              |
| Network Overhead       | 8         | 15       | Client ↔ server round-trip     |

**Parallel Optimization**: Feature Retrieval and Candidate Generation execute in parallel,
saving ~10.8ms at P95.

**E2E P95: ~117.3ms** (within 200ms budget)

## 3. Feature Store Architecture

```
┌─────────────────────────────────────────────────┐
│              FEATURE STORE (Redis Cluster)        │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐               │
│  │ Batch Layer  │  │ Stream Layer │               │
│  │ (Daily)      │  │ (Real-time)  │               │
│  │              │  │              │               │
│  │ User RFM     │  │ Cart state   │               │
│  │ Item stats   │  │ Session CTR  │               │
│  │ PMI matrix   │  │ Real-time    │               │
│  │ Embeddings   │  │ interactions │               │
│  └─────────────┘  └─────────────┘               │
│                                                   │
│  Storage: ~0.9 GB                     │
│  Cluster: 3-node, 8GB each                       │
└─────────────────────────────────────────────────┘
```

- **Batch Layer**: Refreshed daily via Spark/Airflow pipeline. Computes user features,
  item popularity, PMI matrix, and embeddings.
- **Stream Layer**: Updated in real-time via Kafka. Tracks cart state, session-level
  CTR, and real-time interactions.

## 4. Graceful Degradation Chain

When latency exceeds thresholds or components fail, the system automatically falls back:

| Priority | Strategy          | Description                    | Latency | Trigger                     |
|----------|-------------------|--------------------------------|---------|------------------------------|
| 1        | Full Pipeline     | L1 + L2 + MMR                 | ~120ms  | Normal operation             |
| 2        | LightGBM Only     | L1 ranking only (skip L2)     | ~65ms   | L2 timeout > 60ms            |
| 3        | PMI Lookup        | Pre-computed PMI from Redis    | ~15ms   | ML serving unavailable       |
| 4        | Popularity        | Most popular items per restaurant | ~5ms | Redis feature store down     |

**Circuit Breaker Pattern**: Each stage has a circuit breaker that opens after 5 consecutive
failures or when P95 exceeds 2x the budget. Recovery is attempted every 30 seconds.

## 5. Scalability Plan

### Target: 10,000 req/sec (Dinner Peak)

| Component          | Instances | Type          | Notes                        |
|--------------------|-----------|---------------|------------------------------|
| ML Serving (Triton)| 10        | c5.2xlarge    | 2,000 rps each, 2x headroom |
| Redis Cluster      | 3         | r5.2xlarge    | 0.9 GB total      |
| API Gateway        | 2         | c5.xlarge     | NGINX with load balancing    |
| Monitoring         | 1         | m5.xlarge     | Prometheus + Grafana         |

### Auto-Scaling Policy
- Scale UP when P95 > 150ms for 3 consecutive minutes
- Scale DOWN when P95 < 80ms for 10 consecutive minutes
- Min instances: 5, Max instances: 20

## 6. Benchmarking Strategy

### Load Testing (Locust/k6)
| Scenario        | Target RPS | Duration | Purpose                        |
|-----------------|-----------|----------|--------------------------------|
| Baseline        | 5,000     | 10 min   | Verify normal operation        |
| Peak            | 10,000    | 10 min   | Verify dinner peak handling    |
| Stress          | 15,000    | 5 min    | Find breaking point            |
| Endurance       | 8,000     | 60 min   | Memory leak detection          |

### Shadow Deployment (5 days)
1. Day 1-2: Deploy to shadow environment, mirror 10% of production traffic
2. Day 3: Increase to 50% mirrored traffic, compare latency distributions
3. Day 4-5: Full mirror, validate metrics match offline evaluation within 5%

### Canary Release
| Stage | Traffic % | Duration | Gate Criteria                     |
|-------|-----------|----------|-----------------------------------|
| 1     | 5%        | 24h      | P95 < 200ms, error rate < 0.1%   |
| 2     | 20%       | 48h      | Above + AOV lift ≥ 2%            |
| 3     | 50%       | 72h      | Above + acceptance rate ≥ 15%    |
| 4     | 100%      | -        | All gates passed                  |

## 7. Monitoring & Alerting

### Key Metrics
- **Latency**: P50, P95, P99 per component and end-to-end
- **Throughput**: Requests per second, queue depth
- **Error Rate**: 4xx, 5xx, timeout rates
- **Business**: CSAO impression rate, CTR, add-to-cart rate, AOV
- **Model**: Feature freshness, prediction distribution drift, fallback rate

### Alert Thresholds
| Metric                  | Warning    | Critical   | Action                    |
|-------------------------|------------|------------|---------------------------|
| E2E P95 latency         | > 150ms    | > 200ms    | Scale up / degrade to L1  |
| Error rate              | > 0.5%     | > 1%       | Page on-call              |
| Fallback rate           | > 5%       | > 20%      | Investigate ML serving    |
| Feature staleness       | > 2h       | > 6h       | Check Airflow pipeline    |
| Prediction drift        | KL > 0.1   | KL > 0.3   | Retrain model             |
