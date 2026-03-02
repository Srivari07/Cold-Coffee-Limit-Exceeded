# CSAO Rail Recommendation — Streaming Pipeline Architecture

## 1. Batch vs Streaming: Why We Need Both

| Aspect              | Batch (Current)                         | Streaming (Proposed)                         |
|---------------------|-----------------------------------------|----------------------------------------------|
| **Analogy**         | Daily newspaper                         | Twitter feed                                  |
| **Latency**         | Hours (Airflow DAG runs daily at 2 AM)  | Seconds (events processed in near-real-time) |
| **Data Freshness**  | Stale by up to 24 hours                 | Sub-second freshness                          |
| **Use Case**        | User RFM, item popularity, PMI matrix   | Cart state, session CTR, trending items       |
| **Cost**            | Lower (batch compute on schedule)       | Higher (always-on streaming cluster)          |
| **Complexity**      | Simpler (ETL pipelines)                 | Complex (exactly-once, ordering, state)       |
| **Error Handling**  | Retry full DAG                          | Dead-letter queue + replay                    |
| **Framework**       | Spark + Airflow                         | Kafka + Flink / Spark Structured Streaming    |

**Key Insight**: Just like you wouldn't wait for tomorrow's newspaper to check live cricket scores,
you shouldn't wait for the nightly batch job to know what's trending on Zomato right now.
The CSAO rail benefits from _both_ — batch features for stable user profiles and streaming
features for real-time cart context.

## 2. Apache Kafka Topic Design

### Topics

| Topic Name            | Partition Key   | Partitions | Retention | Schema                                       |
|-----------------------|-----------------|------------|-----------|----------------------------------------------|
| `cart_events`         | `user_id`       | 32         | 7 days    | `{user_id, item_id, action, restaurant_id, ts}` |
| `impression_events`   | `user_id`       | 32         | 7 days    | `{user_id, item_ids[], position, fired_at, ts}` |
| `order_events`        | `order_id`      | 16         | 30 days   | `{order_id, user_id, items[], total, ts}`       |
| `recommendation_log`  | `user_id`       | 16         | 30 days   | `{user_id, reco_ids[], scores[], latency_ms, ts}` |

### Partitioning Strategy
- Partition by `user_id` for cart and impression events to ensure ordering per user session
- Partition by `order_id` for order events to enable parallel consumption
- 32 partitions for high-volume topics (cart/impression ~50K events/sec at peak)
- 16 partitions for lower-volume topics (orders ~10K/sec at peak)

### Schema Registry
- Use Confluent Schema Registry with Avro schemas
- Backward-compatible schema evolution
- Schema validation at producer side

## 3. Apache Flink / Spark Streaming Jobs

### Job 1: Real-Time Cart State Tracker

```
Source: cart_events (Kafka)
    |
    v
[Keyed by user_id + session_id]
    |
    v
[Stateful Processing]
  - Maintain current cart items per session
  - Compute real-time cart features:
    * cart_total_price (running sum)
    * cart_item_count
    * cart_category_distribution
    * cart_completeness_score (meal completeness)
    * time_since_last_add (seconds)
    |
    v
Sink: Redis (feature store, TTL=2h)
      Key: user:{user_id}:cart_state
```

**Window**: Session window with 30-minute inactivity gap
**State TTL**: 2 hours (auto-expire stale sessions)
**Checkpoint**: Every 60 seconds to RocksDB + S3

### Job 2: Trending Items Detector

```
Source: cart_events + order_events (Kafka)
    |
    v
[Keyed by restaurant_id + item_id]
    |
    v
[Sliding Window: 30min window, 5min slide]
  - Count add-to-cart events per item
  - Count order completions per item
  - Compute trending score:
    score = (cart_adds * 0.3 + orders * 0.7) / time_window_hours
    |
    v
[Top-K Aggregation per restaurant]
  - Keep top 20 trending items per restaurant
    |
    v
Sink: Redis (feature store, TTL=1h)
      Key: restaurant:{restaurant_id}:trending
```

**Window**: Sliding window — 30 minutes wide, sliding every 5 minutes
**Use Case**: Boost trending items in candidate generation and ranking

### Job 3: Session CTR Aggregator

```
Source: impression_events + cart_events (Kafka)
    |
    v
[Keyed by user_id + session_id]
    |
    v
[Session Window: 30min gap]
  - Count impressions per session
  - Count clicks (add-to-cart) per session
  - Compute session-level CTR = clicks / impressions
  - Compute session-level position-weighted CTR
    |
    v
[Enrichment: join with user profile from Redis]
    |
    v
Sink: Redis (feature store, TTL=2h)
      Key: user:{user_id}:session_ctr
```

**Latency**: < 1 second from event to feature availability
**Use Case**: Real-time engagement signal for ranking model

### Job 4: Live Inventory Filter

```
Source: order_events (Kafka)
    |
    v
[Keyed by restaurant_id + item_id]
    |
    v
[Tumbling Window: 1 minute]
  - Track order velocity per item
  - Cross-reference with inventory API (async call)
  - Mark items as available/unavailable/low-stock
    |
    v
Sink: Redis (bloom filter for fast lookup)
      Key: restaurant:{restaurant_id}:inventory
```

**Window**: 1-minute tumbling window
**Use Case**: Filter out-of-stock items BEFORE candidate generation (saves ~15% wasted ranking compute)

## 4. Feature Store Integration

### Architecture: Feast (Batch) + Redis (Real-time)

```
+-------------------+       +-------------------+       +-------------------+
|   Batch Features  |       |  Stream Features  |       |  Serving Layer    |
|   (Feast + S3)    |       |  (Flink -> Redis) |       |  (Redis Cluster)  |
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
| User RFM          |  -->  | Cart state        |  -->  | Unified Feature   |
| Item popularity   |  |    | Session CTR       |  |    | Vector (200+)     |
| PMI matrix        |  |    | Trending items    |  |    |                   |
| Embeddings        |  |    | Live inventory    |  |    | Served in <5ms    |
| Historical CTR    |  |    | Real-time CTR     |  |    |                   |
+-------------------+  |    +-------------------+  |    +-------------------+
                       |                           |
                       +------ Daily Sync ---------+
                       |    (Feast materialize)    |
                       +------ Real-time Push -----+
                            (Flink sink)
```

### Feature Groups by Source

| Feature Group        | Source    | Refresh Rate     | Storage     | Retrieval Latency |
|----------------------|-----------|------------------|-------------|-------------------|
| User RFM             | Batch     | Daily (2 AM)     | Feast -> S3 | ~3ms (Redis)      |
| Item Popularity      | Batch     | Daily (2 AM)     | Feast -> S3 | ~2ms (Redis)      |
| PMI Matrix           | Batch     | Daily (2 AM)     | Feast -> S3 | ~3ms (Redis)      |
| Embeddings           | Batch     | Weekly           | Feast -> S3 | ~4ms (Redis)      |
| Cart State           | Streaming | Real-time (<1s)  | Redis       | ~1ms (Redis)      |
| Session CTR          | Streaming | Real-time (<1s)  | Redis       | ~1ms (Redis)      |
| Trending Items       | Streaming | 5-min slide      | Redis       | ~1ms (Redis)      |
| Inventory Status     | Streaming | 1-min tumble     | Redis       | ~1ms (Redis)      |

## 5. Data Flow Diagram (End-to-End)

```
                              ┌──────────────────────────────────────────────────────────────┐
                              │                    DATA SOURCES                                │
                              │                                                                │
 User App ──────┐             │   ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
                │             │   │ Cart      │   │Impression│   │ Order    │                 │
                ▼             │   │ Events   │   │ Events   │   │ Events   │                 │
         ┌──────────┐        │   └────┬─────┘   └────┬─────┘   └────┬─────┘                 │
         │  API     │        │        │               │              │                        │
         │ Gateway  │        │        ▼               ▼              ▼                        │
         └────┬─────┘        │   ┌─────────────────────────────────────────┐                  │
              │              │   │            APACHE KAFKA                  │                  │
              │              │   │  cart_events | impression_events |       │                  │
              │              │   │  order_events | recommendation_log       │                  │
              │              │   └───────────┬──────────────────────────────┘                  │
              │              │               │                                                  │
              │              │               ▼                                                  │
              │              │   ┌───────────────────────────────────────────┐                 │
              │              │   │          APACHE FLINK CLUSTER             │                 │
              │              │   │                                           │                 │
              │              │   │  ┌──────────┐  ┌──────────┐             │                 │
              │              │   │  │Cart State│  │Trending  │             │                 │
              │              │   │  │ Tracker  │  │ Detector │             │                 │
              │              │   │  └────┬─────┘  └────┬─────┘             │                 │
              │              │   │       │             │                    │                 │
              │              │   │  ┌────┴─────┐  ┌────┴─────┐             │                 │
              │              │   │  │Session   │  │Inventory │             │                 │
              │              │   │  │CTR Agg   │  │ Filter   │             │                 │
              │              │   │  └────┬─────┘  └────┬─────┘             │                 │
              │              │   └───────┼─────────────┼───────────────────┘                  │
              │              │           │             │                                       │
              │              │           ▼             ▼                                       │
              │              │   ┌───────────────────────────────────────────┐                 │
              │              │   │        REDIS CLUSTER (Feature Store)      │                 │
              │              │   │                                           │                 │
              │              │   │  Batch Features    │  Stream Features    │                 │
              │              │   │  (User RFM, PMI,   │  (Cart state,       │                 │
              │              │   │   Embeddings)       │   Session CTR,     │                 │
              │              │   │                     │   Trending,        │                 │
              │              │   │   Updated daily     │   Inventory)       │                 │
              │              │   │   via Feast         │   Updated <1s     │                 │
              │              │   └──────────┬──────────────────────────────┘                  │
              │              │              │                                                  │
              │              └──────────────┼──────────────────────────────────────────────────┘
              │                             │
              ▼                             ▼
     ┌────────────────────────────────────────────┐
     │           ML SERVING PIPELINE               │
     │                                              │
     │  Feature         Candidate      L1 Ranking  │
     │  Retrieval  -->  Generation --> (LightGBM)  │
     │  (~18ms)         (~10ms)        (~22ms)     │
     │                                    |         │
     │                              L2 Ranking     │
     │                              (DCN-v2)       │
     │                              (~30ms)        │
     │                                    |         │
     │                              MMR Rerank     │
     │                              (~6ms)         │
     └─────────────────────┬──────────────────────┘
                           │
                           ▼
                    Response to User
                    (Total P95 < 200ms)
```

## 6. Latency Impact of Streaming Features

| Component                  | Without Streaming | With Streaming | Delta   |
|----------------------------|-------------------|----------------|---------|
| Feature Retrieval          | ~18ms             | ~23ms          | +5ms    |
| Candidate Generation       | ~10ms             | ~8ms           | -2ms    |
| Overall P95                | ~142ms            | ~145ms         | **+3ms**|

**Net impact**: Streaming features add approximately **+5ms** to feature retrieval
(additional Redis lookups for cart state, session CTR, trending items), but _save_ ~2ms
in candidate generation (live inventory filtering reduces candidate set size).

**Net P95 increase: ~3ms** — well within the 200ms budget.

### Why the tradeoff is worth it:
- **Cart state features** improve NDCG@5 by ~0.02 (real-time cart context)
- **Trending items** boost HR@5 by ~0.01 (social proof signal)
- **Session CTR** improves personalization for repeat-fire sequences
- **Inventory filter** eliminates out-of-stock recommendations (better UX, no wasted ranking)

## 7. Failure Modes & Recovery

| Failure Scenario             | Impact                        | Recovery Strategy                           |
|------------------------------|-------------------------------|---------------------------------------------|
| Kafka broker down            | Events buffered at producer   | Multi-broker cluster (3+), auto-rebalance   |
| Flink job crash              | Streaming features stale      | Checkpoint restore from RocksDB + S3        |
| Redis feature store down     | Feature retrieval fails       | Fallback to batch features (Feast offline)  |
| Schema mismatch              | Events rejected               | Schema Registry validation + DLQ            |
| Network partition            | Partial event loss            | At-least-once delivery + idempotent sinks   |

## 8. Deployment & Operations

### Infrastructure Requirements
| Component         | Instances | Spec          | Notes                                |
|-------------------|-----------|---------------|--------------------------------------|
| Kafka Cluster     | 3 brokers | m5.2xlarge    | 32 partitions, 7-day retention       |
| Flink Cluster     | 4 TMs     | c5.2xlarge    | 2 slots each, RocksDB state backend  |
| Schema Registry   | 2         | t3.medium     | HA pair                              |
| Monitoring        | 1         | m5.xlarge     | Prometheus + Grafana + PagerDuty     |

### Monitoring
- **Consumer lag**: Alert if > 10,000 messages behind
- **Processing latency**: Alert if P95 > 5 seconds
- **Checkpoint duration**: Alert if > 60 seconds
- **State size**: Alert if > 80% of RocksDB allocation
