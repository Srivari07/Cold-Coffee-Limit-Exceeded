# Feature Dictionary — CSAO Rail Recommendation System

> **Total columns**: 229 base (221 features + 6 meta + 2 labels) + 17 dynamic features at L2 stage = **246 total**
>
> **Sample context**: User U004006 — dinner-only, veg-preferring Delhi user with 18 orders, ordering from a North Indian mid-tier restaurant. Cart has 6 items (Rs.843), fully complete. Candidate item is an expensive (Rs.272) non-veg main course at position 5 in the rail, on fire sequence 3+.

---

## Meta Fields (6) — Identifiers, not used by model

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 1 | `interaction_id` | string | INT00678416 | Unique ID for this CSAO rail impression event |
| 2 | `order_id` | string | O0039395 | Order session this belongs to |
| 3 | `user_id` | string | U004006 | Customer identifier |
| 4 | `restaurant_id` | string | R00548 | Restaurant being ordered from |
| 5 | `item_id` | string | I010620 | Candidate add-on item being recommended in the rail |
| 6 | `position` | int | 5 | Position in the CSAO rail (1 = leftmost, most visible) |

---

## Group 1: User Features (34) — Who is this customer?

These features describe the customer's historical behavior, preferences, and engagement patterns.

### Behavioral History

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 7 | `user_order_count` | int | 18 | Total lifetime orders placed by this user |
| 8 | `last_order` | date | 2025-04-26 | Date of the user's most recent order (used to derive recency) |
| 9 | `monetary_avg` | float | 354.78 | Average order value (Rs.) across all orders — RFM monetary component |
| 10 | `avg_items_per_order` | float | 2.5 | Average number of items per order — indicates order complexity |
| 11 | `weekend_order_ratio` | float | 0.33 | Fraction of orders placed on weekends (Sat/Sun) |
| 12 | `recency_days` | int | 4 | Days since last order — lower = more recently active |
| 13 | `is_cold_start` | binary | 0 | 1 if user has < 3 orders (insufficient history), 0 otherwise |
| 14 | `frequency_30d` | int | 2 | Number of orders in last 30 days |
| 15 | `frequency_90d` | int | 8 | Number of orders in last 90 days |
| 16 | `rfm_segment` | int | 2 | RFM bucket (0=dormant, 1=at-risk, 2=moderate, 3=loyal, 4=champion) |
| 17 | `days_since_signup` | int | 221 | Days since account creation |

### Mealtime Preferences

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 18 | `breakfast_order_ratio` | float | 0.0 | Fraction of orders placed during breakfast hours (7-10 AM) |
| 19 | `lunch_order_ratio` | float | 0.0 | Fraction during lunch (12-3 PM) |
| 20 | `snack_order_ratio` | float | 0.0 | Fraction during snack time (3-6 PM) |
| 21 | `dinner_order_ratio` | float | 1.0 | Fraction during dinner (7-10 PM). **This user orders 100% at dinner** |
| 22 | `late_night_order_ratio` | float | 0.0 | Fraction during late night (10 PM-2 AM) |
| 23 | `mealtime_coverage` | float | 0.2 | How many mealtimes user is active in (1/5 = 0.2 for dinner-only) |

### Add-on & CSAO Engagement

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 24 | `addon_adoption_rate` | float | 0.6 | Fraction of orders where user added an item from CSAO rail |
| 25 | `price_sensitivity` | float | 2.78 | Coefficient of variation of order values. Higher = more price-conscious |
| 26 | `csao_historical_ctr` | float | 0.325 | Click-through rate on CSAO rail items historically (32.5%) |
| 27 | `csao_historical_accept_rate` | float | 0.125 | Fraction of shown items actually added to cart (12.5%) |

### Cuisine & Category Preferences

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 28 | `cuisine_pref_north_indian` | float | 0.28 | Fraction of orders from North Indian restaurants |
| 29 | `cuisine_pref_south_indian` | float | 0.0 | Fraction from South Indian |
| 30 | `cuisine_pref_chinese` | float | 0.22 | Fraction from Chinese |
| 31 | `cuisine_pref_biryani` | float | 0.11 | Fraction from Biryani specialists |
| 32 | `cuisine_pref_street_food` | float | 0.17 | Fraction from Street Food |
| 33 | `cuisine_pref_continental` | float | 0.11 | Fraction from Continental |
| 34 | `cuisine_pref_maharashtrian` | float | 0.11 | Fraction from Maharashtrian |
| 35 | `category_pref_main` | float | 0.56 | Fraction of items ordered that are main courses |
| 36 | `category_pref_side` | float | 0.20 | Fraction that are sides |
| 37 | `category_pref_drink` | float | 0.09 | Fraction that are drinks |
| 38 | `category_pref_dessert` | float | 0.07 | Fraction that are desserts |
| 39 | `category_pref_bread` | float | 0.02 | Fraction that are breads |
| 40 | `category_pref_unknown` | float | 0.07 | Fraction uncategorized |

### Dietary

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 41 | `veg_ratio` | float | 1.0 | Fraction of orders that are vegetarian. 1.0 = pure veg user |

---

## Group 2: Cart Features (54) — What's already in the cart?

These capture the current state of the user's cart at the moment the CSAO rail fires.

### Cart Composition

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 42 | `cart_item_count` | int | 6 | Number of items currently in the cart |
| 43 | `cart_total_value` | float | 843 | Total cart value in Rs. |
| 44 | `cart_avg_price` | float | 140.5 | Average price of items in cart |
| 45 | `cart_max_price` | float | 210.75 | Price of most expensive cart item |
| 46 | `cart_min_price` | float | 84.3 | Price of cheapest cart item |
| 47 | `cart_price_std` | float | 42.15 | Standard deviation of cart item prices |
| 48 | `cart_veg_ratio` | float | 1.0 | Fraction of cart items that are veg |

### Meal Completeness

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 49 | `cart_has_main` | binary | 1 | Cart contains at least one main course |
| 50 | `cart_has_side` | binary | 1 | Cart contains a side dish |
| 51 | `cart_has_drink` | binary | 1 | Cart contains a drink |
| 52 | `cart_has_dessert` | binary | 1 | Cart contains a dessert |
| 53 | `cart_has_bread` | binary | 1 | Cart contains a bread |
| 54 | `cart_completeness_score` | float | 1.0 | **Cuisine-aware** completeness. Uses MEAL_TEMPLATES: North Indian needs main+side+bread (3 required), all present → 1.0 |
| 55 | `cart_missing_components` | int | 0 | Number of missing required components. 0 = nothing missing |
| 56 | `cart_generic_completeness` | float | 1.0 | Generic completeness: just checks main+side+drink presence |

### Cart Budget & Pricing

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 57 | `cart_price_headroom` | float | 0.0 | `max(0, user_avg_order_value - cart_total_value)`. 0 = cart exceeds typical spend |
| 58 | `cart_price_tier_match` | binary | 0 | 1 if candidate item's price tier matches cart's average tier |
| 59 | `cart_cuisine_entropy` | float | 0.5 | Shannon entropy of cuisines in cart. Higher = more cuisine variety |

### Cart Sequence

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 60 | `cart_last_added_category_main` | binary | 1 | Last item added to cart was a main course |
| 61 | `cart_last_added_category_side` | binary | 0 | Last added was a side |
| 62 | `cart_last_added_category_drink` | binary | 0 | Last added was a drink |
| 63 | `cart_last_added_category_dessert` | binary | 0 | Last added was a dessert |
| 64 | `cart_last_added_category_bread` | binary | 0 | Last added was a bread |
| 65 | `cart_items_added_count` | int | 6 | Total items added to cart during this session (including from CSAO) |

### Cart Embeddings (30 dimensions)

| # | Fields | Type | Description |
|---|--------|------|-------------|
| 66-95 | `cart_emb_0` to `cart_emb_29` | float | 30-dimensional TF-IDF embedding of concatenated cart item names. Captures semantic meaning of what's in the cart. Proxy for LLM sentence-transformer embeddings. Sparse (many zeros) since TF-IDF. |

---

## Group 3: Item Features (48) — The candidate add-on being recommended

### Basic Item Properties

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 96 | `item_price` | float | 272 | Price of the candidate item in Rs. |
| 97 | `item_is_veg` | binary | 0 | 1 if vegetarian, 0 if non-veg |
| 98 | `item_is_bestseller` | binary | 0 | 1 if marked as bestseller at the restaurant |
| 99 | `item_price_vs_rest_avg` | float | 1.94 | Ratio: item_price / restaurant_avg_item_price. >1 means pricier than average |

### Item Popularity

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 100 | `item_order_count_7d` | int | 0 | Orders in last 7 days — short-term popularity |
| 101 | `item_order_count_30d` | int | 3 | Orders in last 30 days |
| 102 | `item_popularity_rank` | int | 8 | Rank among restaurant's items (1 = most popular) |
| 103 | `item_addon_order_rate` | float | 0.29 | Fraction of times this item is ordered as an add-on (vs. primary) |
| 104 | `item_popularity_velocity` | float | 0.0 | Trend: (7d_orders - 30d_avg_weekly) / max. Positive = trending up |

### Item Price Buckets (one-hot)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 105 | `item_price_bucket_low` | binary | 0 | 1 if price < Rs.100 |
| 106 | `item_price_bucket_mid` | binary | 0 | 1 if Rs.100-200 |
| 107 | `item_price_bucket_high` | binary | 1 | 1 if > Rs.200. **This item is in the high bucket** |

### Item Category (one-hot)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 108 | `item_cat_main` | binary | 1 | Item is a main course |
| 109 | `item_cat_side` | binary | 0 | Item is a side |
| 110 | `item_cat_drink` | binary | 0 | Item is a drink |
| 111 | `item_cat_dessert` | binary | 0 | Item is a dessert |
| 112 | `item_cat_bread` | binary | 0 | Item is a bread |
| 113 | `item_cat_unknown` | binary | 0 | Uncategorized |

### Item Embeddings (30 dimensions)

| # | Fields | Type | Description |
|---|--------|------|-------------|
| 114-143 | `item_emb_0` to `item_emb_29` | float | 30-dimensional TF-IDF embedding of the candidate item's name. Used to compute semantic similarity with cart items. Sparse vector. |

---

## Group 4: Restaurant Features (16) — Context about the restaurant

### Restaurant Quality & Type

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 144 | `rest_avg_rating` | float | 3.8 | Restaurant's average rating (1-5 scale) |
| 145 | `rest_is_chain` | binary | 0 | 1 if restaurant is a chain (McDonald's, Domino's), 0 if independent |

### Restaurant Cuisine (one-hot)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 146 | `rest_cuisine_north_indian` | binary | 1 | Restaurant serves North Indian cuisine |
| 147 | `rest_cuisine_south_indian` | binary | 0 | South Indian |
| 148 | `rest_cuisine_chinese` | binary | 0 | Chinese |
| 149 | `rest_cuisine_biryani` | binary | 0 | Biryani |
| 150 | `rest_cuisine_street_food` | binary | 0 | Street Food |
| 151 | `rest_cuisine_continental` | binary | 0 | Continental |
| 152 | `rest_cuisine_maharashtrian` | binary | 0 | Maharashtrian |

### Restaurant Pricing Tier (one-hot)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 153 | `rest_tier_budget` | binary | 0 | Budget tier restaurant |
| 154 | `rest_tier_mid` | binary | 1 | Mid-tier pricing |
| 155 | `rest_tier_premium` | binary | 0 | Premium tier |

### Restaurant Volume

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 156 | `rest_menu_size` | int | 22 | Total items on the menu |
| 157 | `rest_addon_depth` | int | 10 | Number of menu items suitable as add-ons |
| 158 | `rest_avg_aov` | float | 403.41 | Average order value at this restaurant |
| 159 | `rest_monthly_volume` | float | 36.25 | Average monthly order count |

---

## Group 5: Temporal & Geographic Features (18) — When and where

### Time Encoding

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 160 | `hour_sin` | float | -1.0 | Sine of hour: `sin(2π * hour/24)`. Cyclical encoding so 11 PM and 1 AM are close |
| 161 | `hour_cos` | float | ~0.0 | Cosine of hour: `cos(2π * hour/24)`. Together with sin, uniquely encodes hour |
| 162 | `dow_sin` | float | 0.78 | Sine of day-of-week: `sin(2π * dow/7)`. Cyclical: Sunday and Monday are close |
| 163 | `dow_cos` | float | 0.62 | Cosine of day-of-week |

### Mealtime (one-hot)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 164 | `meal_breakfast` | binary | 0 | 7-10 AM |
| 165 | `meal_lunch` | binary | 0 | 12-3 PM |
| 166 | `meal_snack` | binary | 0 | 3-6 PM |
| 167 | `meal_dinner` | binary | 1 | 7-10 PM. **This order is at dinner time** |
| 168 | `meal_late_night` | binary | 0 | 10 PM-2 AM |

### Day Type

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 169 | `is_weekend` | binary | 0 | 1 if Saturday or Sunday |
| 170 | `is_holiday` | binary | 0 | 1 if a public holiday |

### City (one-hot)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 171 | `city_mumbai` | binary | 0 | Order from Mumbai |
| 172 | `city_delhi` | binary | 1 | **Order from Delhi** |
| 173 | `city_bangalore` | binary | 0 | Bangalore |
| 174 | `city_hyderabad` | binary | 0 | Hyderabad |
| 175 | `city_pune` | binary | 0 | Pune |

### Zone-level Aggregates

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 176 | `zone_avg_aov` | float | 347.89 | Average order value in this delivery zone (locality-level) |
| 177 | `zone_addon_rate` | float | 0.15 | Fraction of orders in this zone that add CSAO items |

### Fire Sequence (Continuous Firing)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 178 | `fire_seq_is_first` | binary | 0 | 1 if this is the first rail firing (fire_sequence=1) |
| 179 | `fire_seq_is_late` | binary | 1 | 1 if fire_sequence >= 3 (late fires with higher fatigue) |
| 180 | `fire_seq_fatigue` | float | 0.53 | Fatigue factor: `1/(1 + 0.3*(fire_seq - 1))`. Decays from 1.0 (fire 1) toward 0.3 (fire 5). Models user attention decline |

---

## Group 6: Cross Features (37) — Interactions between feature groups

Cross features capture **relationships** between different entities (user-item, cart-item, item-context). These are the most powerful signals because they encode "how well does this specific item fit this specific user's current cart in this specific context?"

### PMI (Pointwise Mutual Information) — Co-purchase signals

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 181 | `cross_max_pmi` | float | 0.0 | Highest PMI between candidate and any cart item. PMI measures how often two items are bought together vs. independently. High PMI = strong pairing (e.g., Biryani+Raita). 0 = no co-purchase data |
| 182 | `cross_avg_pmi` | float | 0.0 | Average PMI across all cart items |
| 183 | `cross_min_pmi` | float | 0.0 | Lowest PMI with any cart item |
| 184 | `cross_has_high_pmi` | binary | 0 | 1 if max_pmi > threshold (strong co-purchase signal exists) |
| 185 | `cross_pmi_std` | float | 0.0 | Std dev of PMI values — measures consistency of pairing |

### Embedding Similarity

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 186 | `cross_embedding_similarity` | float | 0.57 | Cosine similarity between candidate item embedding and average cart embedding. Higher = more semantically similar names |

### Price Interactions

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 187 | `cross_price_ratio` | float | 1.94 | item_price / restaurant_avg_price. >1 = pricier than typical |
| 188 | `cross_price_vs_headroom` | float | 272 | Item price relative to remaining budget. High = over budget |
| 189 | `cross_fits_budget` | binary | 0 | 1 if adding this item keeps total under user's typical spend |

### Gap-filling Signals

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 190 | `cross_fills_missing_component` | binary | 0 | 1 if this item fills a missing meal component (e.g., cart has no drink and item is a drink). 0 here because cart is complete |
| 191 | `cross_same_category_in_cart` | binary | 0 | 1 if cart already has an item in the same category |

### User-Item Match

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 192 | `cross_user_category_pref` | float | 0.56 | User's historical preference for this item's category. 0.56 = user orders mains 56% of the time |
| 193 | `cross_veg_match` | binary | 0 | 1 if item veg status matches user preference. **0 = veg user shown non-veg item = mismatch** |
| 194 | `cross_position_in_rail` | int | 5 | Position in the rail — captures visibility bias (position 1 gets more clicks) |

### Composite Cross Features

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 195 | `cross_pmi_x_completeness` | float | 0.0 | PMI * completeness — "strongly paired AND fills a gap" |
| 196 | `cross_price_ratio_x_headroom` | float | 0.0 | Price ratio * headroom — "relatively cheap AND fits budget" |
| 197 | `cross_fills_gap_x_user_pref` | float | 0.0 | Fills gap * user pref — "fills missing component AND user likes this category" |
| 198 | `cross_popularity_x_dinner` | float | 3.0 | Popularity rank * dinner flag — popularity weighted by mealtime |
| 199 | `cross_veg_match_x_price` | float | 0.0 | Veg match * normalized price. 0 because veg mismatch kills the signal |
| 200 | `cross_bestseller_x_pmi` | float | 0.0 | Bestseller * PMI — "popular AND co-purchased" |
| 201 | `cross_cold_start_x_popularity` | float | 0.0 | Cold start * popularity — for new users, lean on item popularity |
| 202 | `cross_weekend_x_addon` | float | 0.0 | Weekend * addon rate — weekend indulgence effect |
| 203 | `cross_chain_x_bestseller` | float | 0.0 | Chain restaurant * bestseller — chains have stronger bestseller signals |
| 204 | `cross_mealtime_coverage_x_pmi` | float | 0.0 | Mealtime coverage * PMI — diverse users may respond differently to co-purchase |
| 205 | `cross_recency_x_popularity` | float | 0.75 | Recency * popularity — recent users engage more with popular items |
| 206 | `cross_frequency_x_price` | float | 3.87 | Order frequency * normalized price — frequent users tolerate higher prices |
| 207 | `cross_rfm_x_addon_rate` | float | 0.57 | RFM segment * item addon rate — loyal users + high addon items |
| 208 | `cross_cart_size_x_completeness` | float | 6.0 | Cart size * completeness — large complete carts behave differently |
| 209 | `cross_rest_rating_x_bestseller` | float | 0.0 | Restaurant rating * bestseller — trusted restaurants' bestsellers |
| 210 | `cross_rest_volume_x_popularity` | float | 1.09 | Restaurant volume * item popularity — busy restaurant's popular items |
| 211 | `cross_user_addon_x_item_addon` | float | 0.17 | User addon rate * item addon rate — addon-loving user + addon-friendly item |
| 212 | `cross_csao_ctr_x_position` | float | 0.065 | Historical CTR * position — high CTR users at good positions |
| 213 | `cross_price_sensitivity_x_item_price` | float | 2.52 | Price sensitivity * item price — sensitive users with expensive items |
| 214 | `cross_generic_comp_x_fills` | float | 0.0 | Generic completeness * fills gap — alternative completeness signal |

### Fatigue Cross Features (Continuous Firing)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 215 | `cross_fatigue_x_pmi` | float | 0.0 | Fatigue * PMI — even strong co-purchase signals weaken with fatigue |
| 216 | `cross_fatigue_x_popularity` | float | 1.58 | Fatigue * popularity — popular items get discounted by user tiredness |
| 217 | `cross_late_fire_x_bestseller` | binary | 0 | Late fire * bestseller — only bestsellers survive late fires |

---

## Group 7: LLM Features (10) — Semantic understanding via embeddings

These use TF-IDF as a proxy for LLM sentence-transformer embeddings (production would use `all-MiniLM-L6-v2` or similar).

### Semantic Similarity

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 218 | `llm_max_semantic_sim` | float | 0.57 | Max cosine similarity between candidate embedding and any cart item embedding. "How similar is this item to the most similar cart item?" |
| 219 | `llm_avg_semantic_sim` | float | 0.46 | Average similarity across all cart items |
| 220 | `llm_min_semantic_sim` | float | 0.28 | Minimum similarity — the most dissimilar cart item |
| 221 | `llm_complementarity_score` | float | 0.28 | `1 - max_similarity`. Lower = more similar (less complementary). Higher = more different (more complementary). Used to recommend items that fill gaps rather than duplicates |
| 222 | `llm_name_overlap` | float | 0.17 | Word-level overlap between candidate name and cart item names (Jaccard-like). 17% overlap in this case |

### Embedding Norms

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 223 | `llm_cand_emb_norm` | float | 1.0 | L2 norm of candidate item's embedding vector |
| 224 | `llm_cart_emb_norm` | float | 1.0 | L2 norm of average cart embedding vector |
| 225 | `llm_emb_norm_ratio` | float | 1.0 | Ratio of norms. Measures relative "richness" — items with longer names have larger norms |

### LLM Composite Features

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 226 | `llm_comp_x_pmi` | float | 0.0 | Complementarity * PMI — "semantically different AND frequently co-purchased" = strong signal |
| 227 | `llm_sim_x_fills_gap` | float | 0.0 | Similarity * fills_gap — "semantically related to cart AND fills a missing component" |

---

## Labels (2) — Target variables

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 228 | `label` | binary | 0 | **Primary target**: 1 if user added this item to cart, 0 otherwise. Used for LambdaRank training |
| 229 | `click` | binary | 1 | **Secondary signal**: 1 if user clicked/tapped on the item (even if not added). Used for CTR analysis. In this sample: user clicked but didn't add = browsed but rejected |

### Label Combinations

| label | click | Meaning | Example |
|-------|-------|---------|---------|
| 0 | 0 | Ignored — user didn't notice or wasn't interested | Item scrolled past |
| 0 | 1 | **Clicked but rejected** — user was curious but didn't convert | Checked price/description, said no |
| 1 | 1 | **Converted** — user clicked and added to cart | Saw Raita, tapped, added |
| 1 | 0 | Quick-add without detailed click (rare) | One-tap add button |

---

---

## Session 4 — New Features Added by Model Training Pipeline

These features are **generated dynamically** during `04_model_training.py` and added to the train/val/test DataFrames. They are NOT in the original feature CSVs but are used by L2 models.

### Two-Tower Score (1 feature)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 230 | `two_tower_score` | float | 0.342 | Cosine similarity between query embedding (user+cart+context) and item embedding from the Two-Tower model. Used as an L1 candidate retrieval signal |

### SASRec Sequential Embeddings (16 features)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| 231-246 | `sequential_emb_0` .. `sequential_emb_15` | float | 0.087 | 16-dimensional context vector from the SASRec Transformer model. Encodes the sequential pattern of items added so far in the cart. Fed into DCN-v2 L2 to capture "what item comes next" patterns |

### IPS Weights (training only)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| — | `ips_weight` | float | 9.11 | Inverse Propensity Score weight for position debiasing. Computed as `1/P(click|position)`, capped at 95th percentile. Used as sample weight in LightGBM L1 training only (not a model feature) |

### L1/L2 Scores (pipeline intermediates)

| # | Field | Type | Sample | Description |
|---|-------|------|--------|-------------|
| — | `l1_score` | float | 0.428 | LightGBM L1 prediction score. Used to select top-30 candidates for L2 re-ranking |
| — | `l2_score` | float | 0.651 | DCN-v2 L2 prediction score. Used by MMR for diversity re-ranking |

---

## Feature Group Summary

| Group | Count | Purpose | Key Insight for Model |
|-------|-------|---------|----------------------|
| Meta | 6 | Identifiers | Dropped before training |
| User | 34 | Customer profile | "Who is ordering and what do they like?" |
| Cart | 54 | Current cart state | "What's already in the cart and what's missing?" |
| Item | 48 | Candidate item | "What are we recommending?" |
| Restaurant | 16 | Restaurant context | "Where are they ordering from?" |
| Temporal/Geo | 18 | When/where | "When and where is this happening?" |
| Cross | 37 | Feature interactions | "How well does THIS item fit THIS user's cart NOW?" |
| LLM | 10 | Semantic similarity | "Do the names/descriptions make sense together?" |
| Labels | 2 | Targets | "Did the user add it?" |
| **Base Total** | **229** | | From feature engineering |
| Two-Tower | 1 | Retrieval score | "How relevant is this item per embedding similarity?" |
| SASRec | 16 | Sequential context | "What item should come next in this cart sequence?" |
| **L2 Total** | **246** | | Available at DCN-v2 L2 stage |
