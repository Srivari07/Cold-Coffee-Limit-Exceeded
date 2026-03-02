export interface Restaurant {
  restaurant_id: string;
  name: string;
  city: string;
  cuisine_type: string;
  price_tier: string;
  avg_rating: number;
  is_chain: boolean;
  delivery_time: string;
  emoji: string;
  item_count: number;
}

export interface MenuItem {
  item_id: string;
  restaurant_id: string;
  item_name: string;
  category: string;
  price: number;
  is_veg: boolean;
  is_bestseller: boolean;
}

export interface RestaurantDetail extends Restaurant {
  menu: Record<string, MenuItem[]>;
}

export interface CartItem {
  item_id: string;
  name: string;
  price: number;
  category: string;
  is_veg: boolean;
  quantity: number;
  source: 'menu' | 'recommendation';
}

export interface Recommendation {
  item_id: string;
  item_name: string;
  price: number;
  category: string;
  is_veg: boolean;
  score: number;
  reason: string;
}

export interface RecommendationResponse {
  recommendations: Recommendation[];
  cart_completeness: number;
  missing_components: string[];
  latency_ms: number;
}

export interface User {
  user_id: string;
  city: string;
  user_type: string;
  is_veg: boolean;
  avg_budget: number;
}

export interface Stats {
  total_restaurants: number;
  total_users: number;
  total_orders: number;
  total_menu_items: number;
  cities: string[];
  cuisines: string[];
}
