import type { CartItem, RecommendationResponse, Restaurant, RestaurantDetail, Stats, User } from './types';

const API_BASE = 'http://localhost:8000/api';

export async function fetchRestaurants(city?: string, cuisine?: string, search?: string): Promise<Restaurant[]> {
  const params = new URLSearchParams();
  if (city) params.set('city', city);
  if (cuisine) params.set('cuisine', cuisine);
  if (search) params.set('search', search);
  const url = `${API_BASE}/restaurants${params.toString() ? '?' + params : ''}`;
  const res = await fetch(url);
  return res.json();
}

export async function fetchRestaurant(id: string): Promise<RestaurantDetail> {
  const res = await fetch(`${API_BASE}/restaurants/${id}`);
  return res.json();
}

export async function fetchUser(userId: string): Promise<User> {
  const res = await fetch(`${API_BASE}/users/${userId}`);
  return res.json();
}

export async function fetchRecommendations(
  userId: string,
  restaurantId: string,
  cartItems: CartItem[],
  mealPeriod: string,
  city: string,
): Promise<RecommendationResponse> {
  const res = await fetch(`${API_BASE}/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      restaurant_id: restaurantId,
      cart_items: cartItems.map(i => ({
        item_id: i.item_id,
        name: i.name,
        price: i.price,
        category: i.category,
      })),
      meal_period: mealPeriod,
      city: city,
    }),
  });
  return res.json();
}

export async function placeOrder(
  userId: string,
  restaurantId: string,
  items: CartItem[],
  total: number,
  mealPeriod: string,
  city: string,
  recsShown: { item_id: string; position: number; score: number }[],
  recsAccepted: string[],
): Promise<{ order_id: string; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/orders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      restaurant_id: restaurantId,
      items: items.map(i => ({
        item_id: i.item_id,
        name: i.name,
        price: i.price,
        category: i.category,
        is_veg: i.is_veg,
        quantity: i.quantity,
        source: i.source,
      })),
      total,
      meal_period: mealPeriod,
      city,
      recs_shown: recsShown,
      recs_accepted: recsAccepted,
    }),
  });
  return res.json();
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/stats`);
  return res.json();
}
