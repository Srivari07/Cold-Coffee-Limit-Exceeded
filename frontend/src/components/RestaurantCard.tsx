import type { Restaurant } from '../types';

interface RestaurantCardProps {
  restaurant: Restaurant;
  onClick: () => void;
}

const GRADIENT_COLORS: Record<string, string> = {
  'North Indian': 'linear-gradient(135deg, #FF6B35, #F7931A)',
  'South Indian': 'linear-gradient(135deg, #2E7D32, #66BB6A)',
  'Biryani': 'linear-gradient(135deg, #E65100, #FF9800)',
  'Chinese': 'linear-gradient(135deg, #C62828, #EF5350)',
  'Street Food': 'linear-gradient(135deg, #F9A825, #FFD54F)',
  'Maharashtrian': 'linear-gradient(135deg, #6A1B9A, #AB47BC)',
  'Continental': 'linear-gradient(135deg, #1565C0, #42A5F5)',
};

export default function RestaurantCard({ restaurant, onClick }: RestaurantCardProps) {
  const gradient = GRADIENT_COLORS[restaurant.cuisine_type] || GRADIENT_COLORS['North Indian'];

  return (
    <div className="restaurant-card" onClick={onClick}>
      <div className="restaurant-image" style={{ background: gradient }}>
        <span className="restaurant-emoji">{restaurant.emoji}</span>
        {restaurant.is_chain && <span className="chain-badge">Chain</span>}
      </div>
      <div className="restaurant-info">
        <h3 className="restaurant-name">{restaurant.name}</h3>
        <div className="restaurant-meta">
          <span className="cuisine-badge">{restaurant.cuisine_type}</span>
          <span className="price-badge">{restaurant.price_tier}</span>
        </div>
        <div className="restaurant-stats">
          <span className="rating">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="#F4C150">
              <path d="M6 0l1.76 3.57L12 4.14 8.88 7.1l.74 4.32L6 9.27 2.38 11.42l.74-4.32L0 4.14l4.24-.57z" />
            </svg>
            {restaurant.avg_rating}
          </span>
          <span className="delivery-time">{restaurant.delivery_time}</span>
          <span className="item-count">{restaurant.item_count} items</span>
        </div>
      </div>
    </div>
  );
}
