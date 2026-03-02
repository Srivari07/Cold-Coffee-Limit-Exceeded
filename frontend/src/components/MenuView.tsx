import { useEffect, useState } from 'react';
import { fetchRestaurant } from '../api';
import type { CartItem, MenuItem, RestaurantDetail } from '../types';

interface MenuViewProps {
  restaurantId: string;
  cart: CartItem[];
  onAddToCart: (item: MenuItem) => void;
  onRemoveFromCart: (itemId: string) => void;
  onOpenCart: () => void;
  onRestaurantLoaded?: (name: string) => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  main: 'Mains',
  bread: 'Breads',
  side: 'Sides',
  drink: 'Drinks',
  dessert: 'Desserts',
};

export default function MenuView({ restaurantId, cart, onAddToCart, onRemoveFromCart, onOpenCart, onRestaurantLoaded }: MenuViewProps) {
  const [restaurant, setRestaurant] = useState<RestaurantDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchRestaurant(restaurantId)
      .then(data => {
        setRestaurant(data);
        onRestaurantLoaded?.(data.name);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [restaurantId]);

  const getCartQuantity = (itemId: string) => {
    const item = cart.find(c => c.item_id === itemId);
    return item?.quantity || 0;
  };

  const cartTotal = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const cartCount = cart.reduce((sum, item) => sum + item.quantity, 0);

  if (loading || !restaurant) {
    return (
      <div className="menu-view">
        <div className="skeleton-banner shimmer" />
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="skeleton-menu-item shimmer" />
        ))}
      </div>
    );
  }

  return (
    <div className="menu-view">
      {/* Restaurant info banner */}
      <div className="restaurant-banner">
        <h2 className="banner-name">{restaurant.name}</h2>
        <div className="banner-meta">
          <span className="banner-cuisine">{restaurant.cuisine_type}</span>
          <span className="banner-dot">·</span>
          <span className="banner-price">{restaurant.price_tier}</span>
          <span className="banner-dot">·</span>
          <span className="banner-rating">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="#F4C150">
              <path d="M6 0l1.76 3.57L12 4.14 8.88 7.1l.74 4.32L6 9.27 2.38 11.42l.74-4.32L0 4.14l4.24-.57z" />
            </svg>
            {restaurant.avg_rating}
          </span>
          <span className="banner-dot">·</span>
          <span className="banner-time">{restaurant.delivery_time}</span>
        </div>
        <span className="banner-city">{restaurant.city}</span>
      </div>

      {/* Menu sections */}
      <div className="menu-sections">
        {Object.entries(restaurant.menu).map(([category, items]) => (
          <div key={category} className="menu-section">
            <h3 className="section-header">{CATEGORY_LABELS[category] || category}</h3>
            {items.map((item: MenuItem) => {
              const qty = getCartQuantity(item.item_id);
              return (
                <div key={item.item_id} className="menu-item">
                  <div className="menu-item-left">
                    <span className={`veg-indicator ${item.is_veg ? 'veg' : 'non-veg'}`}>
                      <span className="veg-dot" />
                    </span>
                    <div className="menu-item-info">
                      <span className="menu-item-name">
                        {item.item_name}
                        {item.is_bestseller && <span className="bestseller-badge">Bestseller</span>}
                      </span>
                      <span className="menu-item-price">₹{item.price}</span>
                    </div>
                  </div>
                  <div className="menu-item-right">
                    {qty === 0 ? (
                      <button className="add-btn" onClick={() => onAddToCart(item)}>ADD</button>
                    ) : (
                      <div className="qty-control">
                        <button className="qty-btn" onClick={() => onRemoveFromCart(item.item_id)}>−</button>
                        <span className="qty-value">{qty}</span>
                        <button className="qty-btn" onClick={() => onAddToCart(item)}>+</button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Floating cart bar */}
      {cartCount > 0 && (
        <div className="floating-cart-bar" onClick={onOpenCart}>
          <span className="cart-bar-info">
            {`${cartCount} item${cartCount > 1 ? 's' : ''} | ₹${cartTotal}`}
          </span>
          <span className="cart-bar-action">View Cart →</span>
        </div>
      )}
    </div>
  );
}
