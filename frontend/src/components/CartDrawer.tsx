import type { CartItem, Recommendation, RecommendationResponse } from '../types';
import RecommendationRail from './RecommendationRail';

interface CartDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  cart: CartItem[];
  restaurantName: string;
  recommendations: RecommendationResponse | null;
  recLoading: boolean;
  onAddToCart: (item: { item_id: string; item_name: string; price: number; category: string; is_veg: boolean }, source?: 'menu' | 'recommendation') => void;
  onRemoveFromCart: (itemId: string) => void;
  onPlaceOrder: () => void;
  orderLoading: boolean;
}

export default function CartDrawer({
  isOpen,
  onClose,
  cart,
  restaurantName,
  recommendations,
  recLoading,
  onAddToCart,
  onRemoveFromCart,
  onPlaceOrder,
  orderLoading,
}: CartDrawerProps) {
  const cartTotal = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const completeness = recommendations?.cart_completeness ?? 0;
  const missing = recommendations?.missing_components ?? [];

  const completenessColor =
    completeness < 0.33 ? '#E23744' :
    completeness < 0.67 ? '#F4C150' : '#0F8A65';

  const handleRecAdd = (rec: Recommendation) => {
    onAddToCart({
      item_id: rec.item_id,
      item_name: rec.item_name,
      price: rec.price,
      category: rec.category,
      is_veg: rec.is_veg,
    }, 'recommendation');
  };

  const COMPONENT_EMOJI: Record<string, string> = {
    main: '🍛', bread: '🫓', side: '🥗', drink: '🥤', dessert: '🍮',
  };

  if (!isOpen) return null;

  return (
    <div className="cart-overlay" onClick={onClose}>
      <div className="cart-drawer" onClick={e => e.stopPropagation()}>
        <div className="cart-header">
          <div>
            <h2 className="cart-title">Your Cart</h2>
            <p className="cart-restaurant">{restaurantName}</p>
          </div>
          <button className="cart-close" onClick={onClose}>✕</button>
        </div>

        <div className="cart-body">
          {/* Cart items */}
          <div className="cart-items">
            {cart.map(item => (
              <div key={item.item_id} className="cart-item">
                <div className="cart-item-left">
                  <span className={`veg-indicator small ${item.is_veg ? 'veg' : 'non-veg'}`}>
                    <span className="veg-dot" />
                  </span>
                  <span className="cart-item-name">{item.name}</span>
                </div>
                <div className="cart-item-right">
                  <div className="qty-control small">
                    <button className="qty-btn" onClick={() => onRemoveFromCart(item.item_id)}>−</button>
                    <span className="qty-value">{item.quantity}</span>
                    <button className="qty-btn" onClick={() => onAddToCart(item)}>+</button>
                  </div>
                  <span className="cart-item-price">₹{item.price * item.quantity}</span>
                </div>
              </div>
            ))}
          </div>

          {/* CSAO Recommendation Rail */}
          <RecommendationRail
            recommendations={recommendations?.recommendations ?? []}
            latencyMs={recommendations?.latency_ms ?? 0}
            loading={recLoading}
            onAddItem={handleRecAdd}
          />

          {/* Cart completeness */}
          <div className="completeness-section">
            <div className="completeness-header">
              <span>Meal completeness</span>
              <span style={{ color: completenessColor, fontWeight: 600 }}>
                {Math.round(completeness * 100)}%
              </span>
            </div>
            <div className="completeness-bar">
              <div
                className="completeness-fill"
                style={{ width: `${completeness * 100}%`, backgroundColor: completenessColor }}
              />
            </div>
            {missing.length > 0 && (
              <p className="completeness-hint">
                Consider adding {missing.map(c => `${COMPONENT_EMOJI[c] || ''} ${c}`).join(', ')}
              </p>
            )}
          </div>
        </div>

        {/* Place order button */}
        <div className="cart-footer">
          <div className="cart-total">
            <span>Subtotal</span>
            <span className="total-amount">₹{cartTotal}</span>
          </div>
          <button
            className="place-order-btn"
            onClick={onPlaceOrder}
            disabled={orderLoading || cart.length === 0}
          >
            {orderLoading ? 'Placing Order...' : `Place Order — ₹${cartTotal}`}
          </button>
        </div>
      </div>
    </div>
  );
}
