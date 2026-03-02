import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchRecommendations, fetchStats, fetchUser, placeOrder } from './api';
import CartDrawer from './components/CartDrawer';
import Header from './components/Header';
import MenuView from './components/MenuView';
import OrderSuccess from './components/OrderSuccess';
import RestaurantList from './components/RestaurantList';
import UserSwitcher from './components/UserSwitcher';
import type { CartItem, MenuItem, RecommendationResponse, User } from './types';
import './App.css';

function getMealPeriod(): string {
  const hour = new Date().getHours();
  if (hour >= 6 && hour < 11) return 'breakfast';
  if (hour >= 11 && hour < 15) return 'lunch';
  if (hour >= 15 && hour < 18) return 'snack';
  return 'dinner';
}

export default function App() {
  // Navigation
  const [screen, setScreen] = useState<'list' | 'menu'>('list');
  const [selectedRestaurantId, setSelectedRestaurantId] = useState('');
  const [selectedRestaurantName, setSelectedRestaurantName] = useState('');

  // Filters
  const [selectedCity, setSelectedCity] = useState('Mumbai');
  const [cities, setCities] = useState<string[]>([]);
  const [cuisines, setCuisines] = useState<string[]>([]);

  // User
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [showUserSwitcher, setShowUserSwitcher] = useState(false);

  // Cart
  const [cart, setCart] = useState<CartItem[]>([]);
  const [isCartOpen, setIsCartOpen] = useState(false);

  // Track which item_ids were added from the recommendation rail
  const recsAcceptedRef = useRef<Set<string>>(new Set());

  // Recommendations
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const [recLoading, setRecLoading] = useState(false);

  // Order
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderSuccess, setOrderSuccess] = useState<string | null>(null);

  // Refs for latest values (avoids stale closures)
  const currentUserRef = useRef(currentUser);
  currentUserRef.current = currentUser;
  const selectedCityRef = useRef(selectedCity);
  selectedCityRef.current = selectedCity;

  // Load stats & default user on mount
  useEffect(() => {
    fetchStats().then(stats => {
      setCities(stats.cities);
      setCuisines(stats.cuisines);
    }).catch(console.error);

    fetchUser('U000000').then(setCurrentUser).catch(console.error);
  }, []);

  // Fetch recommendations — reads latest user/city from refs to avoid stale closures
  const fetchRecs = useCallback(async (currentCart: CartItem[], overrideUser?: User) => {
    const user = overrideUser || currentUserRef.current;
    const city = selectedCityRef.current;

    if (currentCart.length === 0 || !selectedRestaurantId) {
      setRecommendations(null);
      return;
    }
    if (!user) {
      console.warn('Skipping recommendations: no user loaded yet');
      setRecommendations(null);
      return;
    }
    setRecLoading(true);
    try {
      const res = await fetchRecommendations(
        user.user_id,
        selectedRestaurantId,
        currentCart,
        getMealPeriod(),
        city,
      );
      setRecommendations(res);
    } catch (err) {
      console.error('Recommendation fetch failed:', err);
    } finally {
      setRecLoading(false);
    }
  }, [selectedRestaurantId]);

  // Trigger recs on cart change
  useEffect(() => {
    fetchRecs(cart);
  }, [cart, fetchRecs]);

  const handleSelectRestaurant = (id: string) => {
    setSelectedRestaurantId(id);
    setScreen('menu');
    setCart([]);
    setRecommendations(null);
  };

  const handleAddToCart = (item: MenuItem | { item_id: string; item_name: string; price: number; category: string; is_veg: boolean }, source: 'menu' | 'recommendation' = 'menu') => {
    if (source === 'recommendation') {
      recsAcceptedRef.current.add(item.item_id);
    }
    setCart(prev => {
      const existing = prev.find(c => c.item_id === item.item_id);
      if (existing) {
        return prev.map(c =>
          c.item_id === item.item_id ? { ...c, quantity: c.quantity + 1 } : c
        );
      }
      return [...prev, {
        item_id: item.item_id,
        name: item.item_name,
        price: item.price,
        category: item.category,
        is_veg: item.is_veg,
        quantity: 1,
        source,
      }];
    });
  };

  const handleRemoveFromCart = (itemId: string) => {
    setCart(prev => {
      const existing = prev.find(c => c.item_id === itemId);
      if (!existing) return prev;
      if (existing.quantity <= 1) {
        return prev.filter(c => c.item_id !== itemId);
      }
      return prev.map(c =>
        c.item_id === itemId ? { ...c, quantity: c.quantity - 1 } : c
      );
    });
  };

  const handlePlaceOrder = async () => {
    if (!currentUser || cart.length === 0) return;
    setOrderLoading(true);
    try {
      const total = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);

      // Collect recommendation interaction data
      const recsShown = recommendations?.recommendations?.map((r, i) => ({
        item_id: r.item_id,
        position: i + 1,
        score: r.score,
      })) ?? [];
      const recsAccepted = Array.from(recsAcceptedRef.current);

      const result = await placeOrder(
        currentUser.user_id,
        selectedRestaurantId,
        cart,
        total,
        getMealPeriod(),
        selectedCity,
        recsShown,
        recsAccepted,
      );
      setOrderSuccess(result.order_id);
      setCart([]);
      setRecommendations(null);
      setIsCartOpen(false);
      recsAcceptedRef.current = new Set();
    } catch (err) {
      console.error('Order failed:', err);
    } finally {
      setOrderLoading(false);
    }
  };

  const handleBack = () => {
    setScreen('list');
    setCart([]);
    setRecommendations(null);
    setSelectedRestaurantId('');
    setSelectedRestaurantName('');
    recsAcceptedRef.current = new Set();
  };

  const handleUserSwitch = async (userId: string) => {
    try {
      const user = await fetchUser(userId);
      setCurrentUser(user);
      setSelectedCity(user.city);
      if (cart.length > 0) {
        // Pass new user directly to avoid stale closure
        fetchRecs(cart, user);
      }
    } catch {
      console.error('User not found');
    }
  };

  return (
    <div className="app">
      <Header
        cities={cities}
        selectedCity={selectedCity}
        onCityChange={setSelectedCity}
        currentUser={currentUser}
        onUserSwitch={() => setShowUserSwitcher(true)}
        onBack={screen === 'menu' ? handleBack : undefined}
        title={screen === 'menu' ? selectedRestaurantName || 'Menu' : undefined}
      />

      <main className="main-content">
        {screen === 'list' ? (
          <RestaurantList
            city={selectedCity}
            cuisines={cuisines}
            onSelectRestaurant={handleSelectRestaurant}
          />
        ) : (
          <MenuView
            restaurantId={selectedRestaurantId}
            cart={cart}
            onAddToCart={handleAddToCart}
            onRemoveFromCart={handleRemoveFromCart}
            onOpenCart={() => setIsCartOpen(true)}
            onRestaurantLoaded={setSelectedRestaurantName}
          />
        )}
      </main>

      <CartDrawer
        isOpen={isCartOpen}
        onClose={() => setIsCartOpen(false)}
        cart={cart}
        restaurantName={selectedRestaurantName}
        recommendations={recommendations}
        recLoading={recLoading}
        onAddToCart={handleAddToCart}
        onRemoveFromCart={handleRemoveFromCart}
        onPlaceOrder={handlePlaceOrder}
        orderLoading={orderLoading}
      />

      <UserSwitcher
        isOpen={showUserSwitcher}
        onClose={() => setShowUserSwitcher(false)}
        currentUser={currentUser}
        onSwitch={handleUserSwitch}
      />

      {orderSuccess && (
        <OrderSuccess
          orderId={orderSuccess}
          onClose={() => setOrderSuccess(null)}
        />
      )}
    </div>
  );
}
