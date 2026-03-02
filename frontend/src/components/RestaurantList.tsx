import { useEffect, useState } from 'react';
import { fetchRestaurants } from '../api';
import type { Restaurant } from '../types';
import FilterPills from './FilterPills';
import RestaurantCard from './RestaurantCard';

interface RestaurantListProps {
  city: string;
  cuisines: string[];
  onSelectRestaurant: (id: string) => void;
}

export default function RestaurantList({ city, cuisines, onSelectRestaurant }: RestaurantListProps) {
  const [restaurants, setRestaurants] = useState<Restaurant[]>([]);
  const [selectedCuisine, setSelectedCuisine] = useState('');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchRestaurants(city, selectedCuisine || undefined, search || undefined)
      .then(setRestaurants)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [city, selectedCuisine, search]);

  return (
    <div className="restaurant-list">
      <div className="search-bar">
        <svg className="search-icon" width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#9C9C9C" strokeWidth="2">
          <circle cx="8" cy="8" r="6" />
          <path d="M13 13l4 4" />
        </svg>
        <input
          type="text"
          placeholder="Search restaurants..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="search-input"
        />
      </div>

      <FilterPills
        items={cuisines}
        selected={selectedCuisine}
        onSelect={setSelectedCuisine}
        allLabel="All Cuisines"
      />

      {loading ? (
        <div className="loading">
          {[1, 2, 3].map(i => (
            <div key={i} className="skeleton-card">
              <div className="skeleton-image shimmer" />
              <div className="skeleton-text shimmer" />
              <div className="skeleton-text short shimmer" />
            </div>
          ))}
        </div>
      ) : (
        <div className="cards-container">
          <p className="results-count">{restaurants.length} restaurants</p>
          {restaurants.map(r => (
            <RestaurantCard
              key={r.restaurant_id}
              restaurant={r}
              onClick={() => onSelectRestaurant(r.restaurant_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
