import { useState } from 'react';
import type { User } from '../types';

interface HeaderProps {
  cities: string[];
  selectedCity: string;
  onCityChange: (city: string) => void;
  currentUser: User | null;
  onUserSwitch: () => void;
  onBack?: () => void;
  title?: string;
}

export default function Header({ cities, selectedCity, onCityChange, currentUser, onUserSwitch, onBack, title }: HeaderProps) {
  const [showCityDropdown, setShowCityDropdown] = useState(false);

  return (
    <header className="header">
      <div className="header-inner">
        {onBack ? (
          <button className="header-back" onClick={onBack}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
          </button>
        ) : null}

        {title ? (
          <h1 className="header-title">{title}</h1>
        ) : (
          <div className="header-logo">zomato</div>
        )}

        <div className="header-right">
          {!onBack && (
            <div className="city-selector" onClick={() => setShowCityDropdown(!showCityDropdown)}>
              <span className="city-name">{selectedCity}</span>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                <path d="M3 5l3 3 3-3" />
              </svg>
              {showCityDropdown && (
                <div className="city-dropdown">
                  {cities.map(c => (
                    <div
                      key={c}
                      className={`city-option ${c === selectedCity ? 'active' : ''}`}
                      onClick={(e) => { e.stopPropagation(); onCityChange(c); setShowCityDropdown(false); }}
                    >
                      {c}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <button className="user-badge" onClick={onUserSwitch} title="Switch user">
            <span className="user-icon">{currentUser?.is_veg ? '🟢' : '🔴'}</span>
            <span className="user-id">{currentUser?.user_id?.slice(-5) || 'Guest'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
