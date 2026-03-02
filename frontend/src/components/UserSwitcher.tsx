import { useState } from 'react';
import type { User } from '../types';

interface UserSwitcherProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: User | null;
  onSwitch: (userId: string) => void;
}

const DEMO_USERS = [
  { id: 'U000000', desc: 'Non-veg, Delhi, Budget ₹430' },
  { id: 'U000001', desc: 'Veg, Pune, Budget ₹233' },
  { id: 'U000002', desc: 'Non-veg, Bangalore, Budget ₹359' },
  { id: 'U000010', desc: 'Regular user' },
  { id: 'U000050', desc: 'Power user' },
];

export default function UserSwitcher({ isOpen, onClose, currentUser, onSwitch }: UserSwitcherProps) {
  const [customId, setCustomId] = useState('');

  if (!isOpen) return null;

  return (
    <div className="cart-overlay" onClick={onClose}>
      <div className="user-switcher" onClick={e => e.stopPropagation()}>
        <div className="user-switcher-header">
          <h3>Switch Demo User</h3>
          <button className="cart-close" onClick={onClose}>✕</button>
        </div>
        {currentUser && (
          <div className="current-user-info">
            <span className="user-label">Current:</span>
            <span>{currentUser.user_id}</span>
            <span className={`veg-indicator small ${currentUser.is_veg ? 'veg' : 'non-veg'}`}>
              <span className="veg-dot" />
            </span>
            <span>{currentUser.city}</span>
            <span>₹{currentUser.avg_budget}</span>
          </div>
        )}
        <div className="user-list">
          {DEMO_USERS.map(u => (
            <button
              key={u.id}
              className={`user-option ${currentUser?.user_id === u.id ? 'active' : ''}`}
              onClick={() => { onSwitch(u.id); onClose(); }}
            >
              <span className="user-option-id">{u.id}</span>
              <span className="user-option-desc">{u.desc}</span>
            </button>
          ))}
        </div>
        <div className="custom-user">
          <input
            type="text"
            placeholder="Enter user ID (e.g., U12345)"
            value={customId}
            onChange={e => setCustomId(e.target.value)}
            className="custom-user-input"
          />
          <button
            className="custom-user-btn"
            onClick={() => { if (customId) { onSwitch(customId); onClose(); } }}
            disabled={!customId}
          >
            Switch
          </button>
        </div>
      </div>
    </div>
  );
}
