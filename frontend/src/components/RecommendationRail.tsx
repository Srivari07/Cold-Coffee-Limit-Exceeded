import type { Recommendation } from '../types';

interface RecommendationRailProps {
  recommendations: Recommendation[];
  latencyMs: number;
  loading: boolean;
  onAddItem: (rec: Recommendation) => void;
}

export default function RecommendationRail({ recommendations, latencyMs, loading, onAddItem }: RecommendationRailProps) {
  if (loading) {
    return (
      <div className="rec-rail">
        <div className="rec-rail-header">
          <span className="rec-rail-title">Customers also ordered ✨</span>
        </div>
        <div className="rec-rail-scroll">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="rec-card-skeleton shimmer" />
          ))}
        </div>
      </div>
    );
  }

  if (recommendations.length === 0) return null;

  return (
    <div className="rec-rail">
      <div className="rec-rail-header">
        <span className="rec-rail-title">Customers also ordered ✨</span>
        <span className="rec-rail-latency">Powered by ML · {latencyMs}ms</span>
      </div>
      <div className="rec-rail-scroll">
        {recommendations.map(rec => (
          <div key={rec.item_id} className="rec-card">
            <div className="rec-card-top">
              <span className={`veg-indicator small ${rec.is_veg ? 'veg' : 'non-veg'}`}>
                <span className="veg-dot" />
              </span>
              <span className="rec-category-badge">{rec.category}</span>
            </div>
            <span className="rec-item-name">{rec.item_name}</span>
            <span className="rec-item-price">₹{rec.price}</span>
            <span className="rec-match">{Math.round(rec.score * 100)}% match</span>
            <span className="rec-reason">{rec.reason}</span>
            <button className="rec-add-btn" onClick={() => onAddItem(rec)}>+ ADD</button>
          </div>
        ))}
      </div>
    </div>
  );
}
