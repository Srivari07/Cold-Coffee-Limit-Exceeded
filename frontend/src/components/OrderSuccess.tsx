interface OrderSuccessProps {
  orderId: string;
  onClose: () => void;
}

export default function OrderSuccess({ orderId, onClose }: OrderSuccessProps) {
  return (
    <div className="cart-overlay" onClick={onClose}>
      <div className="order-success" onClick={e => e.stopPropagation()}>
        <div className="success-icon">🎉</div>
        <h2 className="success-title">Order Placed!</h2>
        <p className="success-order-id">Order ID: {orderId}</p>
        <p className="success-message">Your food is being prepared</p>
        <button className="success-btn" onClick={onClose}>
          Continue Browsing
        </button>
      </div>
    </div>
  );
}
