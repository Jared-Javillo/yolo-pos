const video = document.getElementById("video-feed");
const statusBanner = document.getElementById("status");
const orderList = document.getElementById("order-list");
const subtotalEl = document.getElementById("subtotal");
const payBtn = document.getElementById("pay-btn");

let order = [];

// Render order list
const renderOrder = () => {
  if (order.length === 0) {
    orderList.innerHTML = '<div class="empty-order">No items added</div>';
    return;
  }
  
  orderList.innerHTML = order.map(item => `
    <div class="order-item" data-id="${item.id}">
      <div class="item-details">
        <div>
          <span class="item-qty">${item.quantity}</span>
          <span class="item-name">${item.name}</span>
        </div>
      </div>
      <div class="item-prices">
        <div class="item-unit-price">$${item.price.toFixed(2)}</div>
        <div class="item-total-price">$${(item.price * item.quantity).toFixed(2)}</div>
      </div>
    </div>
  `).join("");
};

// Update totals
const updateTotals = () => {
  const subtotal = order.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  
  subtotalEl.textContent = `$${subtotal.toFixed(2)}`;
  payBtn.textContent = `Pay $${subtotal.toFixed(2)}`;
};

// Event listeners
orderList.addEventListener("click", (e) => {
  const orderItem = e.target.closest(".order-item");
  if (orderItem) {
    const itemId = parseInt(orderItem.dataset.id);
    const item = order.find(i => i.id === itemId);
    if (item) {
      if (item.quantity > 1) {
        item.quantity--;
      } else {
        order = order.filter(i => i.id !== itemId);
      }
      renderOrder();
      updateTotals();
    }
  }
});

payBtn.addEventListener("click", () => {
  if (order.length === 0) return;
  
  alert(`Payment processed: ${payBtn.textContent}`);
  order = [];
  renderOrder();
  updateTotals();
});

// Camera stream management
const showStatus = (text) => {
  statusBanner.textContent = text;
  statusBanner.hidden = false;
};

const hideStatus = () => {
  statusBanner.hidden = true;
};

const reloadStream = () => {
  const cacheBuster = Date.now();
  video.src = `/video-stream?cb=${cacheBuster}`;
};

video.addEventListener("error", () => {
  showStatus("Reconnecting…");
  setTimeout(reloadStream, 1500);
});

video.addEventListener("load", hideStatus);

document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "r") {
    showStatus("Reloading feed…");
    reloadStream();
  }
});

// Initialize
renderOrder();
updateTotals();
showStatus("Initializing…");
