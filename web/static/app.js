const video = document.getElementById("video-feed");
const statusBanner = document.getElementById("status");
const orderList = document.getElementById("order-list");
const subtotalEl = document.getElementById("subtotal");
const payBtn = document.getElementById("pay-btn");

let order = [];
let itemPrices = {};

// Fetch prices from backend
const loadPrices = async () => {
  try {
    const response = await fetch("/prices");
    if (response.ok) {
      itemPrices = await response.json();
      console.log("Prices loaded:", itemPrices);
    }
  } catch (error) {
    console.error("Failed to load prices:", error);
  }
};

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
        <div class="item-unit-price">₱${item.price.toFixed(2)}</div>
        <div class="item-total-price">₱${(item.price * item.quantity).toFixed(2)}</div>
      </div>
    </div>
  `).join("");
};

// Update totals
const updateTotals = () => {
  const subtotal = order.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  
  subtotalEl.textContent = `₱${subtotal.toFixed(2)}`;
  payBtn.textContent = `Pay ₱${subtotal.toFixed(2)}`;
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
  
  console.log("Order details:", order);
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

// YOLO Detection polling
const addOrIncrementItem = (className, price) => {
  // Check if item already exists in order
  const existingItem = order.find(item => item.name === className);
  
  if (existingItem) {
    existingItem.quantity++;
  } else {
    // Add new item with actual price from detection or price lookup
    const itemPrice = price || itemPrices[className] || 0;
    const newId = order.length > 0 ? Math.max(...order.map(i => i.id)) + 1 : 1;
    order.push({
      id: newId,
      name: className,
      price: itemPrice,
      quantity: 1
    });
  }
  
  renderOrder();
  updateTotals();
};

const pollDetections = async () => {
  try {
    const response = await fetch("/detections");
    if (response.ok) {
      const detections = await response.json();
      
      // Add each detected item to the order with its price
      detections.forEach(detection => {
        console.log("Detection received:", detection);
        addOrIncrementItem(detection.class_name, detection.price);
      });
    }
  } catch (error) {
    console.error("Detection polling error:", error);
  }
};

// Poll detections every 500ms
setInterval(pollDetections, 500);

// Recording state indicator
const recordingIndicator = document.getElementById("recording-indicator");
const indicatorText = document.querySelector(".indicator-text");

const updateRecordingIndicator = (isRecording) => {
  if (isRecording) {
    recordingIndicator.classList.add("recording");
    recordingIndicator.classList.remove("paused");
    indicatorText.textContent = "RECORDING";
  } else {
    recordingIndicator.classList.add("paused");
    recordingIndicator.classList.remove("recording");
    indicatorText.textContent = "PAUSED";
  }
};

// Poll recording state
const pollRecordingState = async () => {
  try {
    const response = await fetch("/recording-state");
    if (response.ok) {
      const data = await response.json();
      updateRecordingIndicator(data.recording);
    }
  } catch (error) {
    console.error("Recording state polling error:", error);
  }
};

// Poll recording state every 300ms for responsiveness
setInterval(pollRecordingState, 300);

// Initialize
loadPrices();
renderOrder();
updateTotals();
showStatus("Initializing…");
pollRecordingState();
