# Jetson Orin YOLO POS System Setup

This guide provisions a Jetson Orin to run a real-time YOLO-based Point of Sale system with:
- YOLOv8 object detection for 6 product categories
- Hand gesture control (palm→fist to start/stop recording)
- Automatic item detection and order management
- FastAPI backend with GPU acceleration
- Chromium kiosk mode fullscreen display

## 1. Prerequisites

1. **Jetson Orin** running JetPack 5.x or 6.x (Ubuntu 20.04/22.04) with graphical desktop.
2. **USB camera** connected and visible via `ls /dev/video*` (typically `/dev/video0`).
3. **YOLO model file** `model/best.pt` present in repository (6 product classes trained).
4. **Minimum 8GB RAM** recommended for YOLO + MediaPipe.

### Install System Packages:
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl git nano
```

### Install Chromium:
If Chromium is not available through APT:
```bash
sudo apt install snapd
sudo systemctl enable snapd
sudo systemctl start snapd
sudo snap install chromium
```

### Camera Permissions:
Add your user to the `video` group:
```bash
sudo usermod -aG video $USER
```

### Verify GPU Access:
Check CUDA is available:
```bash
jtop  # Install with: sudo -H pip3 install jetson-stats
# or
nvcc --version
```

Reboot after group/permission changes:
```bash
sudo reboot
```

## 2. Project deployment

```bash
cd ~
git clone https://github.com/<your-org>/camera-web-server.git
cd camera-web-server
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Verify GPU Setup

Before starting the server, verify GPU acceleration:

```bash
source .venv/bin/activate
python check_gpu.py
```

**Expected output:**
```
CUDA available: True
GPU device: Orin
CUDA version: 11.4 (or 12.x depending on JetPack)
✅ GPU is available and ready for use!
```

If GPU is not detected:
- Check JetPack includes PyTorch with CUDA support
- Install compatible PyTorch: See [NVIDIA PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

## 4. Quick Manual Test

### Start the Server:
```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Watch startup logs for:**
```
CUDA available: True
GPU device: Orin
YOLO model loaded on GPU
```

### Test the Interface:
1. Open browser: `chromium-browser --app=http://127.0.0.1:8000`
2. Verify video feed appears
3. Check top-right corner shows "PAUSED" indicator

### Test Hand Gestures:
1. **Start recording**: Show open palm ✋, then close to fist ✊
   - Indicator should change to red "RECORDING"
2. **Stop recording**: Repeat palm→fist gesture
   - Indicator should change to gray "PAUSED"

### Test Product Detection:
1. With recording active, show trained products to camera:
   - c2_na_green (C2 Green Tea)
   - colgate_toothpaste
   - 555-sardines
   - meadows-truffle
   - double-black
   - NongshimCupNoodles
2. Items should appear in left panel order list
3. Each item costs $1.00
4. 2-second cooldown between same item detections

### Stop Server:
```bash
Ctrl+C
```

## 5. Make helper scripts executable

```bash
cd ~/camera-web-server
chmod +x scripts/start_server.sh scripts/start_kiosk.sh
```

## 6. Systemd service: camera stream

Open the service file as root:

```bash
sudo nano /etc/systemd/system/camera-stream.service
```

Paste the following definition (update `User`/paths if needed):

```
[Unit]
Description=FastAPI USB camera stream
After=network-online.target
Wants=network-online.target

[Service]
User=jetson
WorkingDirectory=/home/jetson/camera-web-server
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/jetson/camera-web-server/scripts/start_server.sh
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now camera-stream.service
sudo systemctl status camera-stream.service
```

## 7. Auto-login (desktop session)

To ensure the kiosk launches on the desktop, enable automatic login for your user (example assumes GNOME / gdm3):

```bash
sudo nano /etc/gdm3/custom.conf
```
Uncomment / add:
```
[daemon]
AutomaticLoginEnable = true
AutomaticLogin = jetson
```
Save, then reboot once to confirm the system goes straight to the desktop session.

## 8. Systemd service: Chromium kiosk

Create / edit the kiosk service via:

```bash
sudo nano /etc/systemd/system/camera-kiosk.service
```

Insert this definition:

```
[Unit]
Description=Chromium kiosk for camera feed
After=graphical.target camera-stream.service
Requires=camera-stream.service

[Service]
User=jetson
WorkingDirectory=/home/jetson/camera-web-server
Environment=APP_URL=http://127.0.0.1:8000
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/jetson/.Xauthority
ExecStart=/home/jetson/camera-web-server/scripts/start_kiosk.sh
LimitMEMLOCK=infinity
LimitNOFILE=65535
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now camera-kiosk.service
sudo systemctl status camera-kiosk.service
```

## 9. System Verification

### Reboot and Test:
```bash
sudo reboot
```

**Expected behavior:**
1. System auto-logins to desktop
2. FastAPI server starts automatically
3. Chromium opens fullscreen at `http://127.0.0.1:8000`
4. Video feed displays with "PAUSED" indicator
5. Hand gestures control recording state
6. Products are detected and added to order when recording

### Check Service Status:
```bash
# Camera stream service
sudo systemctl status camera-stream

# Kiosk service
sudo systemctl status camera-kiosk

# View logs
journalctl -u camera-stream -f
journalctl -u camera-kiosk -f
```

### Verify GPU Usage:
```bash
# Monitor GPU in real-time
sudo tegrastats

# Or use jtop
sudo jtop
```

**Expected GPU usage:**
- GPU utilization: 30-60% during YOLO inference
- Memory: 1-2GB for models and buffers
- Temperature: Monitor stays under thermal limits

## 10. Troubleshooting

### Camera Issues:
```bash
# Test camera directly
gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink

# List video devices
ls -la /dev/video*

# Check camera permissions
groups $USER  # Should include 'video'
```

### GPU Not Detected:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify JetPack version
sudo apt-cache show nvidia-jetpack

# Check GPU with jtop
sudo jtop
```

### YOLO Model Issues:
```bash
# Verify model exists
ls -lh model/best.pt

# Check model loads correctly
python -c "from ultralytics import YOLO; m=YOLO('model/best.pt'); print('Model loaded')"
```

### Hand Gesture Not Working:
- Ensure good lighting conditions
- Hand should be clearly visible to camera
- Try adjusting `min_detection_confidence` in `camera.py` (lower = more sensitive)
- Check MediaPipe logs for hand detection

### Performance Issues:
```bash
# Reduce inference frequency in camera.py:
# Change: inference_interval=5 to inference_interval=10

# Monitor resource usage
sudo tegrastats
htop
```

### Service Failures:
```bash
# Restart services
sudo systemctl restart camera-stream
sudo systemctl restart camera-kiosk

# Check detailed logs
journalctl -u camera-stream --no-pager -n 100
journalctl -u camera-kiosk --no-pager -n 100
```

### Chromium Issues:
```bash
# Clear Chromium cache
rm -rf ~/.config/chromium

# Test Chromium manually
chromium-browser --app=http://127.0.0.1:8000
```

### Detection Logs:
Check detection activity:
```bash
# View detection log file
tail -f logs/detections.log
```

## 11. Performance Optimization

### Adjust YOLO Settings:
Edit `app/camera.py` to tune performance:
```python
# Reduce inference frequency (CPU savings)
inference_interval=10  # Default: 5

# Adjust confidence threshold
confidence_threshold=0.6  # Default: 0.7 (lower = more detections)

# Change cooldown period
cooldown_seconds=1.5  # Default: 2.0 (faster re-detection)
```

### Memory Management:
```bash
# Enable swap if needed
sudo systemctl enable nvzramconfig

# Monitor memory
free -h
```

### Power Mode:
```bash
# Set to maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 12. Security Considerations

- Create dedicated user for POS services (not `jetson`)
- Restrict SSH access
- Enable firewall for port 8000
- Regularly update JetPack and dependencies
- Backup YOLO model and configuration files

## 13. Product Customization

### Add New Products:
1. Retrain YOLO model with new product images
2. Export to `model/best.pt`
3. Replace existing model file
4. Restart service: `sudo systemctl restart camera-stream`

### Adjust Pricing:
Modify `web/static/app.js`:
```javascript
// Change from:
price: 1.00

// To custom pricing per product:
price: productPrices[className] || 1.00
```
