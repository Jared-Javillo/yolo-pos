# Jetson Orin USB Camera Kiosk Setup

The following steps provision a Jetson Orin so that, on every boot, a FastAPI server streams a USB camera (VGA resolution) and Chromium automatically launches in kiosk mode to display the live feed fullscreen.

## 1. Prerequisites

1. Jetson Orin running JetPack / Ubuntu 20.04 or newer with a graphical desktop.
2. USB camera connected and visible via `ls /dev/video*` (typically `/dev/video0`).
3. OS packages:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip curl git nano
   ```
   If Chromium is not available through APT on your image, install it from Snap instead:
   ```bash
   sudo apt install snapd
   sudo systemctl enable snapd
   sudo systemctl start snapd
   sudo snap install chromium
   ```
4. (Recommended) Add your desktop user (e.g. `jetson`) to the `video` group for camera access:
   ```bash
   sudo usermod -aG video jetson
   ```
5. Reboot once so group membership updates.

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

## 3. Quick manual test

1. Connect the USB camera.
2. Start the server manually:
   ```bash
   source .venv/bin/activate
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. On the Jetson, run `chromium-browser --app=http://127.0.0.1:8000` and verify the fullscreen image is shown.
4. Stop the server with `Ctrl+C`.

## 4. Make helper scripts executable

```bash
cd ~/camera-web-server
chmod +x scripts/start_server.sh scripts/start_kiosk.sh
```

## 5. Systemd service: camera stream

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

## 6. Auto-login (desktop session)

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

## 7. Systemd service: Chromium kiosk

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

## 8. Verification

1. Reboot the Jetson: `sudo reboot`.
2. The system should auto-login to the desktop, launch the FastAPI server, and open Chromium in kiosk mode pointed at `http://127.0.0.1:8000`.
3. If the page shows "Reconnecting…", inspect `journalctl -u camera-stream -u camera-kiosk` for troubleshooting.

## 9. Troubleshooting tips

- Confirm the USB camera works with `gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink`.
- Update resolution or frame rate in `app/camera.py` if your sensor prefers other modes.
- If Chromium complains about crashes, clear the profile directory under `~/.config/chromium`.
- For additional security, pin the kiosk services to an offline user account dedicated to the signage workload.
