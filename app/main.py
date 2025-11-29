from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera import CameraNotReadyError, USBCameraStream

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="Jetson USB Camera Stream", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = USBCameraStream()


@app.on_event("startup")
async def startup_event() -> None:
    camera.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    camera.stop()


def _frame_generator():
    while True:
        try:
            frame = camera.get_frame()
        except CameraNotReadyError:
            time.sleep(0.1)
            continue

        boundary = b"--frame\r\n"
        headers = b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
        yield boundary + headers + frame + b"\r\n"
        time.sleep(0.01)


@app.get("/", response_class=FileResponse)
async def read_index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html missing")
    return FileResponse(index_path)


@app.get("/video-stream")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        camera.get_frame()
    except CameraNotReadyError:
        return {"status": "warming_up"}
    return {"status": "ok"}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
