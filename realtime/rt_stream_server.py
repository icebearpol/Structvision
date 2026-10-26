"""
STRUCTVISION - Real-Time Ingest & Live WebSocket HUD Stream Server.
Combines YOLO11-seg inference with GSDEngine to broadcast physical crack metrics.
"""

import asyncio
import json
import logging
import time
from typing import Set, Union

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from cv_pipeline.gsd_engine import GSDEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STRUCTVISION_RT")

app = FastAPI(title="STRUCTVISION Real-Time Engine")

# Active WebSocket HUD Clients
connected_clients: Set[WebSocket] = set()

# Initialize Models & Pipeline Engines - Targeting primary pipeline path
WEIGHTS_PATH = "cv_pipeline/models/structvision_crack_seg/weights/best.pt"

try:
    model = YOLO(WEIGHTS_PATH)
    logger.info(f"Successfully loaded custom crack weights from {WEIGHTS_PATH}")
except Exception:
    logger.warning(
        f"Custom crack model not found at {WEIGHTS_PATH}. Falling back to default yolo11n-seg.pt"
    )
    model = YOLO("yolo11n-seg.pt")

gsd_engine = GSDEngine(sensor_width_mm=13.2, focal_length_mm=8.8)  # Default 1" sensor parameters

# Active Drone State Telemetry (Injected via UDP/Telemetry API or default)
current_telemetry = {
    "altitude_m": 15.0,  # Default flight altitude in meters
    "sensor_width_mm": 13.2,
    "focal_length_mm": 8.8,
}


@app.websocket("/ws/hud")
async def websocket_hud_endpoint(websocket: WebSocket):
    """WebSocket connection endpoint for the client-side inspection HUD."""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"HUD Client Connected: {websocket.client}")

    try:
        while True:
            # Handle incoming control packets from client HUD (e.g., altitude override)
            data_raw = await websocket.receive_text()
            data = json.loads(data_raw)

            if "override_altitude_m" in data:
                current_telemetry["altitude_m"] = float(data["override_altitude_m"])
                logger.info(f"Updated Flight Altitude: {current_telemetry['altitude_m']}m")

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"HUD Client Disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        connected_clients.discard(websocket)


async def broadcast_hud_data(payload: dict):
    """Broadcasts json payload to all connected HUD clients."""
    if not connected_clients:
        return

    dead_clients = set()
    message = json.dumps(payload)

    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead_clients.add(client)

    for dead in dead_clients:
        connected_clients.discard(dead)


def process_frame_and_calculate_metrics(frame: np.ndarray, telemetry: dict) -> dict:
    """
    Runs YOLO11-seg detection and calculates physical dimensions using GSDEngine.
    """
    img_height, img_width = frame.shape[:2]
    alt_m = telemetry.get("altitude_m", 15.0)

    # 1. Compute GSD (cm/px) for current frame scale
    # formula: GSD = (Altitude * SensorWidth) / (FocalLength * ImageWidth) * 100
    gsd_cm_px = (alt_m * telemetry["sensor_width_mm"]) / (
        telemetry["focal_length_mm"] * img_width
    ) * 100.0

    # 2. Execute YOLO11-seg Inference
    results = model(frame, verbose=False)[0]
    detections = []

    if results.masks is not None:
        # Extract polygons & boxes
        for mask_xy, box in zip(results.masks.xy, results.boxes):
            if len(mask_xy) < 3:
                continue

            polygon = mask_xy.tolist()  # [[x1, y1], [x2, y2], ...]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # 3. Calculate Physical Dimensions via GSDEngine logic
            physical_metrics = gsd_engine.convert_polygon_to_physical(
                polygon=polygon, gsd_cm_px=gsd_cm_px
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox_xyxy": [round(x, 1) for x in box.xyxy[0].tolist()],
                    "polygon_contour": polygon,
                    "metrics": {
                        "gsd_cm_px": physical_metrics["gsd_cm_px"],
                        "length_cm": physical_metrics["length_cm"],
                        "area_cm2": physical_metrics["area_cm2"],
                        "avg_width_mm": physical_metrics["avg_width_mm"],
                    },
                }
            )

    return {
        "timestamp": time.time(),
        "telemetry": {"altitude_m": alt_m, "gsd_cm_px": round(gsd_cm_px, 4)},
        "detection_count": len(detections),
        "detections": detections,
    }


async def rtsp_inference_worker(rtsp_url: Union[str, int] = 0):
    """
    Background worker loop reading RTSP video frames (or webcam=0 for testing).
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"Unable to open video stream source: {rtsp_url}")
        return

    logger.info("RTSP Stream Worker Started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        # Execute processing pipeline in thread pool to prevent blocking event loop
        hud_payload = await asyncio.to_thread(
            process_frame_and_calculate_metrics, frame, current_telemetry
        )

        # Broadcast telemetry & detection measurements over WebSockets
        await broadcast_hud_data(hud_payload)

        # ~30 FPS throttle
        await asyncio.sleep(0.033)


@app.on_event("startup")
async def startup_event():
    """Launch background video worker when FastAPI boots."""
    camera_stream_url = "http://192.168.1.6:8080/video"
    asyncio.create_task(rtsp_inference_worker(rtsp_url=camera_stream_url))