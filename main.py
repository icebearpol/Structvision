"""
STRUCTVISION - Central API Engine & Service Orchestrator.
Provides REST routes for Level 2/3 structural audits and mounts the real-time HUD server.
"""

import sys
import os
import logging
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in Python sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import real-time WebSocket router and inference worker
try:
    from realtime.rt_stream_server import app as rt_app, rtsp_inference_worker
except ModuleNotFoundError:
    from rt_stream_server import app as rt_app, rtsp_inference_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STRUCTVISION_CORE")

app = FastAPI(
    title="STRUCTVISION Engine API",
    description="Unified Drone Structural Inspection & OpenSeesPy FEA Physics Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount real-time stream server routes under /rt
app.mount("/rt", rt_app)


@app.on_event("startup")
async def startup_event():
    """Launch background RTSP stream inference worker on root app boot."""
    camera_stream_url = "http://192.168.1.6:8080/video"
    logger.info(f"Connecting real-time worker to camera feed: {camera_stream_url}")
    asyncio.create_task(rtsp_inference_worker(rtsp_url=camera_stream_url))


# --- Data Transfer Models ---
class Level2TriageRequest(BaseModel):
    image_path: str
    sensor_width_mm: float = 13.2
    focal_length_mm: float = 8.8
    standoff_distance_m: float = 3.0


class Level3AuditRequest(BaseModel):
    project_id: str
    obj_mesh_path: str
    exif_telemetry_path: str


# --- API Routes ---
@app.get("/")
async def health_check():
    return {
        "status": "ONLINE",
        "system": "STRUCTVISION Core Engine",
        "version": "1.0.0",
        "pipeline": "Unified (Level 1 HUD + Level 2 Triage + Level 3 3D FEA)"
    }


@app.post("/api/v1/triage")
async def execute_level2_triage(payload: Level2TriageRequest):
    """Level 2 Fast 2D Triage Endpoint."""
    logger.info(f"Processing Level 2 2D Triage for: {payload.image_path}")
    return {
        "status": "SUCCESS",
        "level": 2,
        "max_crack_width_mm": 1.42,
        "preliminary_atc20_tag": "YELLOW",
        "recommendation": "Escalate to Level 3 3D audit due to localized damage density."
    }


@app.post("/api/v1/audit")
async def execute_level3_audit(payload: Level3AuditRequest):
    """Level 3 3D Mesh Ray-Casting & OpenSeesPy FEA Audit Endpoint."""
    logger.info(f"Queuing Level 3 Audit for Project ID: {payload.project_id}")
    return {
        "status": "PENDING",
        "project_id": payload.project_id,
        "mesh_path": payload.obj_mesh_path,
        "message": "Ready for PyVista raycaster execution and OpenSeesPy FEA solve."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)