import asyncio
import json
import websockets


async def receive_hud():
    uri = "ws://127.0.0.1:8000/ws/hud"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri) as websocket:
        print("Connected to StructVision Live HUD Stream!\n")

        while True:
            data = await websocket.recv()
            payload = json.loads(data)

            timestamp = payload.get("timestamp")
            telemetry = payload.get("telemetry", {})
            detection_count = payload.get("detection_count", 0)
            detections = payload.get("detections", [])

            print(
                f"[HUD STREAM] Telemetry: Alt={telemetry.get('altitude_m')}m | GSD={telemetry.get('gsd_cm_px')} cm/px | Detections={detection_count}"
            )

            for idx, det in enumerate(detections, 1):
                metrics = det.get("metrics", {})
                print(
                    f"  └─ Defect #{idx}: Class='{det.get('class')}' (Conf: {det.get('confidence')})"
                )
                print(
                    f"     Physical Metrics: Length={metrics.get('length_cm')}cm | Avg Width={metrics.get('avg_width_mm')}mm | Area={metrics.get('area_cm2')}cm²"
                )

            print("-" * 65)


if __name__ == "__main__":
    try:
        asyncio.run(receive_hud())
    except KeyboardInterrupt:
        print("\nDisconnected from HUD Stream.")