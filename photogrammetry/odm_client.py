"""
STRUCTVISION - NodeODM Photogrammetry Task Client.
Dispatches inspection image batches to NodeODM and downloads orthomosaics.
"""

from pathlib import Path
import asyncio
from pyodm import Node, exceptions


class ODMTaskManager:
    def __init__(self, host: str = "localhost", port: int = 3000):
        self.host = host
        self.port = port
        self.node = Node(host, port)

    async def verify_node_connection(self) -> bool:
        """Checks if the NodeODM Docker container is responding."""
        try:
            info = await asyncio.to_thread(self.node.info)
            print(f"[ODMTaskManager] Connected to NodeODM Engine v{info.version}")
            return True
        except Exception as e:
            print(f"[ODMTaskManager] Failed to connect to NodeODM: {e}")
            return False

    async def process_dataset(
        self,
        image_paths: list[Path],
        output_dir: Path,
        orthophoto_resolution_cm: float = 1.0,
    ) -> dict:
        """
        Dispatches images to NodeODM to reconstruct an orthomosaic.
        """
        if not image_paths:
            raise ValueError("No images provided for photogrammetry processing.")

        output_dir.mkdir(parents=True, exist_ok=True)
        file_list = [str(p) for p in image_paths if p.exists()]

        print(f"[ODMTaskManager] Submitting {len(file_list)} images to NodeODM...")

        # Run synchronous PyODM calls in a thread pool to preserve FastAPI concurrency
        task = await asyncio.to_thread(
            self.node.create_task,
            files=file_list,
            options={
                "orthophoto-resolution": orthophoto_resolution_cm,
                "fast-orthophoto": True,
                "dsm": False,
            },
        )

        print(f"[ODMTaskManager] Task created with UUID: {task.uuid}")

        # Poll status until complete
        while True:
            info = await asyncio.to_thread(task.info)
            print(f"[ODMTaskManager] Task [{task.uuid}] Progress: {info.progress}% | Status: {info.status}")

            if info.status.name == "COMPLETED":
                break
            elif info.status.name == "FAILED":
                raise RuntimeError(f"NodeODM Processing Failed: {info.last_error}")

            await asyncio.sleep(5)

        # Download results
        print("[ODMTaskManager] Task complete. Downloading assets...")
        await asyncio.to_thread(task.download_assets, str(output_dir))

        return {
            "task_uuid": task.uuid,
            "output_directory": str(output_dir),
            "orthophoto_path": str(output_dir / "orthophoto.tif"),
        }


if __name__ == "__main__":
    # Sanity test connection
    client = ODMTaskManager()
    asyncio.run(client.verify_node_connection())