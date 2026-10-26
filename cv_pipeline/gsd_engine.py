"""
STRUCTVISION - Ground Sampling Distance (GSD) & Physical Measurement Engine.
Calculates real-world dimensions (crack length, area, avg width) from image EXIF and YOLO masks.
"""

from pathlib import Path
import math
from PIL import Image, ExifTags


class GSDEngine:
    def __init__(self, sensor_width_mm: float = 13.2, focal_length_mm: float = 8.8):
        """
        Default parameters set for standard 1-inch UAV sensors (e.g., DJI Phantom 4 Pro / Mavic 3E).
        """
        self.sensor_width_mm = sensor_width_mm
        self.focal_length_mm = focal_length_mm

    def extract_exif(self, image_path: Path) -> dict:
        """Extracts key camera parameters and GPS altitude from EXIF data."""
        metadata = {}
        try:
            with Image.open(image_path) as img:
                metadata["image_width"] = img.width
                metadata["image_height"] = img.height
                
                exif_raw = img._getexif()
                if not exif_raw:
                    return metadata

                exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

                if "FocalLength" in exif:
                    fl = exif["FocalLength"]
                    metadata["focal_length_mm"] = float(fl[0] / fl[1]) if isinstance(fl, tuple) else float(fl)

                if "GPSInfo" in exif:
                    gps = exif["GPSInfo"]
                    if 6 in gps:  # Tag 6 = GPSAltitude
                        alt = gps[6]
                        metadata["altitude_m"] = float(alt[0] / alt[1]) if isinstance(alt, tuple) else float(alt)

        except Exception as e:
            print(f"[GSDEngine] EXIF Extraction Warning ({image_path.name}): {e}")

        return metadata

    def calculate_gsd(
        self,
        image_path: Path,
        flight_altitude_m: float = None,
        override_sensor_width_mm: float = None,
        override_focal_length_mm: float = None,
    ) -> float:
        """
        Computes Ground Sampling Distance (GSD) in cm/pixel using:
        GSD = (Altitude * SensorWidth) / (FocalLength * ImageWidth) * 100
        """
        exif_data = self.extract_exif(image_path)

        img_width = exif_data.get("image_width")
        if not img_width:
            raise ValueError(f"Unable to resolve image width for {image_path}")

        focal_length = override_focal_length_mm or exif_data.get("focal_length_mm") or self.focal_length_mm
        sensor_width = override_sensor_width_mm or self.sensor_width_mm
        altitude = flight_altitude_m or exif_data.get("altitude_m")

        if not altitude or altitude <= 0:
            raise ValueError("Flight altitude missing. Please pass 'flight_altitude_m'.")

        gsd_cm_px = (altitude * sensor_width) / (focal_length * img_width) * 100.0
        return round(gsd_cm_px, 4)

    def convert_polygon_to_physical(self, polygon: list[list[float]], gsd_cm_px: float) -> dict:
        """
        Converts YOLO pixel segmentation masks into physical surface measurements.
        """
        gsd_mm_px = gsd_cm_px * 10.0

        # Compute perimeter length in pixels
        length_px = 0.0
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            length_px += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        # Shoelace formula for polygon area in pixels²
        n = len(polygon)
        area_px = 0.0
        for i in range(n):
            j = (i + 1) % n
            area_px += polygon[i][0] * polygon[j][1]
            area_px -= polygon[j][0] * polygon[i][1]
        area_px = abs(area_px) / 2.0

        half_perimeter_px = length_px / 2.0 or 1.0
        physical_length_cm = round(half_perimeter_px * gsd_cm_px, 2)
        physical_area_cm2 = round(area_px * (gsd_cm_px ** 2), 2)
        avg_width_mm = round((area_px / half_perimeter_px) * gsd_mm_px, 2)

        return {
            "gsd_cm_px": gsd_cm_px,
            "length_cm": physical_length_cm,
            "area_cm2": physical_area_cm2,
            "avg_width_mm": avg_width_mm,
        }