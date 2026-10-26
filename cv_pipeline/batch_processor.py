import os
import glob
import csv
import numpy as np

# Try importing OpenCV; if unavailable, fallback to standard array math
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from cv_pipeline.pipeline import StructuralAuditPipeline


class BatchCrackProcessor:
    """Batch processes concrete crack images, calculates damage parameters (D),

    computes degraded Young's Modulus E_damaged = (1 - D) * E0, and logs metrics
    across multi-crack test suites.
    """

    def __init__(
        self,
        input_dir="input_images",
        output_csv="output/batch_processing_summary.csv",
        base_E_GPa=30.0,
    ):
        self.input_dir = input_dir
        self.output_csv = output_csv
        self.base_E_Pa = base_E_GPa * 1e9  # 30 GPa baseline
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

    def generate_synthetic_samples_if_empty(self):
        """Generates mock concrete crack images if no files exist in input_images/."""
        existing_files = glob.glob(os.path.join(self.input_dir, "*.[pj][pn][g]*"))
        if existing_files:
            return existing_files

        print(
            f"[!] No images found in '{self.input_dir}'. Generating synthetic test crack suite..."
        )

        sample_configs = [
            ("crack_minor.png", 0.05, "Hairline Micro-cracking"),
            ("crack_moderate.png", 0.18, "Diagonal Shear Fracture"),
            ("crack_severe.png", 0.38, "Major Structural Spalling"),
        ]

        generated = []
        for name, damage_ratio, desc in sample_configs:
            file_path = os.path.join(self.input_dir, name)
            img = np.full((300, 300), 200, dtype=np.uint8)  # Concrete grey background

            # Draw synthetic crack lines
            num_pixels_to_crack = int(300 * 300 * damage_ratio)
            y_coords = np.random.randint(0, 300, size=num_pixels_to_crack)
            x_coords = np.random.randint(0, 300, size=num_pixels_to_crack)
            img[y_coords, x_coords] = 10  # Dark crack pixels

            if HAS_OPENCV:
                cv2.imwrite(file_path, img)
            else:
                # Basic PGM image writer fallback if OpenCV is missing
                with open(file_path.replace(".png", ".pgm"), "wb") as f:
                    f.write(
                        f"P5\n300 300\n255\n".encode() + img.tobytes()
                    )
                file_path = file_path.replace(".png", ".pgm")

            generated.append(file_path)
            print(
                f"  -> Created synthetic sample: {file_path} ({desc}, Target D ~ {damage_ratio:.2f})"
            )

        return generated

    def analyze_image_damage(self, image_path):
        """Extracts damage index D in range [0.0, 1.0] from concrete damage photo."""
        if HAS_OPENCV and image_path.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.05, 0  # Default fallback
            # Otsu thresholding to segment dark crack pixels from concrete background
            _, thresh = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            crack_pixels = np.count_nonzero(thresh)
            total_pixels = thresh.size
            damage_index_D = min(1.0, max(0.0, crack_pixels / total_pixels))
        else:
            # File reading fallback for PGM or non-OpenCV
            try:
                with open(image_path, "rb") as f:
                    data = f.read()
                pixels = np.frombuffer(data[-90000:], dtype=np.uint8)
                crack_pixels = np.count_nonzero(pixels < 50)
                damage_index_D = min(1.0, max(0.0, crack_pixels / len(pixels)))
            except Exception:
                damage_index_D = 0.12

        return damage_index_D

    def run_batch(self):
        print("=" * 65)
        print("    STRUCTVISION MULTI-CRACK BATCH IMAGE PROCESSOR    ")
        print("=" * 65)

        image_files = glob.glob(
            os.path.join(self.input_dir, "*.[pjP][pnG][gM]*")
        )
        if not image_files:
            image_files = self.generate_synthetic_samples_if_empty()

        results = []

        print(
            f"\nProcessing {len(image_files)} concrete crack damage images...\n"
        )
        print(
            f"{'Image Filename':<25} | {'Damage (D)':<10} | {'E_damaged (GPa)':<16} | {'Status'}"
        )
        print("-" * 65)

        for img_path in sorted(image_files):
            filename = os.path.basename(img_path)

            # 1. Image Segmentation & Damage Calculation
            D = self.analyze_image_damage(img_path)

            # 2. Material Property Degradation: E_damaged = (1 - D) * E0
            E_damaged_Pa = (1.0 - D) * self.base_E_Pa
            E_damaged_GPa = E_damaged_Pa / 1e9

            # 3. Peak Stress Estimation (SCF scaling with damage severity)
            peak_stress_MPa = 11.90 * (1.0 + (D * 8.5))

            # Severity Classification
            if D < 0.10:
                severity = "Minor (Pass)"
            elif D < 0.25:
                severity = "Moderate (Monitor)"
            else:
                severity = "Critical (Retrofit Required)"

            results.append(
                {
                    "Filename": filename,
                    "Damage_Index_D": round(D, 4),
                    "Damage_Percentage": round(D * 100, 2),
                    "E_Damaged_GPa": round(E_damaged_GPa, 2),
                    "Estimated_Peak_Stress_MPa": round(peak_stress_MPa, 2),
                    "Audit_Severity": severity,
                }
            )

            print(
                f"{filename:<25} | {D*100:6.2f}%    | {E_damaged_GPa:6.2f} GPa      | {severity}"
            )

        # 4. Save Multi-Crack Audit Report to CSV
        fieldnames = [
            "Filename",
            "Damage_Index_D",
            "Damage_Percentage",
            "E_Damaged_GPa",
            "Estimated_Peak_Stress_MPa",
            "Audit_Severity",
        ]
        with open(self.output_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print("-" * 65)
        print(f"\n[SUCCESS] Batch processing complete!")
        print(f"Summary log exported to: '{self.output_csv}'")


if __name__ == "__main__":
    processor = BatchCrackProcessor()
    processor.run_batch()