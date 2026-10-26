"""
STRUCTVISION - Raycaster Mock Verification Script
Generates a mock 3D structural wall mesh and tests 2D-to-3D projection using PyVista.
"""

import os
import sys
import numpy as np
import pyvista as pv

# Inject project root directory into sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv_pipeline.raycaster import MeshRaycaster


def run_mock_test():
    print("\n--- STEP 1: Creating Synthetic 3D Structural Wall ---")
    # Generate a simple flat 3D wall mesh (10m x 5m plane at Z=0)
    wall_mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=5)
    mock_obj_path = os.path.join("cv_pipeline", "mock_wall.obj")
    wall_mesh.save(mock_obj_path)
    print(f"Saved synthetic mesh to: {mock_obj_path}")

    print("\n--- STEP 2: Initializing MeshRaycaster ---")
    raycaster = MeshRaycaster(mock_obj_path)

    print("\n--- STEP 3: Simulating 2D YOLO Crack Mask Projection ---")
    # Define camera intrinsics (1080p camera parameters)
    fx, fy = 1000.0, 1000.0
    cx, cy = 960.0, 540.0
    K_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # Camera sitting at (0, 0, -5.0) pointing straight at the wall at Z=0
    camera_pos = (0.0, 0.0, -5.0)
    rotation_matrix = np.eye(3, dtype=np.float64)

    # Sample 2D crack polygon coordinates around the center of the image
    crack_2d_pixels = [
        (960, 540),  # Center
        (970, 550),
        (980, 560),
        (990, 570)
    ]

    # Project 2D pixels onto 3D mesh
    results = raycaster.project_2d_mask_to_3d_nodes(
        polygon_pixels=crack_2d_pixels,
        camera_position=camera_pos,
        camera_rotation_matrix=rotation_matrix,
        intrinsic_matrix=K_matrix
    )

    print("\n--- STEP 4: Exporting FEA-Ready VTK Damage Map ---")
    if len(results["mapped_3d_nodes"]) > 0:
        vtk_path = raycaster.generate_fea_damage_map(
            hit_3d_nodes=results["mapped_3d_nodes"],
            output_vtks_path=os.path.join("cv_pipeline", "damaged_wall.vtk")
        )
        print(f"SUCCESS: Raycaster mapped {len(results['mapped_3d_nodes'])} nodes to 3D surface!")
    else:
        print("WARNING: Rays missed the surface.")


if __name__ == "__main__":
    run_mock_test()