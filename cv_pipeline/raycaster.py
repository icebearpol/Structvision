"""
STRUCTVISION - Level 3 3D Ray-Casting & Spatial Mapping Engine.
Projects 2D YOLO segmentation masks onto 3D OpenDroneMap OBJ meshes using PyVista ray tracing.
"""

import os
import numpy as np
import pyvista as pv
import logging
from typing import List, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STRUCTVISION_RAYCASTER")


class MeshRaycaster:
    def __init__(self, obj_path: str):
        """Initialize and load the 3D target mesh using PyVista."""
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"3D Mesh file not found at: {obj_path}")

        logger.info(f"Loading 3D mesh: {obj_path}")
        self.mesh = pv.read(obj_path)
        logger.info(f"Mesh loaded successfully. Vertices: {self.mesh.n_points}, Faces: {self.mesh.n_cells}")

    def ray_cast_single_point(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        """
        Cast a single ray from camera center through a 2D pixel vector into the 3D scene.
        Returns the closest 3D coordinate intersection point on the mesh surface.
        """
        ray_end = ray_origin + (ray_direction * 1000.0)
        points, intersection_cells = self.mesh.ray_trace(ray_origin, ray_end)

        if len(points) > 0:
            return points[0]
        return None

    def project_2d_mask_to_3d_nodes(
        self,
        polygon_pixels: List[Tuple[int, int]],
        camera_position: Tuple[float, float, float],
        camera_rotation_matrix: np.ndarray,
        intrinsic_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Projects 2D YOLO polygon mask vertices onto the 3D mesh surface."""
        cam_origin = np.array(camera_position, dtype=np.float64)
        inv_K = np.linalg.inv(intrinsic_matrix)
        inv_R = camera_rotation_matrix.T

        hit_3d_points = []
        
        for pixel_x, pixel_y in polygon_pixels:
            pixel_homo = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
            ray_cam = inv_K @ pixel_homo
            ray_world_dir = inv_R @ ray_cam
            ray_world_dir = ray_world_dir / np.linalg.norm(ray_world_dir)

            point_3d = self.ray_cast_single_point(cam_origin, ray_world_dir)
            
            if point_3d is not None:
                hit_3d_points.append(point_3d)

        hit_3d_points = np.array(hit_3d_points) if len(hit_3d_points) > 0 else np.empty((0, 3))
        
        logger.info(f"Projected {len(polygon_pixels)} 2D polygon vertices -> {len(hit_3d_points)} 3D mesh surface hits.")
        
        return {
            "mapped_3d_nodes": hit_3d_points,
            "surface_facet_count": len(hit_3d_points)
        }

    def generate_fea_damage_map(self, hit_3d_nodes: np.ndarray, output_vtks_path: str = "damaged_mesh.vtk"):
        """Highlights projected crack regions on the PyVista 3D mesh and exports VTK/FEA ready damage map."""
        if len(hit_3d_nodes) == 0:
            logger.warning("No 3D points provided to generate damage map.")
            return None

        damage_scalar = np.zeros(self.mesh.n_points)
        
        # Iterate over each projected 3D hit point to find its closest mesh vertex ID
        closest_point_ids = [self.mesh.find_closest_point(pt) for pt in hit_3d_nodes]
        damage_scalar[closest_point_ids] = 1.0
        
        self.mesh["crack_damage"] = damage_scalar
        self.mesh.save(output_vtks_path)
        logger.info(f"Saved VTK damage map to: {output_vtks_path}")
        return output_vtks_path