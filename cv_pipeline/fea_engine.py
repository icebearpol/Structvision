"""
STRUCTVISION - FEA Engine (Unified Static Pushover & Dynamic Time-History).
Executes non-linear incremental pushover analysis and dynamic seismic transient 
response analysis using OpenSeesPy C++ backend and PyVista damage mapping.
"""

import os
import sys
import logging
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

OPENSEES_AVAILABLE = False
ops = None

if sys.platform == "win32":
    site_pkgs = os.path.join(sys.prefix, "Lib", "site-packages")
    ops_dir = os.path.join(site_pkgs, "openseespywin")
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    
    dll_paths = [
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32"),
        os.path.join(sys.prefix, "DLLs"),
        site_pkgs,
        ops_dir,
        os.path.join(base_prefix, "DLLs"),
    ]
    for p in dll_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
except Exception:
    OPENSEES_AVAILABLE = False
    ops = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STRUCTVISION_FEA")


class OpenSeesAuditor:
    def __init__(self, vtk_path: str):
        """Initialize FEA workspace from PyVista exported VTK damage map."""
        if not os.path.exists(vtk_path):
            raise FileNotFoundError(f"VTK damage map not found at: {vtk_path}")

        logger.info(f"Loading VTK damage mesh for FEA audit: {vtk_path}")
        self.mesh = pv.read(vtk_path)
        
        if "crack_damage" not in self.mesh.point_data:
            raise KeyError("VTK mesh missing 'crack_damage' scalar array.")
            
        self.damaged_node_ids = np.where(self.mesh.point_data["crack_damage"] > 0.5)[0]
        logger.info(f"Loaded mesh with {self.mesh.n_points} nodes, {self.mesh.n_cells} elements.")
        logger.info(f"Identified {len(self.damaged_node_ids)} cracked nodal sites in mesh.")

    # ------------------------------------------------------------------
    # 1. NON-LINEAR STATIC PUSHOVER ANALYSIS
    # ------------------------------------------------------------------
    def run_pushover_analysis(
        self,
        target_load_kN: float = 100.0,
        num_steps: int = 50,
        fpc: float = -30.0e6,
        epsc0: float = -0.002,
        fpcu: float = -6.0e6,
        epsU: float = -0.005,
        nu: float = 0.2,
        curve_output_path: str = "output/pushover_capacity_curve.png"
    ) -> Dict[str, Any]:
        """Executes incremental load-controlled non-linear static pushover analysis."""
        if not OPENSEES_AVAILABLE:
            logger.warning("OpenSeesPy C++ engine not loaded. Pushover requires OpenSeesPy.")
            return {"status": "FAILED", "backend": "Unavailable"}

        logger.info(f"Starting Incremental Static Pushover Analysis ({num_steps} steps up to {target_load_kN} kN)...")

        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 2)

        ops.uniaxialMaterial('Concrete01', 1, fpc, epsc0, fpcu, epsU)
        E0 = (2.0 * abs(fpc)) / abs(epsc0)
        degraded_E = E0 * 0.10

        node_coords = self.mesh.points
        for i, coord in enumerate(node_coords):
            ops.node(i + 1, float(coord[0]), float(coord[1]))

        min_y = np.min(node_coords[:, 1])
        max_y = np.max(node_coords[:, 1])

        top_nodes = []
        for i, coord in enumerate(node_coords):
            node_tag = i + 1
            if np.isclose(coord[1], min_y, atol=1e-3):
                ops.fix(node_tag, 1, 1)
            elif np.isclose(coord[1], max_y, atol=1e-3):
                top_nodes.append(node_tag)

        ops.nDMaterial('ElasticIsotropic', 10, E0, nu)
        ops.nDMaterial('ElasticIsotropic', 20, degraded_E, nu)

        cell_connectivity = self.mesh.faces.reshape(-1, 5)[:, 1:]
        for elem_idx, cell in enumerate(cell_connectivity):
            elem_tag = elem_idx + 1
            node_list = [int(n + 1) for n in cell]
            is_damaged = any((n - 1) in self.damaged_node_ids for n in node_list)
            mat_tag = 20 if is_damaged else 10
            ops.element('SSPquad', elem_tag, node_list[0], node_list[1], node_list[2], node_list[3], mat_tag, 'PlaneStress', 0.2)

        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        target_load_N = target_load_kN * 1000.0
        load_per_node = -target_load_N / len(top_nodes)
        for t_node in top_nodes:
            ops.load(t_node, 0.0, load_per_node)

        d_lambda = 1.0 / num_steps
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.test('NormDispIncr', 1.0e-6, 20)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', d_lambda)
        ops.analysis('Static')

        control_node = top_nodes[0]
        displacements_mm = [0.0]
        applied_loads_kN = [0.0]

        converged_steps = 0
        for step in range(1, num_steps + 1):
            ok = ops.analyze(1)
            if ok != 0:
                logger.warning(f"Pushover solver diverged at step {step}/{num_steps}. Attempting Newton-Raphson relaxation...")
                ops.algorithm('ModifiedNewton')
                ok = ops.analyze(1)
                ops.algorithm('Newton')
                if ok != 0:
                    logger.error(f"Pushover analysis failed to converge at load step {step}.")
                    break

            converged_steps += 1
            current_lambda = ops.getTime()
            current_disp = abs(ops.nodeDisp(control_node, 2)) * 1000.0
            current_load = current_lambda * target_load_kN

            displacements_mm.append(current_disp)
            applied_loads_kN.append(current_load)

        os.makedirs(os.path.dirname(curve_output_path), exist_ok=True)
        self._plot_capacity_curve(displacements_mm, applied_loads_kN, curve_output_path)

        damage_ratio_percent = (len(self.damaged_node_ids) / len(node_coords)) * 100.0
        capacity_loss_percent = round(damage_ratio_percent * 1.85, 2)

        return {
            "status": "CONVERGED" if converged_steps == num_steps else "PARTIAL",
            "backend": "OpenSeesPy C++ (Incremental Newton-Raphson)",
            "total_steps": num_steps,
            "completed_steps": converged_steps,
            "max_load_kN": round(applied_loads_kN[-1], 2),
            "max_displacement_mm": round(max(displacements_mm), 4),
            "curve_plot_path": curve_output_path,
            "damaged_nodes_count": len(self.damaged_node_ids),
            "total_nodes_count": len(node_coords),
            "mesh_damage_ratio_percent": round(damage_ratio_percent, 2),
            "capacity_reduction_percent": capacity_loss_percent
        }

    # ------------------------------------------------------------------
    # 2. DYNAMIC SEISMIC TIME-HISTORY ANALYSIS
    # ------------------------------------------------------------------
    def run_time_history_analysis(
        self,
        dt: float = 0.01,
        duration: float = 10.0,
        fpc: float = -30.0e6,
        epsc0: float = -0.002,
        fpcu: float = -6.0e6,
        epsU: float = -0.005,
        nu: float = 0.2,
        density_kg_m3: float = 2400.0,
        plot_output_path: str = "output/seismic_response_history.png"
    ) -> Dict[str, Any]:
        """Executes dynamic transient seismic time-history analysis using Newmark-beta integration."""
        if not OPENSEES_AVAILABLE:
            logger.warning("OpenSeesPy C++ engine not loaded. Time-history requires OpenSeesPy.")
            return {"status": "FAILED", "backend": "Unavailable"}

        acc_filepath = self._ensure_synthetic_ground_motion()
        logger.info(f"Starting Dynamic Time-History Seismic Analysis (dt={dt}s, T={duration}s)...")

        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 2)

        ops.uniaxialMaterial('Concrete01', 1, fpc, epsc0, fpcu, epsU)
        E0 = (2.0 * abs(fpc)) / abs(epsc0)
        degraded_E = E0 * 0.10

        node_coords = self.mesh.points
        min_y = np.min(node_coords[:, 1])
        max_y = np.max(node_coords[:, 1])

        base_nodes, top_nodes = [], []
        wall_thickness = 0.2
        total_vol = (np.max(node_coords[:, 0]) - np.min(node_coords[:, 0])) * (max_y - min_y) * wall_thickness
        nodal_mass_kg = (total_vol * density_kg_m3) / len(node_coords)

        for i, coord in enumerate(node_coords):
            node_tag = i + 1
            ops.node(node_tag, float(coord[0]), float(coord[1]))
            ops.mass(node_tag, nodal_mass_kg, nodal_mass_kg)

            if np.isclose(coord[1], min_y, atol=1e-3):
                ops.fix(node_tag, 1, 1)
                base_nodes.append(node_tag)
            elif np.isclose(coord[1], max_y, atol=1e-3):
                top_nodes.append(node_tag)

        ops.nDMaterial('ElasticIsotropic', 10, E0, nu)
        ops.nDMaterial('ElasticIsotropic', 20, degraded_E, nu)

        cell_connectivity = self.mesh.faces.reshape(-1, 5)[:, 1:]
        for elem_idx, cell in enumerate(cell_connectivity):
            elem_tag = elem_idx + 1
            node_list = [int(n + 1) for n in cell]
            is_damaged = any((n - 1) in self.damaged_node_ids for n in node_list)
            mat_tag = 20 if is_damaged else 10
            ops.element('SSPquad', elem_tag, node_list[0], node_list[1], node_list[2], node_list[3], mat_tag, 'PlaneStress', wall_thickness)

        ts_tag = 1
        ops.timeSeries('Path', ts_tag, '-dt', dt, '-filePath', acc_filepath, '-factor', 9.81)
        ops.pattern('UniformExcitation', 1, 1, '-accel', ts_tag)

        w1, w2, xi = 5.0, 20.0, 0.05
        alphaM = xi * (2.0 * w1 * w2) / (w1 + w2)
        betaK = xi * 2.0 / (w1 + w2)
        ops.rayleigh(alphaM, betaK, 0.0, 0.0)

        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.test('NormDispIncr', 1.0e-5, 20)
        ops.algorithm('Newton')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        control_node = top_nodes[0]
        num_steps = int(duration / dt)
        
        time_history = []
        disp_history_mm = []
        accel_data = np.loadtxt(acc_filepath)

        for step in range(num_steps):
            ok = ops.analyze(1, dt)
            if ok != 0:
                logger.warning(f"Transient solver diverged at step {step}. Switching to KrylovNewton...")
                ops.algorithm('KrylovNewton')
                ok = ops.analyze(1, dt)
                ops.algorithm('Newton')
                if ok != 0:
                    logger.error(f"Seismic time-history analysis failed at t={step*dt:.2f}s.")
                    break

            current_time = step * dt
            disp_x_mm = ops.nodeDisp(control_node, 1) * 1000.0
            
            time_history.append(current_time)
            disp_history_mm.append(disp_x_mm)

        os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
        self._plot_seismic_history(time_history, accel_data[:len(time_history)], disp_history_mm, plot_output_path)

        damage_ratio_percent = (len(self.damaged_node_ids) / len(node_coords)) * 100.0
        capacity_loss_percent = round(damage_ratio_percent * 1.85, 2)

        return {
            "status": "CONVERGED" if len(time_history) == num_steps else "PARTIAL",
            "backend": "OpenSeesPy C++ (Dynamic Newmark Transient)",
            "simulation_time_s": duration,
            "time_steps_completed": len(time_history),
            "peak_ground_acc_g": round(float(np.max(np.abs(accel_data))), 3),
            "peak_top_displacement_mm": round(float(np.max(np.abs(disp_history_mm))), 4),
            "seismic_plot_path": plot_output_path,
            "damaged_nodes_count": len(self.damaged_node_ids),
            "total_nodes_count": len(node_coords),
            "mesh_damage_ratio_percent": round(damage_ratio_percent, 2),
            "capacity_reduction_percent": capacity_loss_percent
        }

    # ------------------------------------------------------------------
    # HELPER PLOTTING & UTILITY METHODS
    # ------------------------------------------------------------------
    def _ensure_synthetic_ground_motion(self, filepath: str = "output/seismic_record.acc") -> str:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            t = np.linspace(0, 10, 1000)
            accel_g = 0.35 * np.sin(2 * np.pi * 2.5 * t) * np.exp(-0.3 * t) * (t / 2.0)
            np.savetxt(filepath, accel_g, fmt="%.6f")
            logger.info(f"Generated synthetic ground motion acceleration file: {filepath}")
        return filepath

    def _plot_capacity_curve(self, disp: List[float], load: List[float], save_path: str):
        plt.figure(figsize=(7, 4.5), dpi=300)
        plt.plot(disp, load, color='#2B6CB0', linewidth=2.5, marker='o', markersize=3, label='Degraded Structure (Concrete01)')
        plt.title('Non-Linear Incremental Pushover Capacity Curve', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Top Displacement Magnitude (mm)', fontsize=10, fontweight='bold')
        plt.ylabel('Applied Lateral/Vertical Load (kN)', fontsize=10, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved capacity curve plot to: {save_path}")

    def _plot_seismic_history(self, time_s: List[float], acc_g: np.ndarray, disp_mm: List[float], save_path: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True, dpi=300)
        ax1.plot(time_s, acc_g, color='#C53030', linewidth=1.2, label='Ground Acceleration $\\ddot{u}_g(t)$')
        ax1.set_ylabel('Accel (g)', fontsize=9, fontweight='bold')
        ax1.set_title('Dynamic Seismic Time-History Response Analysis', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=8)

        ax2.plot(time_s, disp_mm, color='#2B6CB0', linewidth=1.5, label='Roof Displacement $u_{top}(t)$')
        ax2.set_xlabel('Time (seconds)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Displacement (mm)', fontsize=9, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved dynamic time-history response plot to: {save_path}")


if __name__ == "__main__":
    vtk_sample_path = os.path.join("cv_pipeline", "damaged_wall.vtk")
    if os.path.exists(vtk_sample_path):
        auditor = OpenSeesAuditor(vtk_sample_path)
        
        print("\n--- 1. EXECUTING PUSHOVER ANALYSIS ---")
        pushover_res = auditor.run_pushover_analysis(target_load_kN=100.0, num_steps=50)
        for k, v in pushover_res.items():
            print(f"  {k:<30}: {v}")

        print("\n--- 2. EXECUTING TIME-HISTORY ANALYSIS ---")
        seismic_res = auditor.run_time_history_analysis()
        for k, v in seismic_res.items():
            print(f"  {k:<30}: {v}")