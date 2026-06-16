"""
src/imaging/reconstruction.py
Convert segmentation masks → 3D meshes using marching cubes + PyTorch3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.ndimage import binary_fill_holes, gaussian_filter, label as scipy_label
from skimage.measure import marching_cubes

try:
    import torch
    import pytorch3d
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
    P3D_AVAILABLE = True
except ImportError:
    P3D_AVAILABLE = False
    logger.warning("PyTorch3D not available — mesh ops disabled.")


@dataclass
class AnatomicalMesh:
    """A 3D mesh for a single anatomical structure."""
    label_id: int
    name: str
    vertices: np.ndarray        # (V, 3) float32
    faces: np.ndarray           # (F, 3) int32
    normals: Optional[np.ndarray] = None  # (V, 3)
    volume_mm3: float = 0.0
    centroid_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_tumor: bool = False

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def to_dict(self) -> Dict:
        return {
            "label_id": self.label_id,
            "name": self.name,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "volume_mm3": round(self.volume_mm3, 2),
            "centroid_mm": self.centroid_mm.tolist(),
            "is_tumor": self.is_tumor,
        }


@dataclass
class BrainDigitalTwin:
    """
    Complete 3D digital twin of a patient's brain.
    Contains meshes for all segmented structures.
    """
    patient_id: str
    meshes: Dict[str, AnatomicalMesh]  # name → mesh
    voxel_spacing: Tuple[float, float, float]
    affine: Optional[np.ndarray] = None  # voxel → world transform

    @property
    def tumor_meshes(self) -> Dict[str, AnatomicalMesh]:
        return {k: v for k, v in self.meshes.items() if v.is_tumor}

    @property
    def total_tumor_volume_mm3(self) -> float:
        return sum(m.volume_mm3 for m in self.tumor_meshes.values())

    def summary(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "structures": [m.to_dict() for m in self.meshes.values()],
            "tumor_count": len(self.tumor_meshes),
            "total_tumor_volume_mm3": round(self.total_tumor_volume_mm3, 2),
            "voxel_spacing": self.voxel_spacing,
        }


class BrainReconstructionPipeline:
    """
    Convert binary segmentation mask → full 3D anatomical meshes.

    Pipeline:
        mask → per-label binary → Gaussian smooth → marching cubes
             → mesh decimation → centroid & volume computation
             → BrainDigitalTwin
    """

    TUMOR_LABELS = {1, 2, 3}  # necrotic core, edema, enhancing

    def __init__(
        self,
        smooth_sigma: float = 1.0,
        marching_cubes_level: float = 0.5,
        min_voxels: int = 50,
    ):
        self.smooth_sigma = smooth_sigma
        self.mc_level = marching_cubes_level
        self.min_voxels = min_voxels

    def reconstruct(
        self,
        segmentation_result: Dict,
        patient_id: str = "unknown",
    ) -> BrainDigitalTwin:
        """
        Build a full BrainDigitalTwin from segmentation output.

        Args:
            segmentation_result: output of BrainTumorSegmenter.segment()
            patient_id: patient identifier

        Returns:
            BrainDigitalTwin with meshes for each structure
        """
        mask = segmentation_result["mask"]
        spacing = segmentation_result.get("voxel_spacing", (1.0, 1.0, 1.0))
        labels = segmentation_result.get("labels", {})
        structures = segmentation_result.get("structures", [])

        logger.info(f"Reconstructing 3D twin for patient {patient_id} | "
                    f"shape={mask.shape} | structures={len(structures)}")

        meshes = {}
        for struct in structures:
            label_id = struct["label_id"]
            name = struct["name"]
            voxel_count = struct["voxel_count"]

            if voxel_count < self.min_voxels:
                logger.debug(f"Skipping {name} — too few voxels ({voxel_count})")
                continue

            mesh = self._mesh_from_label(mask, label_id, name, spacing)
            if mesh is not None:
                meshes[name] = mesh

        twin = BrainDigitalTwin(
            patient_id=patient_id,
            meshes=meshes,
            voxel_spacing=spacing,
        )

        logger.info(f"Digital twin built: {len(meshes)} structures, "
                    f"total tumor volume = {twin.total_tumor_volume_mm3:.1f} mm³")
        return twin

    def _mesh_from_label(
        self,
        mask: np.ndarray,
        label_id: int,
        name: str,
        spacing: Tuple[float, float, float],
    ) -> Optional[AnatomicalMesh]:
        """Extract a single structure mesh via marching cubes."""
        binary = (mask == label_id).astype(np.float32)

        # Fill holes for cleaner meshes
        binary_int = binary.astype(bool)
        binary_int = binary_fill_holes(binary_int)

        # Gaussian smoothing for smoother surface
        smoothed = gaussian_filter(binary_int.astype(np.float32), sigma=self.smooth_sigma)

        try:
            verts, faces, normals, _ = marching_cubes(
                smoothed,
                level=self.mc_level,
                spacing=spacing,
                allow_degenerate=False,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Marching cubes failed for {name}: {e}")
            return None

        if len(verts) < 4 or len(faces) < 4:
            return None

        # Compute volume via signed mesh volume (divergence theorem)
        volume = self._compute_mesh_volume(verts, faces)
        centroid = verts.mean(axis=0)

        return AnatomicalMesh(
            label_id=label_id,
            name=name,
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32) if normals is not None else None,
            volume_mm3=abs(volume),
            centroid_mm=centroid,
            is_tumor=(label_id in self.TUMOR_LABELS),
        )

    def _compute_mesh_volume(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
    ) -> float:
        """Compute signed mesh volume via divergence theorem."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        signed_vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
        return float(signed_vol)

    def export_obj(
        self,
        twin: BrainDigitalTwin,
        output_dir: str,
        structures: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Export meshes to .obj files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exported = {}
        targets = structures or list(twin.meshes.keys())

        for name in targets:
            if name not in twin.meshes:
                continue
            mesh = twin.meshes[name]
            path = out_dir / f"{twin.patient_id}_{name}.obj"

            with open(path, "w") as f:
                f.write(f"# {name} — patient {twin.patient_id}\n")
                f.write(f"# Vertices: {mesh.vertex_count}, Faces: {mesh.face_count}\n\n")
                for v in mesh.vertices:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                if mesh.normals is not None:
                    for n in mesh.normals:
                        f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
                for face in mesh.faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            exported[name] = str(path)
            logger.info(f"Exported {name} → {path}")

        return exported

    def to_pytorch3d_meshes(
        self,
        twin: BrainDigitalTwin,
        device: str = "cpu",
    ) -> Optional["Meshes"]:
        """Convert all meshes to a PyTorch3D Meshes batch."""
        if not P3D_AVAILABLE:
            logger.warning("PyTorch3D not available")
            return None

        import torch
        verts_list, faces_list = [], []
        for mesh in twin.meshes.values():
            verts_list.append(torch.tensor(mesh.vertices, dtype=torch.float32))
            faces_list.append(torch.tensor(mesh.faces, dtype=torch.int64))

        return Meshes(verts=verts_list, faces=faces_list).to(device)
