"""
src/imaging/segmentation.py
MRI segmentation pipeline using MONAI.
Converts raw NIfTI MRI → labeled organ/tumor masks → 3D meshes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

try:
    import monai
    from monai.data import DataLoader, Dataset
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SwinUNETR, SegResNet, UNet
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        ResizeWithPadOrCropd,
        Spacingd,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logger.warning("MONAI not installed. Using mock segmentation.")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False


# ─── Label map ───────────────────────────────────────────────────────────────
BRAIN_LABELS = {
    0: "background",
    1: "necrotic_tumor_core",
    2: "peritumoral_edema",
    3: "enhancing_tumor",
    4: "white_matter",
    5: "gray_matter",
    6: "csf",
    7: "brainstem",
    8: "cerebellum",
    9: "ventricles",
}

VASCULAR_LABELS = {
    10: "middle_cerebral_artery",
    11: "anterior_cerebral_artery",
    12: "posterior_cerebral_artery",
    13: "basilar_artery",
    14: "internal_carotid_artery",
    15: "dural_sinus",
}

ALL_LABELS = {**BRAIN_LABELS, **VASCULAR_LABELS}


# ─── Preprocessing ────────────────────────────────────────────────────────────
def build_preprocessing_transforms(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Standard BraTS-style MRI preprocessing."""
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=target_spacing,
            mode="bilinear",
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
        EnsureTyped(keys=["image"]),
    ])


# ─── Segmentation Model ───────────────────────────────────────────────────────
class BrainTumorSegmenter:
    """
    Wraps a SwinUNETR (or SegResNet fallback) for brain tumor segmentation.
    Supports BraTS multi-modal input: T1, T1ce, T2, FLAIR.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        model_type: str = "swinunetr",
        num_classes: int = 4,
    ):
        self.num_classes = num_classes
        self.device = self._resolve_device(device)
        self.model = self._build_model(model_type)

        if model_path and Path(model_path).exists():
            self._load_weights(model_path)
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning("No pretrained weights loaded — using random init for dev/test.")

        self.model.to(self.device)
        self.model.eval()

        self.transforms = build_preprocessing_transforms() if MONAI_AVAILABLE else None

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_model(self, model_type: str) -> torch.nn.Module:
        if not MONAI_AVAILABLE:
            return MockSegmentationModel(self.num_classes)

        if model_type == "swinunetr":
            return SwinUNETR(
                in_channels=4,
                out_channels=self.num_classes,
                feature_size=48,
                use_checkpoint=True,
                spatial_dims=3,
            )
        elif model_type == "segresnet":
            return SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=4,
                out_channels=self.num_classes,
                dropout_prob=0.2,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_weights(self, path: str):
        state = torch.load(path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    def segment(
        self,
        mri_paths: Dict[str, str],
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
    ) -> Dict:
        """
        Run segmentation on multi-modal MRI.

        Args:
            mri_paths: {"t1": path, "t1ce": path, "t2": path, "flair": path}
                       OR {"image": combined_4channel_path}
            roi_size:  sliding window size
            overlap:   sliding window overlap

        Returns:
            {
                "mask":   numpy array (H, W, D) with label indices,
                "probs":  numpy array (C, H, W, D) per-class probabilities,
                "labels": dict mapping label_id → structure_name,
                "meta":   image metadata
            }
        """
        # Check if files exist, fallback to mock if they don't
        files_exist = True
        if "image" in mri_paths:
            if mri_paths["image"] == "mock" or not Path(mri_paths["image"]).exists():
                files_exist = False
        else:
            for modality in ["t1", "t1ce", "t2", "flair"]:
                path = mri_paths.get(modality)
                if path and (path == "mock" or not Path(path).exists()):
                    files_exist = False
                    break

        if not MONAI_AVAILABLE or not files_exist:
            return self._mock_segment(mri_paths)

        # Build input tensor
        image_tensor, meta = self._load_multimodal(mri_paths)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=self.model,
                overlap=overlap,
            )

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        mask = np.argmax(probs, axis=0).astype(np.uint8)

        # Populate label 4 (white_matter) using the skull-stripped brain tissue region.
        # Since the MRI is skull-stripped, any voxel that has a non-zero intensity in any MRI modality
        # and is NOT already classified as tumor (labels 1, 2, 3) is healthy brain tissue (white_matter).
        img_np = image_tensor.squeeze(0).cpu().numpy()
        mri_mask = np.max(img_np, axis=0) > 1e-3
        mask[(mask == 0) & mri_mask] = 4

        structures = self._extract_structures(mask)

        return {
            "mask": mask,
            "probs": probs,
            "labels": {i: BRAIN_LABELS.get(i, f"unknown_{i}") for i in np.unique(mask)},
            "structures": structures,
            "meta": meta,
            "shape": mask.shape,
            "voxel_spacing": meta.get("spacing", (1.0, 1.0, 1.0)),
        }

    def _load_multimodal(
        self,
        mri_paths: Dict[str, str],
    ) -> Tuple[torch.Tensor, dict]:
        """Load and stack multi-modal MRI channels."""
        if "image" in mri_paths:
            # Pre-stacked 4-channel volume
            data = self.transforms({"image": mri_paths["image"]})
            return data["image"].unsqueeze(0), data.get("image_meta_dict", {})

        channels = []
        meta = {}
        for modality in ["t1", "t1ce", "t2", "flair"]:
            path = mri_paths.get(modality)
            if path is None:
                channels.append(torch.zeros(1, 128, 128, 128))
                continue
            data = self.transforms({"image": path})
            channels.append(data["image"])
            if not meta:
                meta = data.get("image_meta_dict", {})

        stacked = torch.cat(channels, dim=0).unsqueeze(0)
        return stacked, meta

    def _extract_structures(self, mask: np.ndarray) -> List[Dict]:
        """Extract per-structure stats from the segmentation mask."""
        structures = []
        for label_id, label_name in BRAIN_LABELS.items():
            if label_id == 0:
                continue  # Skip background
            voxels = np.sum(mask == label_id)
            if voxels == 0:
                continue
            coords = np.argwhere(mask == label_id)
            centroid = coords.mean(axis=0).tolist()
            structures.append({
                "label_id": label_id,
                "name": label_name,
                "voxel_count": int(voxels),
                "centroid_voxel": centroid,
                "is_tumor": label_id in (1, 2, 3),
            })
        return structures

    def _mock_segment(self, mri_paths: Dict[str, str]) -> Dict:
        """Return synthetic segmentation for testing without MONAI."""
        logger.info("Using MOCK segmentation (MONAI not available)")
        shape = (128, 128, 128)
        mask = np.zeros(shape, dtype=np.uint8)

        # Synthetic tumor at center
        cx, cy, cz = 64, 64, 64
        for r, label in [(8, 1), (14, 2), (18, 3)]:
            x, y, z = np.ogrid[-cx:shape[0]-cx, -cy:shape[1]-cy, -cz:shape[2]-cz]
            sphere = x**2 + y**2 + z**2 <= r**2
            mask[sphere] = label

        # Brain tissue
        brain_sphere = (
            np.ogrid[-cx:shape[0]-cx, -cy:shape[1]-cy, -cz:shape[2]-cz]
        )
        brain_mask = brain_sphere[0]**2 + brain_sphere[1]**2 + brain_sphere[2]**2 <= 55**2
        mask[brain_mask & (mask == 0)] = 4  # white matter

        return {
            "mask": mask,
            "probs": np.eye(5)[mask.flatten()].reshape(*shape, 5).transpose(3, 0, 1, 2),
            "labels": {0: "background", 1: "necrotic_core", 2: "edema", 3: "enhancing_tumor", 4: "white_matter"},
            "structures": [
                {"label_id": 1, "name": "necrotic_tumor_core", "voxel_count": 512,
                 "centroid_voxel": [64, 64, 64], "is_tumor": True},
                {"label_id": 2, "name": "peritumoral_edema", "voxel_count": 2048,
                 "centroid_voxel": [64, 64, 64], "is_tumor": True},
                {"label_id": 3, "name": "enhancing_tumor", "voxel_count": 1024,
                 "centroid_voxel": [64, 64, 64], "is_tumor": True},
                {"label_id": 4, "name": "white_matter", "voxel_count": int(np.sum(mask == 4)),
                 "centroid_voxel": [64, 64, 64], "is_tumor": False},
            ],
            "meta": {"spacing": (1.0, 1.0, 1.0), "mock": True},
            "shape": shape,
            "voxel_spacing": (1.0, 1.0, 1.0),
        }


class MockSegmentationModel(torch.nn.Module):
    """Dummy model for testing without MONAI."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Conv3d(4, num_classes, 1)

    def forward(self, x):
        return self.conv(x)
