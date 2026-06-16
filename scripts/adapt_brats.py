#!/usr/bin/env python3
"""Adapt BraTS 2023 GLI NIfTI filenames to pipeline-expected modality names.

BraTS 2023 uses:  t1n, t1c, t2w, t2f
Pipeline expects: t1, t1ce, t2, flair

Usage:
    python scripts/adapt_brats.py \\
        --patient-dir data/raw/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000 \\
        --patient-id BRATS001
"""

import shutil
import sys
from pathlib import Path

BRATS_TO_PIPELINE = {
    "t1n": "t1",
    "t1c": "t1ce",
    "t2w": "t2",
    "t2f": "flair",
}


def adapt_braTS_patient(patient_dir: Path, patient_id: str, output_root: Path) -> Path:
    output_dir = output_root / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_src = None
    found = 0

    for fpath in sorted(patient_dir.iterdir()):
        name = fpath.name.lower()
        if not name.endswith(".nii.gz"):
            continue

        for brats_mod, pipe_mod in BRATS_TO_PIPELINE.items():
            if brats_mod in name:
                dest = output_dir / f"{patient_id}_{pipe_mod}.nii.gz"
                shutil.copy2(fpath, dest)
                print(f"  {brats_mod} -> {pipe_mod}  ({dest.name})")
                found += 1
                break
        else:
            if "seg" in name:
                seg_src = fpath

    if seg_src:
        dest = output_dir / f"{patient_id}_seg.nii.gz"
        shutil.copy2(seg_src, dest)
        print(f"  seg -> seg  ({dest.name})")
        found += 1

    print(f"  Copied {found} files to {output_dir}")
    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Adapt BraTS 2023 filenames for pipeline")
    parser.add_argument("--patient-dir", required=True, type=Path, help="BraTS patient directory")
    parser.add_argument("--patient-id", default="BRATS001", help="Output patient ID")
    parser.add_argument("--output-root", default="data/adapted", type=Path, help="Output root")
    args = parser.parse_args()

    patient_dir = args.patient_dir.resolve()
    if not patient_dir.is_dir():
        print(f"Error: {patient_dir} not found")
        sys.exit(1)

    out = adapt_braTS_patient(patient_dir, args.patient_id, args.output_root)
    print(f"\nDone. Point pipeline --mri to {out}")


if __name__ == "__main__":
    main()
