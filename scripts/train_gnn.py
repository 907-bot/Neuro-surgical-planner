"""
scripts/train_gnn.py
Train the AnatomicalGNN on BraTS dataset.

Usage:
    python scripts/train_gnn.py \
        --data-dir data/processed/graphs \
        --epochs 100 \
        --batch-size 16 \
        --lr 1e-3 \
        --output models/gnn_checkpoint.pt

Data format expected in --data-dir:
    Each .json file is a processed anatomical graph
    (output of AnatomicalGraphBuilder.to_json())
    paired with a labels.json containing ground-truth outcomes.
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import track

console = Console()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch_geometric.data import Data, Batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch / PyG required for training.")


# ─── Dataset ──────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    from torch.utils.data import Dataset as TorchDataset

    class AnatomicalGraphDataset(TorchDataset):
        """
        Loads processed anatomical graphs + surgical outcome labels.

        Expected file structure:
            data_dir/
              patient_001_graph.json
              patient_001_labels.json
              patient_002_graph.json
              ...

        labels.json format:
            {
              "blood_loss_ml": 250.0,
              "nerve_damage_prob": 0.12,
              "recovery_score": 0.78,
              "mortality_risk": 0.05,
              "icu_days": 3.0
            }
        """

        def __init__(self, data_dir: str, split: str = "train", val_ratio: float = 0.15):
            self.data_dir = Path(data_dir)
            self.samples = self._discover_samples()

            # Train/val split
            n = len(self.samples)
            n_val = max(1, int(n * val_ratio))
            if split == "train":
                self.samples = self.samples[n_val:]
            else:
                self.samples = self.samples[:n_val]

            logger.info(f"{split} set: {len(self.samples)} patients")

        def _discover_samples(self):
            graphs = sorted(self.data_dir.glob("*_graph.json"))
            valid = []
            for g in graphs:
                label_path = Path(str(g).replace("_graph.json", "_labels.json"))
                if label_path.exists():
                    valid.append((g, label_path))
            return valid

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            graph_path, label_path = self.samples[idx]

            # Load graph
            with open(graph_path) as f:
                graph_data = json.load(f)

            # Build node feature matrix
            nodes = graph_data["nodes"]
            x = torch.tensor(
                [n.get("feature_vector", [0.0] * 14) for n in nodes],
                dtype=torch.float32
            )

            # Build edge index
            node_id_map = {n["id"]: i for i, n in enumerate(nodes)}
            edges = graph_data.get("links", [])
            if edges:
                src = [node_id_map.get(e["source"], 0) for e in edges]
                dst = [node_id_map.get(e["target"], 0) for e in edges]
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Load labels
            with open(label_path) as f:
                labels = json.load(f)

            y = torch.tensor([
                labels.get("blood_loss_ml", 200.0) / 500.0,   # normalize
                labels.get("nerve_damage_prob", 0.1),
                labels.get("recovery_score", 0.7),
                labels.get("mortality_risk", 0.05),
                labels.get("icu_days", 3.0) / 14.0,            # normalize
            ], dtype=torch.float32)

            return Data(x=x, edge_index=edge_index, y=y)

        @staticmethod
        def collate_fn(batch):
            return Batch.from_data_list(batch)


# ─── Loss ─────────────────────────────────────────────────────────────────────
def multitask_loss(outputs, targets, weights=None):
    """
    Weighted MSE/BCE loss across all prediction heads.

    targets: (B, 5) — [blood_loss_norm, nerve_damage, recovery, mortality, icu_norm]
    """
    if weights is None:
        weights = torch.tensor([0.15, 0.20, 0.30, 0.20, 0.15])

    losses = {
        "blood_loss":    nn.functional.mse_loss(outputs["blood_loss"].squeeze(), targets[:, 0]),
        "nerve_damage":  nn.functional.binary_cross_entropy(
                             outputs["nerve_damage_prob"].squeeze(), targets[:, 1]),
        "recovery":      nn.functional.mse_loss(outputs["recovery_score"].squeeze(), targets[:, 2]),
        "mortality":     nn.functional.binary_cross_entropy(
                             outputs["mortality_risk"].squeeze(), targets[:, 3]),
        "icu_days":      nn.functional.mse_loss(outputs["icu_days"].squeeze(), targets[:, 4]),
    }

    total = sum(w * l for w, l in zip(weights, losses.values()))
    return total, losses


# ─── Training Loop ────────────────────────────────────────────────────────────
def train(args):
    if not TORCH_AVAILABLE:
        console.print("[red]PyTorch not available. Cannot train.[/red]")
        return

    from src.graph.gnn_model import AnatomicalGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Training on: {device}[/bold]")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = AnatomicalGraphDataset(args.data_dir, split="train")
    val_ds   = AnatomicalGraphDataset(args.data_dir, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=AnatomicalGraphDataset.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        collate_fn=AnatomicalGraphDataset.collate_fn
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AnatomicalGNN(
        in_channels=14,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss, _ = multitask_loss(out, batch.y.view(-1, 5))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss, _ = multitask_loss(out, batch.y.view(-1, 5))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss   = np.mean(val_losses) if val_losses else 0
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            console.print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            if epoch % 10 == 0:
                console.print(f"  [green]✓ Saved best model (val={val_loss:.4f})[/green]")

    # Save training history
    hist_path = Path(args.output).parent / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Best val loss: {best_val_loss:.4f}")
    console.print(f"Model saved to: {args.output}")
    console.print(f"History saved to: {hist_path}")


# ─── Synthetic data generation (for testing the training loop) ────────────────
def generate_synthetic_data(output_dir: str, n_patients: int = 50):
    """Generate synthetic graph + label JSON files for testing."""
    from src.imaging.segmentation import BrainTumorSegmenter
    from src.imaging.reconstruction import BrainReconstructionPipeline
    from src.graph.anatomical_graph import AnatomicalGraphBuilder
    from src.graph.gnn_model import AnatomicalGNNInference

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter     = BrainTumorSegmenter()
    reconstructor = BrainReconstructionPipeline()
    rng = np.random.RandomState(42)

    console.print(f"Generating {n_patients} synthetic patients → {out_dir}")

    for i in track(range(n_patients), description="Generating..."):
        pid = f"SYNTH_{i:04d}"

        # Vary tumor size to get diverse labels
        tumor_size_factor = rng.uniform(0.5, 2.0)
        seg = segmenter._mock_segment({})
        # Scale tumor voxels
        for s in seg["structures"]:
            if s["is_tumor"]:
                s["voxel_count"] = int(s["voxel_count"] * tumor_size_factor)

        twin = reconstructor.reconstruct(seg, patient_id=pid)
        summary = twin.summary()

        builder = AnatomicalGraphBuilder()
        builder.build(summary, patient_id=pid)
        graph_json = builder.to_json()
        (out_dir / f"{pid}_graph.json").write_text(graph_json)

        # Synthetic labels correlated with tumor size
        ts = min(tumor_size_factor / 2.0, 1.0)
        labels = {
            "blood_loss_ml":     float(150 + ts * 400 + rng.normal(0, 30)),
            "nerve_damage_prob": float(np.clip(ts * 0.4 + rng.normal(0, 0.05), 0, 1)),
            "recovery_score":    float(np.clip(0.85 - ts * 0.4 + rng.normal(0, 0.05), 0, 1)),
            "mortality_risk":    float(np.clip(ts * 0.15 + rng.normal(0, 0.02), 0, 1)),
            "icu_days":          float(max(1, 2 + ts * 8 + rng.normal(0, 0.5))),
        }
        (out_dir / f"{pid}_labels.json").write_text(json.dumps(labels, indent=2))

    console.print(f"[green]Generated {n_patients} patients in {out_dir}[/green]")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anatomical GNN")
    subparsers = parser.add_subparsers(dest="command")

    # train command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir",        default="data/processed/graphs")
    train_parser.add_argument("--epochs",          type=int, default=100)
    train_parser.add_argument("--batch-size",      type=int, default=8)
    train_parser.add_argument("--lr",              type=float, default=1e-3)
    train_parser.add_argument("--hidden-channels", type=int, default=128)
    train_parser.add_argument("--num-layers",      type=int, default=4)
    train_parser.add_argument("--output",          default="models/gnn_checkpoint.pt")

    # generate command
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--output-dir", default="data/processed/graphs")
    gen_parser.add_argument("--n-patients", type=int, default=50)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate_synthetic_data(args.output_dir, args.n_patients)
    else:
        # Default: generate then train on synthetic data
        console.print("[yellow]No command given — generating synthetic data then training[/yellow]")

        class GenArgs:
            output_dir = "data/processed/graphs"
            n_patients = 50

        class TrainArgs:
            data_dir        = "data/processed/graphs"
            epochs          = 30
            batch_size      = 4
            lr              = 1e-3
            hidden_channels = 64
            num_layers      = 3
            output          = "models/gnn_checkpoint.pt"

        generate_synthetic_data(GenArgs.output_dir, GenArgs.n_patients)
        train(TrainArgs())
