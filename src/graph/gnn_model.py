"""
src/graph/gnn_model.py
Anatomical GNN — predicts surgical risk from anatomical graphs.
Architecture: Heterogeneous Graph Transformer with edge-type conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.nn import (
        GATv2Conv,
        GINConv,
        GraphSAGE,
        GraphNorm,
        global_mean_pool,
        global_max_pool,
    )
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("PyTorch Geometric not available — GNN in mock mode.")


@dataclass
class SurgicalRiskPrediction:
    """Output of the Anatomical GNN for a single patient."""
    blood_loss_ml:        float
    nerve_damage_prob:    float
    recovery_score:       float   # 0–1 (higher = better)
    mortality_risk:       float   # 0–1
    icu_days_estimate:    float
    node_risks:           Dict[str, float]  # per-structure risk
    confidence:           float

    def to_dict(self) -> Dict:
        return {
            "blood_loss_ml":     round(self.blood_loss_ml, 1),
            "nerve_damage_prob": round(self.nerve_damage_prob, 4),
            "recovery_score":    round(self.recovery_score, 4),
            "mortality_risk":    round(self.mortality_risk, 4),
            "icu_days_estimate": round(self.icu_days_estimate, 1),
            "node_risks":        {k: round(v, 4) for k, v in self.node_risks.items()},
            "confidence":        round(self.confidence, 4),
        }


# ─── GNN Architecture ─────────────────────────────────────────────────────────

if PYG_AVAILABLE:
    class AnatomicalGATBlock(nn.Module):
        """Graph Attention block with residual connection."""

        def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.conv = GATv2Conv(
                in_channels, out_channels // heads,
                heads=heads, dropout=dropout, add_self_loops=True, concat=True,
            )
            self.norm = GraphNorm(out_channels)
            self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index):
            h = self.conv(x, edge_index)
            h = self.norm(h)
            h = F.elu(h)
            h = self.dropout(h)
            return h + self.proj(x)  # residual

    class AnatomicalGNN(nn.Module):
        """
        Multi-task GNN for surgical risk prediction.

        Inputs:
            x          — (N, F) node feature matrix
            edge_index — (2, E) edge indices
            batch      — (N,) batch assignment (for batched graphs)

        Outputs:
            node_risk_logits  — (N, 1) per-node risk
            blood_loss        — (B,) per-graph blood loss estimate
            nerve_damage_prob — (B,) per-graph nerve damage probability
            recovery_score    — (B,) per-graph recovery score
            mortality_risk    — (B,) per-graph mortality risk
            icu_days          — (B,) per-graph ICU estimate
        """

        def __init__(
            self,
            in_channels: int = 14,     # matches NodeFeatures.to_vector()
            hidden_channels: int = 128,
            num_layers: int = 4,
            heads: int = 4,
            dropout: float = 0.15,
        ):
            super().__init__()

            self.input_proj = nn.Linear(in_channels, hidden_channels)

            self.gat_layers = nn.ModuleList([
                AnatomicalGATBlock(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
                for _ in range(num_layers)
            ])

            self.gin_layer = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                ),
                train_eps=True,
            )

            # Node-level head: per-structure risk
            self.node_risk_head = nn.Sequential(
                nn.Linear(hidden_channels, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

            # Graph-level heads
            graph_in = hidden_channels * 2  # mean + max pool concat

            self.blood_loss_head = nn.Sequential(
                nn.Linear(graph_in, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.nerve_damage_head = nn.Sequential(
                nn.Linear(graph_in, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.recovery_head = nn.Sequential(
                nn.Linear(graph_in, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.mortality_head = nn.Sequential(
                nn.Linear(graph_in, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.icu_head = nn.Sequential(
                nn.Linear(graph_in, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        def forward(self, x, edge_index, batch=None):
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            h = self.input_proj(x)

            for gat in self.gat_layers:
                h = gat(h, edge_index)

            h = self.gin_layer(h, edge_index)

            # Node-level risk
            node_risks = self.node_risk_head(h)

            # Graph-level embedding
            g_mean = global_mean_pool(h, batch)
            g_max  = global_max_pool(h, batch)
            g = torch.cat([g_mean, g_max], dim=-1)

            return {
                "node_risks":        node_risks,
                "blood_loss":        F.relu(self.blood_loss_head(g)),
                "nerve_damage_prob": torch.sigmoid(self.nerve_damage_head(g)),
                "recovery_score":    torch.sigmoid(self.recovery_head(g)),
                "mortality_risk":    torch.sigmoid(self.mortality_head(g)),
                "icu_days":          F.relu(self.icu_head(g)),
            }


# ─── Inference wrapper ────────────────────────────────────────────────────────

class AnatomicalGNNInference:
    """High-level inference wrapper around AnatomicalGNN."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        in_channels: int = 14,
    ):
        self.device = self._resolve_device(device)
        self.in_channels = in_channels

        if PYG_AVAILABLE:
            self.model = AnatomicalGNN(in_channels=in_channels)
            if model_path:
                self._load(model_path)
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
            logger.warning("PyG not available — GNN in mock mode")

    def _resolve_device(self, device: str) -> "torch.device":
        import torch
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load(self, path: str):
        import torch
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        logger.info(f"GNN weights loaded from {path}")

    def predict(
        self,
        feature_matrix: np.ndarray,       # (N, F)
        edge_index: np.ndarray,            # (2, E)
        node_names: List[str],
    ) -> SurgicalRiskPrediction:
        """
        Run GNN inference on an anatomical graph.

        Args:
            feature_matrix: node features (N, F)
            edge_index:     edge connections (2, E)
            node_names:     list of N structure names

        Returns:
            SurgicalRiskPrediction
        """
        if not PYG_AVAILABLE or self.model is None:
            return self._mock_predict(node_names)

        import torch

        x = torch.tensor(feature_matrix, dtype=torch.float32).to(self.device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(self.device)

        with torch.no_grad():
            out = self.model(x, ei)

        node_risks = torch.sigmoid(out["node_risks"]).squeeze(-1).cpu().numpy()
        node_risk_dict = {name: float(risk) for name, risk in zip(node_names, node_risks)}

        return SurgicalRiskPrediction(
            blood_loss_ml=float(out["blood_loss"].item()) * 500,   # scale to mL
            nerve_damage_prob=float(out["nerve_damage_prob"].item()),
            recovery_score=float(out["recovery_score"].item()),
            mortality_risk=float(out["mortality_risk"].item()),
            icu_days_estimate=float(out["icu_days"].item()) * 7,  # scale to days
            node_risks=node_risk_dict,
            confidence=0.72,   # placeholder until calibration
        )

    def _mock_predict(self, node_names: List[str]) -> SurgicalRiskPrediction:
        """Deterministic mock for dev/testing."""
        logger.info("Using MOCK GNN prediction")
        rng = np.random.RandomState(42)

        node_risks = {}
        for name in node_names:
            if "tumor" in name:
                node_risks[name] = rng.uniform(0.6, 0.9)
            elif name in ("brainstem", "internal_carotid_artery"):
                node_risks[name] = rng.uniform(0.3, 0.6)
            else:
                node_risks[name] = rng.uniform(0.05, 0.25)

        return SurgicalRiskPrediction(
            blood_loss_ml=250.0,
            nerve_damage_prob=0.18,
            recovery_score=0.74,
            mortality_risk=0.05,
            icu_days_estimate=3.5,
            node_risks=node_risks,
            confidence=0.60,   # lower confidence for mock
        )
