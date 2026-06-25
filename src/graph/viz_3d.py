"""
src/graph/viz_3d.py
3D Brain Mesh Visualization using Plotly.
Generates interactive 3D figures showing:
  - Semi-transparent brain hemisphere
  - Tumor ellipsoid (colored by size/risk)
  - Critical anatomical structures as spheres
  - Safest surgical approach corridor
  - Highlighted structures in selected surgical plan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available — 3D viz disabled")


# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    "brain":           "rgba(200, 180, 160, 0.15)",   # beige, very transparent
    "brain_edge":      "rgba(180, 160, 140, 0.3)",
    "tumor":           "rgba(220, 50, 50, 0.75)",      # red
    "tumor_edge":      "rgba(180, 20, 20, 0.9)",
    "brainstem":       "rgba(255, 140, 0, 0.8)",       # orange
    "vessel":          "rgba(220, 20, 120, 0.8)",      # crimson
    "edema":           "rgba(255, 200, 50, 0.4)",      # yellow
    "safe_path":       "rgba(50, 200, 100, 0.9)",      # green
    "unsafe_zone":     "rgba(220, 50, 50, 0.3)",       # red transparent
    "highlight":       "rgba(124, 58, 237, 0.9)",      # purple (plan highlight)
    "normal":          "rgba(100, 150, 200, 0.6)",     # blue-grey
    "background":      "#0f0f1a",
    "paper":           "#0f0f1a",
    "grid":            "#1a1a2e",
    "text":            "#e2e8f0",
}

# Critical structures that always get special coloring
CRITICAL_STRUCTURES = {
    "brainstem":              COLORS["brainstem"],
    "internal_carotid_artery": COLORS["vessel"],
    "basilar_artery":         COLORS["vessel"],
    "middle_cerebral_artery": COLORS["vessel"],
    "optic_nerve":            "rgba(255, 215, 0, 0.8)",
    "facial_nerve":           "rgba(255, 215, 0, 0.8)",
}


@dataclass
class Structure3D:
    """A 3D anatomical structure for visualization."""
    name:      str
    centroid:  Tuple[float, float, float]   # (x, y, z) in mm
    radius_mm: float                        # approximate radius
    is_tumor:  bool = False
    is_critical: bool = False
    color:     Optional[str] = None


class BrainMeshBuilder:
    """
    Builds interactive 3D Plotly figures from anatomical twin summaries.

    Usage:
        builder = BrainMeshBuilder()
        fig = builder.build_figure(twin_summary, top_plan=plan_dict)
        st.plotly_chart(fig, use_container_width=True)
    """

    BRAIN_RADIUS_MM = 80.0   # approximate hemisphere radius

    def __init__(self, dark_theme: bool = True):
        self.dark_theme = dark_theme

    # ─── Main entry point ────────────────────────────────────────────────────
    def build_figure(
        self,
        twin_summary: Dict,
        top_plan: Optional[Dict] = None,
        highlight_plan_idx: int = 0,
        show_approach: bool = True,
    ) -> "go.Figure":
        """
        Build full 3D brain figure.

        Args:
            twin_summary: output of DigitalTwin.summary()
            top_plan: dict with 'actions' key (plan to visualize)
            highlight_plan_idx: which plan in top_plans to highlight
            show_approach: whether to draw the surgical approach corridor
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        # Parse structures from twin_summary
        structures = self._parse_structures(twin_summary)

        # 1. Brain hemisphere (transparent sphere)
        self._add_brain_hemisphere(fig)

        # 2. Edema zone (if present)
        edema_structs = [s for s in structures if "edema" in s.name.lower()]
        for s in edema_structs:
            self._add_ellipsoid(
                fig, s.centroid, s.radius_mm * 1.4,
                color=COLORS["edema"], name=s.name, opacity=0.3
            )

        # 3. Tumor (solid, colored by size)
        tumors = [s for s in structures if s.is_tumor]
        for i, tumor in enumerate(tumors):
            risk_color = self._tumor_color(tumor.radius_mm)
            self._add_ellipsoid(
                fig, tumor.centroid, tumor.radius_mm,
                color=risk_color, name=f"Tumor ({tumor.name})",
                opacity=0.85, showlegend=True
            )

        # 4. Critical structures
        for s in structures:
            if not s.is_tumor and not "edema" in s.name.lower():
                color = CRITICAL_STRUCTURES.get(s.name, COLORS["normal"])
                self._add_structure_sphere(
                    fig, s.centroid, s.radius_mm,
                    color=color, name=s.name,
                    is_critical=s.is_critical,
                )

        # 5. Surgical approach corridor
        if show_approach and tumors:
            plan_actions = top_plan.get("actions", []) if top_plan else []
            if plan_actions:
                self._add_approach_corridor(fig, tumors[0].centroid, plan_actions, structures)

        # 6. Highlighted plan structures
        if top_plan:
            self._add_plan_highlight(fig, top_plan, structures)

        # Layout
        self._apply_layout(fig)

        return fig

    def build_simple_figure(
        self,
        tumor_size: float = 0.3,
        tumor_position: str = "frontal",
        critical_proximity: float = 0.5,
    ) -> "go.Figure":
        """
        Build a simple 3D figure from SCM parameters (no twin_summary needed).
        Used in the Streamlit dashboard when no real imaging data exists.
        """
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        # Brain
        self._add_brain_hemisphere(fig)

        # Tumor based on size
        tumor_radius = max(5.0, tumor_size * 40.0)
        positions = {
            "frontal":   (30.0, 50.0, 30.0),
            "temporal":  (60.0, 10.0, 20.0),
            "parietal":  (10.0, 30.0, 60.0),
            "occipital": (-30.0, -40.0, 20.0),
        }
        centroid = positions.get(tumor_position, positions["frontal"])
        risk_color = self._tumor_color(tumor_radius)
        self._add_ellipsoid(
            fig, centroid, tumor_radius,
            color=risk_color, name=f"Tumor ({tumor_size:.0%} normalized)",
            opacity=0.85, showlegend=True
        )

        # Critical structures
        critical_positions = [
            ((0.0, -20.0, -40.0), "Brainstem", COLORS["brainstem"], 12.0),
            ((25.0, 15.0, -10.0), "Internal Carotid A.", COLORS["vessel"], 4.0),
            ((-25.0, 15.0, -10.0), "Internal Carotid A. (L)", COLORS["vessel"], 4.0),
            ((0.0, -10.0, -30.0), "Basilar Artery", COLORS["vessel"], 3.0),
        ]
        for pos, name, color, radius in critical_positions:
            self._add_structure_sphere(fig, pos, radius, color=color,
                                       name=name, is_critical=True)

        # Approach corridor (simplified)
        approach_entry = (centroid[0] * 1.8, centroid[1] * 0.5, centroid[2] + 60)
        self._add_line(
            fig, approach_entry, centroid,
            color=COLORS["safe_path"], name="Approach Corridor",
            dash="dash", width=4
        )

        self._apply_layout(fig)
        return fig

    # ─── Building blocks ─────────────────────────────────────────────────────
    def _parse_structures(self, twin_summary: Dict) -> List[Structure3D]:
        """Parse structures from twin summary dict."""
        structures = []
        raw = twin_summary.get("structures", [])

        for s in raw:
            centroid = s.get("centroid_mm") or s.get("centroid_voxel") or [0, 0, 0]
            # Estimate radius from volume
            volume = s.get("volume_mm3", 500.0)
            radius = (3 * volume / (4 * np.pi)) ** (1/3)
            radius = float(np.clip(radius, 2.0, 45.0))

            structures.append(Structure3D(
                name=s.get("name", "unknown"),
                centroid=tuple(float(c) for c in centroid[:3]),
                radius_mm=radius,
                is_tumor=s.get("is_tumor", False),
                is_critical=s.get("name", "") in CRITICAL_STRUCTURES,
                color=CRITICAL_STRUCTURES.get(s.get("name", ""), None),
            ))

        return structures

    def _add_brain_hemisphere(self, fig: "go.Figure"):
        """Add a semi-transparent brain hemisphere."""
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        r = self.BRAIN_RADIUS_MM

        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.07,
            colorscale=[[0, COLORS["brain"]], [1, COLORS["brain_edge"]]],
            showscale=False,
            name="Brain",
            hoverinfo="name",
            lighting=dict(ambient=0.8, diffuse=0.5, roughness=0.5),
        ))

    def _add_ellipsoid(
        self, fig: "go.Figure", centroid: Tuple,
        radius: float, color: str, name: str,
        opacity: float = 0.8, showlegend: bool = False
    ):
        """Add a solid ellipsoid (tumor/edema) to the figure."""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        cx, cy, cz = centroid

        x = cx + radius * np.outer(np.cos(u), np.sin(v))
        y = cy + radius * np.outer(np.sin(u), np.sin(v))
        z = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=opacity,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            showlegend=showlegend,
            hovertemplate=f"<b>{name}</b><br>Centroid: ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm<br>Radius: {radius:.0f} mm<extra></extra>",
        ))

    def _add_structure_sphere(
        self, fig: "go.Figure", centroid: Tuple,
        radius: float, color: str, name: str, is_critical: bool = False
    ):
        """Add an anatomical structure as a sphere."""
        cx, cy, cz = centroid
        marker_size = max(8, min(20, radius * 0.8))

        # Marker for the structure
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers+text" if is_critical else "markers",
            marker=dict(
                size=marker_size,
                color=color,
                symbol="circle",
                line=dict(color="white", width=1) if is_critical else dict(width=0),
                opacity=0.9,
            ),
            text=[name] if is_critical else [],
            textposition="top center",
            textfont=dict(color=COLORS["text"], size=10),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Position: ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm<extra></extra>",
        ))

    def _add_approach_corridor(
        self, fig: "go.Figure",
        tumor_centroid: Tuple,
        plan_actions: List[str],
        structures: List[Structure3D],
    ):
        """Draw the surgical approach corridor from skull surface to tumor."""
        cx, cy, cz = tumor_centroid

        # Entry point: skull surface in direction of tumor from center
        mag = np.sqrt(cx**2 + cy**2 + (cz + 20)**2)
        if mag < 1e-6:
            entry = (cx, cy, cz + self.BRAIN_RADIUS_MM)
        else:
            scale = self.BRAIN_RADIUS_MM / mag
            entry = (cx * scale, cy * scale, (cz + 20) * scale)

        # Draw approach line
        self._add_line(
            fig, entry, tumor_centroid,
            color=COLORS["safe_path"],
            name=f"Approach: {' → '.join(plan_actions[:2])}",
            dash="dash", width=5
        )

        # Entry point marker
        fig.add_trace(go.Scatter3d(
            x=[entry[0]], y=[entry[1]], z=[entry[2]],
            mode="markers",
            marker=dict(size=10, color=COLORS["safe_path"], symbol="diamond"),
            name="Craniotomy Entry",
            hovertemplate="<b>Craniotomy Entry Point</b><extra></extra>",
        ))

    def _add_plan_highlight(
        self, fig: "go.Figure",
        plan: Dict,
        structures: List[Structure3D],
    ):
        """Highlight structures involved in the surgical plan."""
        actions = plan.get("actions", [])

        # Find structures mentioned in actions
        for s in structures:
            involved = any(
                s.name.lower() in action.lower() or
                "tumor" in action.lower() and s.is_tumor
                for action in actions
            )
            if involved:
                cx, cy, cz = s.centroid
                # Add pulsing ring around involved structures
                fig.add_trace(go.Scatter3d(
                    x=[cx], y=[cy], z=[cz],
                    mode="markers",
                    marker=dict(
                        size=s.radius_mm * 0.9,
                        color=COLORS["highlight"],
                        symbol="circle",
                        opacity=0.3,
                        line=dict(color=COLORS["highlight"], width=3),
                    ),
                    name=f"Surgical target: {s.name}",
                    hovertemplate=f"<b>Surgical Target</b><br>{s.name}<extra></extra>",
                ))

    def _add_line(
        self, fig: "go.Figure",
        start: Tuple, end: Tuple,
        color: str, name: str,
        dash: str = "solid", width: int = 3
    ):
        """Draw a 3D line between two points."""
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name,
            hoverinfo="name",
        ))

    def _tumor_color(self, radius_mm: float) -> str:
        """Color tumor based on size: small=yellow, medium=orange, large=red."""
        if radius_mm < 15:
            return "rgba(255, 200, 50, 0.85)"   # yellow — small
        elif radius_mm < 30:
            return "rgba(255, 120, 0, 0.85)"    # orange — medium
        else:
            return "rgba(220, 30, 30, 0.9)"     # red — large/dangerous

    def _apply_layout(self, fig: "go.Figure"):
        """Apply dark theme 3D layout."""
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="X (mm)", showgrid=True,
                    gridcolor=COLORS["grid"], backgroundcolor=COLORS["background"],
                    color=COLORS["text"],
                ),
                yaxis=dict(
                    title="Y (mm)", showgrid=True,
                    gridcolor=COLORS["grid"], backgroundcolor=COLORS["background"],
                    color=COLORS["text"],
                ),
                zaxis=dict(
                    title="Z (mm)", showgrid=True,
                    gridcolor=COLORS["grid"], backgroundcolor=COLORS["background"],
                    color=COLORS["text"],
                ),
                bgcolor=COLORS["background"],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0, y=0, z=0),
                ),
                aspectmode="cube",
            ),
            paper_bgcolor=COLORS["paper"],
            plot_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"], family="Inter, sans-serif"),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(15, 15, 26, 0.8)",
                bordercolor=COLORS["highlight"],
                borderwidth=1,
                font=dict(color=COLORS["text"]),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=550,
            title=dict(
                text="🧠 3D Brain Anatomy & Surgical Plan",
                font=dict(color=COLORS["text"], size=16),
                x=0.5,
            ),
        )
