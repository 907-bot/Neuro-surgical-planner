"""
src/reports/pdf_generator.py
Clinical-grade PDF report generator for surgical planning output.

Generates a multi-page PDF with:
  Page 1: Patient header, anatomical summary, physiological state
  Page 2: Top 5 surgical plans (color-coded risk table)
  Page 3: Causal attribution for recommended plan
  Footer: Disclaimer on every page
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle, HRFlowable,
        KeepTogether, NextPageTemplate, PageBreak,
    )
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed — PDF export disabled. Run: pip install reportlab")


# ─── Design Tokens ───────────────────────────────────────────────────────────
if REPORTLAB_AVAILABLE:
    PURPLE     = HexColor("#7c3aed")
    DARK_BG    = HexColor("#1e1e2e")
    LIGHT_BG   = HexColor("#f8f7ff")
    RED        = HexColor("#ef4444")
    AMBER      = HexColor("#f59e0b")
    GREEN      = HexColor("#22c55e")
    GREY       = HexColor("#6b7280")
    WHITE      = colors.white
    BLACK      = colors.black
    LIGHT_GREY = HexColor("#e5e7eb")
    RISK_COLORS = {
        "HIGH":   HexColor("#fef2f2"),
        "MEDIUM": HexColor("#fffbeb"),
        "LOW":    HexColor("#f0fdf4"),
    }
    RISK_BORDER = {
        "HIGH":   RED,
        "MEDIUM": AMBER,
        "LOW":    GREEN,
    }


class PDFReportGenerator:
    """
    Generates a clinical-grade PDF surgical planning report.

    Usage:
        generator = PDFReportGenerator()
        pdf_bytes = generator.generate(
            patient_id="P001",
            baseline_scm={...},
            top_plans=[...],
            gnn_prediction={...},
            attribution_chain={...},   # optional
            graph_summary={...},       # optional
        )
        # In Streamlit:
        st.download_button("📄 Download PDF", pdf_bytes, "report.pdf", "application/pdf")
    """

    PAGE_W, PAGE_H = A4
    MARGIN = 2.0 * cm

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab"
            )
        self._setup_styles()

    def _setup_styles(self):
        base = getSampleStyleSheet()

        self.styles = {
            "title": ParagraphStyle(
                "ReportTitle",
                fontSize=22, fontName="Helvetica-Bold",
                textColor=PURPLE, alignment=TA_CENTER,
                spaceAfter=4,
            ),
            "subtitle": ParagraphStyle(
                "Subtitle",
                fontSize=11, fontName="Helvetica",
                textColor=GREY, alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "section": ParagraphStyle(
                "Section",
                fontSize=13, fontName="Helvetica-Bold",
                textColor=PURPLE, spaceBefore=12, spaceAfter=4,
                borderPad=4,
            ),
            "body": ParagraphStyle(
                "Body",
                fontSize=9, fontName="Helvetica",
                textColor=BLACK, leading=14, spaceAfter=2,
            ),
            "small": ParagraphStyle(
                "Small",
                fontSize=8, fontName="Helvetica",
                textColor=GREY, leading=12,
            ),
            "disclaimer": ParagraphStyle(
                "Disclaimer",
                fontSize=7.5, fontName="Helvetica-Oblique",
                textColor=GREY, alignment=TA_CENTER,
            ),
            "plan_rank": ParagraphStyle(
                "PlanRank",
                fontSize=11, fontName="Helvetica-Bold",
                textColor=PURPLE,
            ),
            "mono": ParagraphStyle(
                "Mono",
                fontSize=8, fontName="Courier",
                textColor=HexColor("#374151"), leading=12,
            ),
        }

    def generate(
        self,
        patient_id: str = "UNKNOWN",
        baseline_scm: Optional[Dict] = None,
        top_plans: Optional[List[Dict]] = None,
        gnn_prediction: Optional[Dict] = None,
        attribution_chain: Optional[Dict] = None,
        graph_summary: Optional[Dict] = None,
        patient_params: Optional[Dict] = None,
    ) -> bytes:
        """Generate and return the PDF as bytes."""

        baseline_scm   = baseline_scm or {}
        top_plans      = top_plans or []
        gnn_prediction = gnn_prediction or {}
        graph_summary  = graph_summary or {}
        patient_params = patient_params or {}

        buf = io.BytesIO()
        doc = BaseDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=self.MARGIN + 1.5 * cm,
            bottomMargin=self.MARGIN + 1.5 * cm,
            title=f"Surgical Plan — {patient_id}",
            author="Brain Tumor Surgical Planner v0.1",
        )

        # Build page templates with header/footer
        frame = Frame(
            self.MARGIN, self.MARGIN + 1.5 * cm,
            self.PAGE_W - 2 * self.MARGIN,
            self.PAGE_H - 2 * self.MARGIN - 3 * cm,
            id="main",
        )

        def _header_footer(canvas, doc):
            self._draw_header(canvas, doc, patient_id)
            self._draw_footer(canvas, doc)

        template = PageTemplate(
            id="main_template",
            frames=[frame],
            onPage=_header_footer,
        )
        doc.addPageTemplates([template])

        story = []

        # ── Page 1: Summary ───────────────────────────────────────────────────
        story += self._build_cover(patient_id, baseline_scm, gnn_prediction,
                                   graph_summary, patient_params)
        story.append(PageBreak())

        # ── Page 2: Surgical Plans ────────────────────────────────────────────
        story += self._build_plans_page(top_plans)
        story.append(PageBreak())

        # ── Page 3: Causal Attribution ────────────────────────────────────────
        story += self._build_attribution_page(attribution_chain, top_plans)

        doc.build(story)
        return buf.getvalue()

    # ─── Page Builders ────────────────────────────────────────────────────────
    def _build_cover(
        self, patient_id: str, baseline: Dict,
        gnn: Dict, graph: Dict, params: Dict
    ) -> list:
        story = []
        S = self.styles

        story.append(Paragraph("🧠 Causal Brain Tumor Surgical Plan", S["title"]))
        story.append(Paragraph(
            f"Patient: <b>{patient_id}</b> &nbsp;|&nbsp; "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            S["subtitle"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=12))

        # Patient Parameters
        if params:
            story.append(Paragraph("Patient Parameters (SCM Input)", S["section"]))
            param_data = [["Parameter", "Value"]] + [
                [k.replace("_", " ").title(), f"{v:.2f}"]
                for k, v in params.items()
            ]
            story.append(self._make_table(param_data, col_widths=[9*cm, 4*cm]))
            story.append(Spacer(1, 10))

        # Physiological baseline
        story.append(Paragraph("Baseline Physiological State (SCM)", S["section"]))
        if baseline:
            phys_data = [["Variable", "Value", "Status"]]
            for k, v in baseline.items():
                label = k.replace("_", " ").title()
                val_str = f"{v:.1%}" if 0 <= v <= 1.01 else f"{v:.3f}"
                # Traffic light status
                if k in {"intracranial_pressure", "surgical_risk", "edema_volume"}:
                    status = "⚠ HIGH" if v > 0.4 else ("MODERATE" if v > 0.2 else "✓ OK")
                elif k in {"blood_flow", "neural_function", "recovery_score"}:
                    status = "✓ OK" if v > 0.6 else ("MODERATE" if v > 0.4 else "⚠ LOW")
                else:
                    status = "—"
                phys_data.append([label, val_str, status])

            story.append(self._make_table(phys_data, col_widths=[8*cm, 3*cm, 3*cm]))
            story.append(Spacer(1, 10))

        # GNN Risk Assessment
        story.append(Paragraph("GNN Risk Assessment", S["section"]))
        if gnn:
            risk_data = [
                ["Metric", "Predicted Value"],
                ["Blood Loss Estimate",  f"{gnn.get('blood_loss_ml', 0):.0f} mL"],
                ["Nerve Damage Prob.",   f"{gnn.get('nerve_damage_prob', 0):.1%}"],
                ["Recovery Score",       f"{gnn.get('recovery_score', 0):.1%}"],
                ["Mortality Risk",       f"{gnn.get('mortality_risk', 0):.2%}"],
                ["ICU Days Estimate",    f"{gnn.get('icu_days_estimate', 0):.1f}"],
                ["Model Confidence",     f"{gnn.get('confidence', 0):.0%}"],
            ]
            story.append(self._make_table(risk_data, col_widths=[9*cm, 5*cm]))
        else:
            story.append(Paragraph("GNN prediction not available.", S["body"]))

        return story

    def _build_plans_page(self, top_plans: List[Dict]) -> list:
        story = []
        S = self.styles

        story.append(Paragraph(
            "Top Surgical Plans — Monte-Carlo Counterfactual Search", S["section"]))
        story.append(Paragraph(
            f"Generated from {len(top_plans)} ranked plans via Pearl Do-Calculus × Monte-Carlo simulation.",
            S["body"]
        ))
        story.append(Spacer(1, 8))

        for plan in top_plans:
            risk_level = plan.get("risk_level", "MEDIUM")
            actions_str = " → ".join(plan.get("actions", []))

            bg_color = RISK_COLORS.get(risk_level, RISK_COLORS["MEDIUM"])
            border_color = RISK_BORDER.get(risk_level, AMBER)

            plan_data = [
                [f"Rank #{plan.get('rank', '?')}  [{risk_level} RISK]", ""],
                ["Actions",          actions_str],
                ["Expected Recovery", f"{plan.get('expected_recovery', 0):.1%}"],
                ["Expected Risk",     f"{plan.get('expected_risk', 0):.1%}"],
                ["Net Utility",       f"{plan.get('net_utility', 0):+.4f}"],
                ["Blood Loss",        f"{plan.get('blood_loss_ml', 0):.0f} mL"],
                ["Nerve Damage Prob", f"{plan.get('nerve_damage_prob', 0):.1%}"],
                ["ICU Days",          f"{plan.get('icu_days', 0):.1f}"],
                ["95% CI Recovery",   f"{plan.get('confidence_95', [0,0])[0]:.1%} – "
                                       f"{plan.get('confidence_95', [0,0])[1]:.1%}"],
            ]

            t = Table(plan_data, colWidths=[6*cm, 10*cm])
            t.setStyle(TableStyle([
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
                ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, 0), 10),
                ("SPAN",       (0, 0), (-1, 0)),
                # Body
                ("BACKGROUND", (0, 1), (-1, -1), bg_color),
                ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 1), (-1, -1), 9),
                ("GRID",       (0, 0), (-1, -1), 0.5, LIGHT_GREY),
                ("BOX",        (0, 0), (-1, -1), 2, border_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [bg_color, WHITE]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))

            story.append(KeepTogether([t, Spacer(1, 10)]))

        return story

    def _build_attribution_page(
        self, attribution: Optional[Dict], top_plans: List[Dict]
    ) -> list:
        story = []
        S = self.styles

        story.append(Paragraph("Causal Attribution — Why This Plan?", S["section"]))
        story.append(Paragraph(
            "The table below shows how each surgical action causally affects "
            "downstream physiological variables via Pearl's Do-Calculus.",
            S["body"]
        ))
        story.append(Spacer(1, 8))

        if attribution and attribution.get("variable_deltas"):
            # Attribution table
            attr_data = [["Variable", "Before", "After", "Change", "Effect"]]
            for d in attribution["variable_deltas"]:
                arrow = "↑" if d["delta"] > 0 else "↓"
                effect_icon = "✓" if d["direction"] == "improved" else "⚠"
                attr_data.append([
                    d["label"],
                    f"{d['before']:.1%}" if 0 <= d['before'] <= 1.01 else f"{d['before']:.3f}",
                    f"{d['after']:.1%}" if 0 <= d['after'] <= 1.01 else f"{d['after']:.3f}",
                    f"{arrow}{abs(d['delta_pct']):.0f}%",
                    f"{effect_icon} {d['direction'].title()}",
                ])

            t = Table(attr_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3.5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), PURPLE),
                ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("GRID",        (0, 0), (-1, -1), 0.5, LIGHT_GREY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            # Explanation
            explanation = attribution.get("explanation", "")
            if explanation:
                story.append(Paragraph("Causal Chain Summary:", S["section"]))
                story.append(Paragraph(explanation, S["body"]))

        else:
            story.append(Paragraph(
                "Causal attribution not available. Run with attribution_chain parameter.",
                S["body"]
            ))

        # Research disclaimer
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=GREY))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "⚠ IMPORTANT: This report is generated by an AI research system (Brain Tumor Surgical Planner v0.1). "
            "It is intended for RESEARCH PURPOSES ONLY and must NOT be used for clinical decision-making. "
            "All surgical decisions must be made exclusively by qualified neurosurgeons following institutional protocols. "
            "The causal model is trained on synthetic data and has not been validated in clinical settings.",
            S["disclaimer"]
        ))

        return story

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def _make_table(self, data: list, col_widths: Optional[list] = None) -> Table:
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.5, LIGHT_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    def _draw_header(self, canvas, doc, patient_id: str):
        canvas.saveState()
        canvas.setFillColor(PURPLE)
        canvas.rect(0, self.PAGE_H - 1.5 * cm, self.PAGE_W, 1.5 * cm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(WHITE)
        canvas.drawString(self.MARGIN, self.PAGE_H - 1.0 * cm,
                          "🧠 Brain Tumor Surgical Planner — Causal AI Research")
        canvas.drawRightString(
            self.PAGE_W - self.MARGIN, self.PAGE_H - 1.0 * cm,
            f"Patient: {patient_id}  |  Page {doc.page}"
        )
        canvas.restoreState()

    def _draw_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Oblique", 7)
        canvas.setFillColor(GREY)
        canvas.drawCentredString(
            self.PAGE_W / 2, 0.8 * cm,
            "⚠ AI-generated research output — NOT for clinical use. "
            "All decisions must be made by qualified neurosurgeons."
        )
        canvas.restoreState()
