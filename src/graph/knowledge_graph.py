"""
src/graph/knowledge_graph.py
Neurosymbolic layer — stores anatomical + causal knowledge in Neo4j.
Provides symbolic reasoning on top of the GNN predictions.

Neo4j schema:
  (Structure)-[:COMPRESSES]->(Structure)
  (Structure)-[:SUPPLIES_BLOOD]->(Structure)
  (Structure)-[:DRAINS_INTO]->(Structure)
  (Structure)-[:CONTROLS]->(Structure)
  (SurgicalAction)-[:TARGETS]->(Structure)
  (SurgicalAction)-[:RISKS]->(Structure)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not installed — KG in mock mode")


# ─── Anatomical Knowledge Base ────────────────────────────────────────────────
ANATOMICAL_FACTS = [
    # (source, relation, target, properties)
    ("middle_cerebral_artery",   "SUPPLIES_BLOOD",  "white_matter",          {"weight": 0.9}),
    ("middle_cerebral_artery",   "SUPPLIES_BLOOD",  "gray_matter",           {"weight": 0.9}),
    ("anterior_cerebral_artery", "SUPPLIES_BLOOD",  "frontal_lobe",          {"weight": 0.8}),
    ("posterior_cerebral_artery","SUPPLIES_BLOOD",  "cerebellum",            {"weight": 0.85}),
    ("basilar_artery",           "SUPPLIES_BLOOD",  "brainstem",             {"weight": 0.95}),
    ("internal_carotid_artery",  "BRANCHES_INTO",   "middle_cerebral_artery",{"weight": 1.0}),
    ("gray_matter",              "DRAINS_INTO",     "dural_sinus",           {"weight": 0.8}),
    ("white_matter",             "DRAINS_INTO",     "dural_sinus",           {"weight": 0.8}),
    ("brainstem",                "CONTROLS",        "respiration",           {"weight": 1.0}),
    ("brainstem",                "CONTROLS",        "heart_rate",            {"weight": 1.0}),
    ("hypothalamus",             "CONTROLS",        "autonomic_function",    {"weight": 0.9}),
    ("enhancing_tumor",          "COMPRESSES",      "surrounding_tissue",    {"weight": 0.8}),
    ("peritumoral_edema",        "COMPRESSES",      "white_matter",          {"weight": 0.7}),
]

SURGICAL_FACTS = [
    ("remove_tumor_full",    "TARGETS",  "enhancing_tumor",          {"efficacy": 0.9}),
    ("remove_tumor_full",    "TARGETS",  "necrotic_tumor_core",      {"efficacy": 0.9}),
    ("remove_tumor_full",    "RISKS",    "middle_cerebral_artery",   {"risk": 0.3}),
    ("remove_tumor_partial", "TARGETS",  "enhancing_tumor",          {"efficacy": 0.6}),
    ("clamp_artery",         "TARGETS",  "middle_cerebral_artery",   {"efficacy": 0.8}),
    ("clamp_artery",         "RISKS",    "white_matter",             {"risk": 0.7}),
    ("clamp_artery",         "RISKS",    "gray_matter",              {"risk": 0.7}),
    ("drain_csf",            "TARGETS",  "ventricles",               {"efficacy": 0.9}),
    ("radiosurgery",         "TARGETS",  "enhancing_tumor",          {"efficacy": 0.7}),
    ("radiosurgery",         "RISKS",    "surrounding_tissue",       {"risk": 0.15}),
    ("reduce_edema",         "TARGETS",  "peritumoral_edema",        {"efficacy": 0.85}),
]


# ─── Knowledge Graph Client ───────────────────────────────────────────────────
class AnatomicalKnowledgeGraph:
    """
    Neo4j-backed anatomical knowledge graph.
    Falls back to in-memory NetworkX if Neo4j unavailable.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "surgical_planner",
    ):
        self.uri = uri
        self.driver = None
        self._nx_fallback = None

        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {uri}")
            except Exception as e:
                logger.warning(f"Neo4j unavailable ({e}) — using NetworkX fallback")
                self.driver = None

        if not self.driver:
            self._init_nx_fallback()

    def _init_nx_fallback(self):
        import networkx as nx
        G = nx.MultiDiGraph()
        for src, rel, dst, props in ANATOMICAL_FACTS + SURGICAL_FACTS:
            G.add_edge(src, dst, relation=rel, **props)
        self._nx_fallback = G
        logger.info(f"NetworkX KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def populate(self):
        """Load all anatomical and surgical facts into Neo4j."""
        if not self.driver:
            logger.info("Using NetworkX fallback — no Neo4j population needed")
            return

        with self.driver.session() as session:
            # Constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Structure) REQUIRE s.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:SurgicalAction) REQUIRE a.name IS UNIQUE")

            # Anatomical facts
            for src, rel, dst, props in ANATOMICAL_FACTS:
                session.run(f"""
                    MERGE (a:Structure {{name: $src}})
                    MERGE (b:Structure {{name: $dst}})
                    MERGE (a)-[r:{rel}]->(b)
                    SET r += $props
                """, src=src, dst=dst, props=props)

            # Surgical facts
            for src, rel, dst, props in SURGICAL_FACTS:
                session.run(f"""
                    MERGE (a:SurgicalAction {{name: $src}})
                    MERGE (b:Structure {{name: $dst}})
                    MERGE (a)-[r:{rel}]->(b)
                    SET r += $props
                """, src=src, dst=dst, props=props)

            logger.info(f"Populated Neo4j with {len(ANATOMICAL_FACTS)} anatomical + "
                        f"{len(SURGICAL_FACTS)} surgical facts")

    def get_blood_supply_chain(self, structure: str) -> List[str]:
        """Return all structures that supply blood to the given structure."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Structure)-[:SUPPLIES_BLOOD*1..3]->(b:Structure {name: $name})
                    RETURN a.name AS supplier
                """, name=structure)
                return [r["supplier"] for r in result]
        else:
            if not self._nx_fallback:
                return []
            import networkx as nx
            suppliers = []
            for node in self._nx_fallback.nodes():
                try:
                    paths = nx.all_simple_paths(
                        self._nx_fallback, node, structure, cutoff=3
                    )
                    for path in paths:
                        edges = [
                            self._nx_fallback[path[i]][path[i+1]]
                            for i in range(len(path)-1)
                        ]
                        if all(
                            any(d.get("relation") == "SUPPLIES_BLOOD" for d in e.values())
                            for e in edges
                        ):
                            suppliers.append(node)
                            break
                except Exception:
                    pass
            return list(set(suppliers))

    def get_action_risks(self, action: str) -> List[Dict]:
        """Return all structures at risk from a surgical action."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:SurgicalAction {name: $action})-[r:RISKS]->(s:Structure)
                    RETURN s.name AS structure, r.risk AS risk
                """, action=action)
                return [{"structure": r["structure"], "risk": r["risk"]} for r in result]
        else:
            if not self._nx_fallback:
                return []
            risks = []
            for src, dst, data in self._nx_fallback.edges(data=True):
                if src == action and data.get("relation") == "RISKS":
                    risks.append({"structure": dst, "risk": data.get("risk", 0.5)})
            return risks

    def get_compression_chain(self, tumor: str) -> List[str]:
        """Return all structures compressed by a given tumor."""
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (t:Structure {name: $tumor})-[:COMPRESSES*1..2]->(s:Structure)
                    RETURN s.name AS compressed
                """, tumor=tumor)
                return [r["compressed"] for r in result]
        else:
            if not self._nx_fallback:
                return []
            compressed = []
            for src, dst, data in self._nx_fallback.edges(data=True):
                if src == tumor and data.get("relation") == "COMPRESSES":
                    compressed.append(dst)
            return compressed

    def symbolic_reasoning(self, structures_present: List[str], planned_action: str) -> Dict:
        """
        Neurosymbolic reasoning: combine GNN structure presence with KG facts
        to produce symbolic risk explanations.

        Returns:
            {
              "supply_at_risk":   [...],
              "direct_risks":     [...],
              "compressed_by_tumor": [...],
              "critical_path_disrupted": bool,
              "explanation": str,
            }
        """
        supply_at_risk, direct_risks, compressed = [], [], []

        for struct in structures_present:
            if "tumor" in struct.lower():
                compressed.extend(self.get_compression_chain(struct))

        direct_risks = self.get_action_risks(planned_action)

        # Check if blood supply to critical structures is at risk
        critical = ["brainstem", "gray_matter", "white_matter"]
        for crit in critical:
            suppliers = self.get_blood_supply_chain(crit)
            for risk_item in direct_risks:
                if risk_item["structure"] in suppliers:
                    supply_at_risk.append({
                        "critical_structure": crit,
                        "supply_disrupted":  risk_item["structure"],
                        "risk":              risk_item["risk"],
                    })

        critical_disrupted = any(
            r["risk"] > 0.5 for r in direct_risks
            if r["structure"] in critical
        )

        explanation_parts = []
        if compressed:
            explanation_parts.append(
                f"Tumor compresses: {', '.join(set(compressed))}"
            )
        if supply_at_risk:
            for s in supply_at_risk[:2]:
                explanation_parts.append(
                    f"{s['supply_disrupted']} disruption → {s['critical_structure']} at risk"
                )
        if critical_disrupted:
            explanation_parts.append("⚠ Critical structure directly at risk")

        return {
            "supply_at_risk":             supply_at_risk,
            "direct_risks":               direct_risks,
            "compressed_by_tumor":        list(set(compressed)),
            "critical_path_disrupted":    critical_disrupted,
            "explanation":                " | ".join(explanation_parts) or "No critical conflicts detected",
        }

    def close(self):
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
