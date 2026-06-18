# 🧠 Brain Tumor Surgical Planner — Causal AI MVP

A counterfactual surgical reasoning system built on **Judea Pearl's Do-Calculus**, **3D Anatomical GNNs**, and **Agentic Planning**.

> "Not just predicting outcomes — simulating what *would* happen if the surgeon intervenes at structure X."

---

## Pipeline

```
MRI Input
   ↓
3D Brain Graph (MONAI + PyTorch3D)
   ↓
Tumor Graph Node (PyTorch Geometric)
   ↓
Causal GNN (DoWhy + PyWhy)
   ↓
Do-Calculus Intervention Engine
   ↓
Counterfactual Monte-Carlo Search
   ↓
Top 5 Ranked Surgical Paths
```

---

## Project Structure

```
brain-surgical-planner/
├── src/
│   ├── imaging/          # MRI segmentation & 3D reconstruction
│   ├── graph/            # Anatomical graph construction
│   ├── causal/           # SCM, Do-Calculus, counterfactuals
│   ├── agents/           # LangGraph agentic planning
│   ├── simulation/       # Monte-Carlo surgical path search
│   └── api/              # FastAPI REST interface
├── models/               # Trained GNN checkpoints
├── data/                 # Raw MRI, processed graphs
├── notebooks/            # Research & exploration
├── tests/
├── configs/
└── scripts/
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run segmentation on a sample MRI
python scripts/run_pipeline.py --mri data/raw/sample.nii.gz

# 3. Launch API
uvicorn src.api.main:app --reload

# 4. Launch dashboard
streamlit run scripts/dashboard.py
```

---

## Key Concepts

| Concept | Implementation |
|---|---|
| Anatomical Digital Twin | MONAI segmentation + PyTorch3D mesh |
| Causal Graph | DoWhy SCM with anatomical nodes |
| Surgical Intervention | `do(action)` via Do-Calculus |
| Counterfactual Search | Monte-Carlo tree search over action space |
| Explainability | Neo4j knowledge graph + neurosymbolic reasoning |

---

## Datasets

- [BraTS 2024](https://www.synapse.org/#!Synapse:syn51156910/wiki/) — Brain Tumor Segmentation Challenge
- [TCIA](https://www.cancerimagingarchive.net/) — The Cancer Imaging Archive
- [IXI Dataset](https://brain-development.org/ixi-dataset/) — Healthy brain MRIs

---

## Research Foundation

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Schölkopf, B. et al. (2021). *Toward Causal Representation Learning*
- Kipf & Welling (2017). *Semi-Supervised Classification with GCNs*
- BraTS Challenge Papers (2015–2024)

---

## Status: MVP (Phase 1)

- [x] Project scaffold
- [x] MRI segmentation pipeline
- [x] Anatomical graph construction
- [x] SCM causal model
- [x] Do-Calculus intervention engine
- [x] Counterfactual simulator
- [x] Monte-Carlo surgical path search
- [x] Agentic planner (LangGraph)
- [x] FastAPI interface
- [x] Streamlit dashboard
- [ ] GNN training on BraTS
- [ ] Validation against surgical outcomes
- [ ] SNN real-time physiology layer
# Neuro-surgical-planner
# brain-surgical-planner
