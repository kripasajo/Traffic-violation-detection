Federated Traffic-Violation Detection with DINOv2
A four-stage pipeline that trains two binary classifiers — triple-riding detection and helmet-violation detection — using a frozen DINOv2 backbone and a federated linear probe, without sharing raw video across clients.
---
Overview
Stage	Script	What it does
1	`extract.py`	Downloads RideSafe-400, decodes videos once, and writes labelled JPEG crops for both tasks
2	`features.py`	Passes crops through frozen DINOv2 ViT-B/14 and saves `(X, y, key, vid)` `.npz` files
3	`experiments.py`	Runs probe ablation, centralized baseline, federated FedAvg, DP sweep, and communication-cost analysis; writes a JSON results file
4	`figures.py`	Reads the JSON and writes all paper figures (PDF + PNG) and tables (CSV + Markdown)
---
Quick Start
```bash
# 1. Install dependencies
pip install -r final_run/requirements.txt

# 2. Run the full pipeline (stages 1–4)
cd final_run
bash run_all.sh
```
Outputs land in `final_run/figures/`.
---
Dataset
RideSafe-400 is downloaded automatically from Hugging Face (`DeepBug/RideSafe-400`) during Stage 1. Videos are removed after crops are extracted to save disk space.
Two detection tasks share the same source video, so every video is decoded once:
Triple-riding — unit of evaluation: motorcycle association (one bike + all its riders)
Helmet violation — unit of evaluation: rider track (one rider followed across frames)
The train / val / test split is made at video level and shared by both tasks, so no video that trains one model can test the other.
---
Architecture
```
Local client k
┌──────────────────────────────────┐
│  raw video  →  crops (stay local)│
│  frozen DINOv2 backbone          │  ← never transmitted
│  train linear probe (769 floats) │
└──────────┬───────────────────────┘
           │  probe weights only
           ▼
    Aggregation server (FedAvg)
           │  global weights
           ▼
    All clients (next round)
```
Backbone: DINOv2 ViT-B/14 (`vit_base_patch14_reg4_dinov2.lvd142m`, 768-d + 1 bias = 769 floats per round)
Probe: logistic regression, hyperparameter C selected by `GroupKFold(5)` on train+val only
Aggregation: Flower FedAvg (falls back to NumPy weighted mean if `flwr` is not installed)
Privacy: optional Gaussian mechanism (clip + noise) evaluated across `σ ∈ {0, 0.001, …, 0.1}`
---
Outputs
After `run_all.sh` completes, `final_run/figures/` contains:
File	Content
`fig1_convergence.*`	AP and weight drift per communication round
`fig2_centralized_vs_federated.*`	Federated vs centralized AP across client counts
`fig3_ablation.*`	Effect of standardisation and regularisation on probe AP
`fig4_dataset_composition.*`	Crop counts and class imbalance per split
`fig5_partition_balance.*`	Positive-unit distribution across clients
`fig6_per_client.*`	Per-client: training alone vs federation
`fig7_pr_curves.*`	Precision-recall curves, centralized vs federated
`fig8_communication.*`	Cumulative uplink traffic vs full-model baselines
`fig9_dp_tradeoff.*`	Utility under differential-privacy noise
`fig10_qualitative_*.* `	Ranked TP / TN / FP / FN examples from held-out test
`fig11_architecture.*`	Schematic of the federated setup
`table1_experimental_setup.csv`	Hyperparameters and configuration
`table2_main_results.csv`	Precision / recall / F1 / AP for all runs
`table3_dataset_summary.csv`	Crop and unit counts per split
`results_summary.md`	All tables in Markdown
---
Reproducing Individual Stages
```bash
cd final_run

# Stage 1 only (crop extraction)
python3 extract.py

# Stage 2 — features for triple-riding
python3 features.py --crops crops_triple --out feats/triple \
        --classes normal triple

# Stage 2 — features for helmet detection
python3 features.py --crops crops_helmet --out feats/helmet \
        --classes helmet no_helmet

# Stage 3 — experiments
python3 experiments.py --feats feats/triple --task triple \
        --pos-name "triple riding" --out metrics/triple.json
python3 experiments.py --feats feats/helmet --task helmet \
        --pos-name "no helmet" --out metrics/helmet.json

# Stage 4 — figures
python3 figures.py \
        --metrics metrics/triple.json metrics/helmet.json \
        --crops-triple crops_triple --crops-helmet crops_helmet \
        --out figures
```
---
Key Design Decisions
Letterbox resizing — crops are padded to a square rather than squashed, preserving the aspect-ratio signal that distinguishes a three-rider association from a single-rider one.
Test-time augmentation — horizontal-flip embeddings are averaged into every DINOv2 feature vector.
Unit-level pooling — scores are pooled per tracked object (mean score, majority label) before computing AP and F1, because a real deployed system judges an object once, not once per frame.
Federated standardisation — clients share per-feature `(n, Σx, Σx²)` statistics; the server reconstructs the exact global mean/variance, so no raw features leave the client.
Video-level data splits — the train/val/test split is enforced at source-video level for both tasks, preventing near-duplicate frames from appearing in both training and test sets.
---
Requirements
See `final_run/requirements.txt`.  
`flwr` and `tabulate` are optional; the pipeline falls back gracefully if either is absent.
