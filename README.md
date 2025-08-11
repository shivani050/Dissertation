# AI Methods for Detecting Vulnerabilities in Software Systems
> Sequence (CNN) and Graph (GCN/GraphSAGE) models for source-code vulnerability detection on Juliet, Devign, and Big-Vul.

[![Python](https://img.shields.io/badge/python-3.10+-informational)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproducible-yes-success)]()

This repo contains the code, notebooks, and artifacts for my MSc dissertation.  
It includes preprocessing, training, evaluation, and figures (PR/ROC, confusion matrices, and summary tables).

Written By:
Shivani Saudagar Sawant

## 🔧 Quick start

```bash
# 1) Clone
git clone https://github.com/USERNAME/REPO.git
cd REPO

# 2) (Optional) create a venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) (If PyTorch Geometric wheels are needed)
# See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 5) Run notebooks non-interactively (saves executed copies in outputs/)
python -m pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute "Preprocessing.ipynb"       --output "outputs/Preprocessing.run.ipynb"
jupyter nbconvert --to notebook --execute "Trainmodel Updated.ipynb"  --output "outputs/Train_CNN.run.ipynb"
jupyter nbconvert --to notebook --execute "GCN-GNN.ipynb"             --output "outputs/Train_GCN.run.ipynb"

#Datasets:

| Dataset       | What                                                  | Link                                                                                                           |
| ------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Juliet / SARD | Synthetic CWE cases (balanced)                        | [https://samate.nist.gov/SARD/](https://samate.nist.gov/SARD/)                                                 |
| Devign        | Real C/C++ functions with binary vulnerability labels | [https://github.com/epicosy/devign](https://github.com/epicosy/devign)                                         |
| Big-Vul       | Real CVE-linked commits                               | [https://www.kaggle.com/datasets/kaggler10240/msr-data](https://www.kaggle.com/datasets/kaggler10240/msr-data) |

#Models:
# 1) Sequence CNN (token input): embedding → 1D conv kernels {3,5,7} → global max-pool → dense → sigmoid.

# 2) Graph GCN/GraphSAGE (graph input): AST/CFG/DFG nodes with learned embeddings → 3× GraphSAGE → global pool → dense.

#Reproduce results
#Typical order:

# 1) Preprocessing (Preprocessing.ipynb)

Tokenization, padding/truncation for sequence models
Graph construction (AST/CFG/DFG via Joern), pruning >500 nodes
Stratified splits; leakage checks

# 2) Train sequence model (Trainmodel Updated.ipynb)

Batch size / LR / epochs in the notebook header
Outputs: metrics, PR/ROC, confusion matrix

# 3) Train graph model (GCN-GNN.ipynb)

Node-budget batching; gradient accumulation if needed
Outputs: metrics, PR/ROC, confusion matrix

# 4) Evaluate + plots

Figures saved to results/ (F1/Precision/Recall bars, PR curves)

#Optional: average CNN + GCN probabilities to form a simple ensemble.

#Results:

| Model           | Dataset    |       Acc |      Prec |       Rec |        F1 |
| --------------- | ---------- | --------: | --------: | --------: | --------: |
| Baseline DNN    | Juliet     |     95.1% |     88.0% |     55.0% |     68.0% |
| CNN             | Juliet     |     94.0% |     81.0% |     72.0% |     76.0% |
| Hybrid CNN+Feat | Juliet     |     94.5% |     83.0% |     74.0% |     78.0% |
| GCN             | Juliet     |     93.0% |     90.0% |     60.0% |     72.0% |
| Baseline DNN    | Devign     |     90.0% |     60.0% |     20.0% |     30.0% |
| CNN             | Devign     |     91.2% |     70.0% |     45.0% |     55.0% |
| Hybrid CNN+Feat | Devign     |     91.0% |     69.0% |     47.0% |     56.0% |
| **GCN**         | **Devign** | **92.0%** | **68.0%** | **57.0%** | **62.0%** |


# Acknowledgements
---

# 2) `.gitignore` (drop in root)

```gitignore
# python
__pycache__/
*.pyc
.venv/
.env
.ipynb_checkpoints/

# data and artifacts
data/
outputs/
results/*.ckpt
results/*.pt
results/*.pth
wandb/
logs/

# OS/editor
.DS_Store
Thumbs.db



cff-version: 1.2.0
title: AI Methods in Detecting Vulnerabilities in Software Systems: Code and Artifacts
authors:
  - family-names: Sawant
    given-names: Shivani Saudagar
date-released: 2025-08-11
version: v1.0
repository-code: https://github.com/USERNAME/REPO



