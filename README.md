# AI Methods for Detecting Vulnerabilities in Software Systems
> Sequence (CNN) and Graph (GCN/GraphSAGE) models for source-code vulnerability detection on Juliet, Devign, and Big-Vul.

[![Python](https://img.shields.io/badge/python-3.10+-informational)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproducible-yes-success)]()

This repo contains the code, notebooks, and artifacts for my MSc dissertation.  
It includes preprocessing, training, evaluation, and figures (PR/ROC, confusion matrices, and summary tables).

**Author:** Shivani Saudagar Sawant

---

## 🔧 Quick start

```bash
# 1) Clone
git clone https://github.com/shivani050/Dissertation.git
cd Dissertation

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
