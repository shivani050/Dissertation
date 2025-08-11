# Datasets
| Dataset       | What                                                  | Link                                                                                                           |
| ------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Juliet / SARD | Synthetic CWE cases (balanced)                        | [https://samate.nist.gov/SARD/](https://samate.nist.gov/SARD/)                                                 |
| Devign        | Real C/C++ functions with binary vulnerability labels | [https://github.com/epicosy/devign](https://github.com/epicosy/devign)                                         |
| Big-Vul       | Real CVE-linked commits                               | [https://www.kaggle.com/datasets/kaggler10240/msr-data](https://www.kaggle.com/datasets/kaggler10240/msr-data) |

This repo does not redistribute datasets; please download from the sources above.

# Models
Sequence CNN (token input): embedding → 1D conv kernels {3,5,7} → global max-pool → dense → sigmoid.

Graph GCN/GraphSAGE (graph input): AST/CFG/DFG nodes with learned embeddings → 3× GraphSAGE → global pool → dense.

# Reproduce results (typical order)
1) Preprocessing (Preprocessing.ipynb)
- Tokenization, padding/truncation for sequence models
- Graph construction (AST/CFG/DFG via Joern), pruning > 500 nodes
- Stratified splits; leakage checks

2) Train sequence model (Trainmodel Updated.ipynb)
- Set batch size / LR / epochs in the notebook header
- Outputs: metrics, PR/ROC, confusion matrix

3) Train graph model (GCN-GNN.ipynb)
- Node-budget batching; gradient accumulation if needed
- Outputs: metrics, PR/ROC, confusion matrix

4) Evaluate + plots
- Figures saved to results/ (F1/Precision/Recall bars, PR curves)
- Optional: average CNN + GCN probabilities to form a simple ensemble.

# Results 
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

Accuracy is dominated by the majority (safe) class; F1/PR give a truer picture.

# Acknowledgements
Joern for code graph extraction; PyTorch + PyTorch Geometric for GNNs; Weights & Biases for experiment tracking.
