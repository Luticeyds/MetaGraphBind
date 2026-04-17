# MetaGraphBind

MetaGraphBind is a machine-learning framework for ligand design targeting intrinsic Am³⁺/Eu³⁺ coordination discrimination.

This repository contains two main components:

- **MetaGraphBind**: a graph neural network (GNN) model for predicting first-step metal–ligand stability constants (`logK1`)
- **L-MolGAN**: a scaffold-preserving molecular generation module guided by predicted `ΔlogK1`

This code accompanies the manuscript:

**Machine-Learning-Guided Ligand Optimization for Americium/Europium Coordination Discrimination**

---

## Repository Structure

```text
MetaGraphBind/
├── README.md
├── requirements.txt
├── MetaGraphBind/
│   ├── Dataset/
│   ├── Graph/
│   ├── data/
│   ├── model/
│   ├── trainer/
│   ├── net/
│   ├── main.py
│   └── scaler.joblib
└── L-MolGAN/
    ├── data/
    ├── model/
    ├── main.py
    ├── utils.py
    ├── solver.py
    ├── layers.py
    └── models.py

### Main folders

- `MetaGraphBind/`: GNN-based affinity prediction for `logK1`
- `MetaGraphBind/Dataset/`: dataset loading and preprocessing
- `MetaGraphBind/model/`: model definition
- `MetaGraphBind/trainer/`: training and evaluation utilities
- `MetaGraphBind/net/`: saved checkpoints
- `L-MolGAN/`: scaffold-preserving molecular generation guided by predicted `ΔlogK1`
- `L-MolGAN/data/`: generation datasets and processed inputs
- `L-MolGAN/model/`: generative model components


Installation

Create a Python environment and install dependencies:
pip install -r requirements.txt

Data

Three supervised datasets are used in the transfer-learning workflow:
data/pretrain_jess_general.xlsx
data/transfer_lnan_subset.xlsx
data/finetune_article_dataset.xlsx

Continuous auxiliary features are scaled using a pre-fitted scaler.joblib.
The scaler should be fitted on the designated training data only and then applied to validation, test, fine-tuning, and unlabeled data via transform without refitting.

Training

After updating MetaGraphBind/main.py to support the pipeline mode, the full three-stage workflow can be run with:
cd MetaGraphBind
python main.py --mode pipeline
This performs:
pretraining on the general metal–ligand dataset
transfer learning on the Ln/An subset
final fine-tuning on the Am/Eu phenanthroline dataset
You can also run each stage separately:
python main.py --mode pretrain
python main.py --mode transfer1
python main.py --mode transfer2
python main.py --mode val

Ligand Generation
The L-MolGAN module is used for reward-guided ligand generation.
Example:
cd L-MolGAN
python main.py
Notes
Please avoid hard-coded file paths.
All comments and documentation should be in English.
The public repository is being organized for reproducibility.
Checkpoints are typically saved in MetaGraphBind/net/.
Recommended checkpoint names:
pretrain_general.pt
transfer_lnan.pt
finetune_am_eu.pt

Citation
If you use this repository, please cite the associated manuscript:
Dongsheng Yang, Zhiyuan Zhang, Yulong Que, Yihuang Wu, Tongxin Yu, Shiyi Jiang, Chong Liu
Machine-Learning-Guided Ligand Optimization for Americium/Europium Coordination Discrimination

Contact
Chong Liu
School of Chemical Engineering, Sichuan University
Email: liuchong@scu.edu.cn