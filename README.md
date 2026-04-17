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
│   ├── test.py
│   └── scaler.joblib
└── L-MolGAN/
    ├── data/
    ├── model/
    ├── main.py
    ├── utils.py
    ├── solver.py
    ├── layers.py
    └── models.py
```
### Main folders

- `MetaGraphBind/`: GNN-based affinity prediction for `logK1`
- `MetaGraphBind/Dataset/`: dataset loading and preprocessing
- `MetaGraphBind/model/`: model definition
- `MetaGraphBind/trainer/`: training and evaluation utilities
- `MetaGraphBind/net/`: saved checkpoints
- `L-MolGAN/`: scaffold-preserving molecular generation guided by predicted `ΔlogK1`
- `L-MolGAN/data/`: generation datasets and processed inputs
- `L-MolGAN/model/`: generative model components

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Three supervised datasets are used in the transfer-learning workflow:

- `data/pretrain_jess_general.xlsx`
- `data/transfer_lnan_subset.xlsx`
- `data/finetune_article_dataset.xlsx`

An unlabeled generated dataset can be provided as:

- `data/generated_ligand.xlsx`

Continuous auxiliary features are scaled in a stage-specific manner. For each supervised stage, the dataset is first split into training, test, and validation subsets, and the scaler is fitted only on the training split. The fitted scaler is then saved and reused to transform validation, test, and unlabeled data without refitting.

---

## Training
After updating `MetaGraphBind/main.py` to support the pipeline mode, the full three-stage workflow can be run with:

```bash
python MetaGraphBind/main.py --mode pipeline
```

This performs:

**1.** pretraining on the general metal–ligand dataset

**2.** transfer learning on the Ln/An subset

**3.** final fine-tuning on the Am/Eu phenanthroline dataset

You can also run each stage separately:

```bash
python MetaGraphBind/main.py --mode pretrain
python MetaGraphBind/main.py --mode transfer1
python MetaGraphBind/main.py --mode transfer2
python MetaGraphBind/main.py --mode val
```
---

## Ligand Generation
The `L-MolGAN` module is used for reward-guided ligand generation.

Example:

```bash
cd L-MolGAN
python main.py
```

---

## Citation
If you use this repository, please cite the associated manuscript:

**Dongsheng Yang, Zhiyuan Zhang, Yulong Que, Yihuang Wu, Tongxin Yu, Shiyi Jiang, Chong Liu**
*Machine-Learning-Guided Ligand Optimization for Americium/Europium Coordination Discrimination*

---

## Contact
**Chong Liu**
School of Chemical Engineering, Sichuan University
Email: liuchong@scu.edu.cn