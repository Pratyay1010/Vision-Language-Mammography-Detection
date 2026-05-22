<div align="center">

# Vision-Language-Mammography-Detection

### GroundingDINO-based Open-Vocabulary Mammography Detection  
### CoOp · CoCoOp · Semi-Supervised Vision-Language Alignment

<br>

<p align="center">
  <img src="assets/dataset_overview/mammogram_sample_01.jpg" style="width:180px; height:180px; object-fit:cover;">
  <img src="assets/dataset_overview/mammogram_sample_02.jpg" style="width:180px; height:180px; object-fit:cover;">
  <img src="assets/dataset_overview/mammogram_sample_04.png" style="width:180px; height:180px; object-fit:cover;">
  <img src="assets/dataset_overview/mammogram_sample_06.jpg" style="width:180px; height:180px; object-fit:cover;">
</p>

<br>

![Python](https://img.shields.io/badge/Python-3.10+-111111?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square)
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-Vision--Language-black?style=flat-square)
![Open Vocabulary](https://img.shields.io/badge/Open--Vocabulary-Detection-1f6feb?style=flat-square)
![Research](https://img.shields.io/badge/Research-Computer%20Vision-6e40c9?style=flat-square)

</div>

---

## Overview

**Vision-Language-Mammography-Detection** explores open-vocabulary lesion grounding in mammography using a vision-language training pipeline built on top of **GroundingDINO**.

Core research directions include:

- Open-vocabulary mammography detection
- Vision-language feature alignment
- CoOp prompt learning
- CoCoOp prompt learning
- Hybrid semi-supervised optimization
- Weakly supervised grounding for medical imaging

Designed for modular experimentation and reproducible research workflows.

---

## Repository Structure

```text
src/
├── detection/
├── evaluation/
├── prompt_learning/
├── training/
└── utils/
```

---

## Training Dynamics

<div align="center">

### CoOp Prompt Learning

<img src="assets/training_curves/coop_training_curve_best.png" width="80%">

<br><br>

### CoCoOp Prompt Learning

<img src="assets/training_curves/cocoop_training_curve_best.png" width="80%">

<br><br>

### Hybrid Semi-Supervised Training

<img src="assets/training_curves/hybrid_training_curve.png" width="80%">

</div>

---

## Installation

```bash
git clone https://github.com/your-username/Vision-Language-Mammography-Detection.git

cd Vision-Language-Mammography-Detection

pip install -r requirements.txt
```

### Install GroundingDINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install -e .

cd ..
```

---

## Training

```bash
python -m src.training.train
```

### Prompt Learning

```bash
python -m src.prompt_learning.train_coop

python -m src.prompt_learning.train_cocoop
```

### Evaluation

```bash
python -m src.evaluation.evaluate
```

---

## Core Components

| Module | Description |
|---|---|
| `detection/` | GroundingDINO-based open-vocabulary detection |
| `prompt_learning/` | CoOp and CoCoOp prompt optimization |
| `training/` | Semi-supervised and hybrid training pipelines |
| `evaluation/` | Localization and grounding evaluation |
| `utils/` | Shared utilities and experiment helpers |

---

## Research Focus

- Vision-language representation learning for mammography
- Prompt-conditioned grounding
- Open-vocabulary medical detection
- Semi-supervised lesion localization
- Robust feature alignment under limited annotation settings

---

<div align="center">

Research-oriented computer vision repository for vision-language mammography grounding.

</div>
