# medxai 🚀

[![PyPI version](https://img.shields.io/pypi/v/medxai.svg)](https://pypi.org/project/medxai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**medxai** is a production-grade, lightweight, and highly optimized Python toolkit tailored for Medical Imaging AI research. It bridges the gap between raw medical scans, deep learning models, radiomics extraction, and explainable AI (XAI).

## ✨ Key Features

- **Device-Agnostic Metrics**: Robust implementation of Dice, IoU, Precision, and Recall supporting both CPU and GPU tensors with built-in shape validation.
- **Advanced Segmentation Losses**: Memory-optimized loss functions including Focal Tversky Loss and Hybrid BCE-Dice Loss.
- **Medical Preprocessing**: Safe implementations of CLAHE (multi-channel adaptive equalization), ROI cropping, and min-max normalization.
- **Radiomics Extraction**: High-fidelity GLCM texture feature analysis and uniform LBP pattern extractions.
- **Explainable AI (XAI)**: Memory-safe GradCAM implementation featuring automated hook cleanups to avoid CUDA memory leaks.

## 📦 Installation

Install `medxai` via PyPI:

```bash
pip install medxai

## ⚡ Quick Start

### 1. Compute Segmentation Metrics (GPU Supported)
```python
import torch
from medxai.metrics import dice_score

# Works perfectly on CUDA or CPU
pred = torch.ones(1, 1, 256, 256)
target = torch.ones(1, 1, 256, 256)

score = dice_score(pred, target)
print(f"Dice Coefficient: {score.item()}")

import cv2
from medxai.radiomics import extract_glcm

image = cv2.imread("ultrasound_sample.png", cv2.IMREAD_GRAYSCALE)
features = extract_glcm(image)
print(f"Contrast: {features['contrast']}, Homogeneity: {features['homogeneity']}")

from medxai.explain import GradCAM

# Initialize GradCAM safely
cam_loader = GradCAM(model=your_segmentation_model, target_layer=your_model.encoder.layer4)
heatmap = cam_loader.generate(input_tensor=scan_tensor, class_idx=1)

# Clean up hooks afterward to prevent memory leak
cam_loader.remove_hooks()

## 🤝 Citation

If you find this toolkit useful in your medical imaging research, please consider citing it using the following BibTeX entry:

```bibtex
@software{rahman2026medxai,
  author       = {Rahman, Mohammad Amanour},
  title        = {medxai: A Production-Grade Medical Imaging AI Toolkit},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/yourusername/medxai](https://github.com/yourusername/medxai)}}
}