#  MedXAI

[![PyPI Version](https://img.shields.io/pypi/v/medxai.svg)](https://pypi.org/project/medxai/)
[![Python](https://img.shields.io/pypi/pyversions/medxai.svg)](https://pypi.org/project/medxai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MedXAI** is a lightweight, production-ready Python toolkit for Medical Imaging AI research. It provides efficient implementations of segmentation metrics, loss functions, preprocessing utilities, radiomics feature extraction, and Explainable AI (XAI), enabling researchers to build reliable and reproducible deep learning pipelines.

Designed with simplicity, efficiency, and extensibility in mind, MedXAI seamlessly integrates with **PyTorch**, **NumPy**, and **OpenCV**, making it suitable for both academic research and real-world medical imaging applications.

---

#  Features

###  Segmentation Metrics
- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- GPU & CPU compatible
- Automatic tensor shape validation

---

###  Advanced Loss Functions
- Dice Loss
- BCE + Dice Hybrid Loss
- Focal Tversky Loss
- Memory-efficient implementations

---

###  Medical Image Preprocessing
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- ROI Cropping
- Min-Max Normalization
- Multi-channel image support

---

###  Radiomics Feature Extraction
- Gray Level Co-occurrence Matrix (GLCM)
- Local Binary Pattern (LBP)
- Texture feature computation
- Optimized NumPy implementation

---

###  Explainable AI (XAI)
- Grad-CAM visualization
- Automatic hook management
- CUDA memory-safe implementation
- Compatible with custom PyTorch models

---

#  Installation

Install the latest stable version from PyPI.

```bash
pip install medxai
```

Or install the latest development version from GitHub.

```bash
pip install git+https://github.com/yourusername/medxai.git
```

---

#  Quick Start

## Compute Dice Score

```python
import torch
from medxai.metrics import dice_score

pred = torch.ones(1, 1, 256, 256)
target = torch.ones(1, 1, 256, 256)

score = dice_score(pred, target)

print(score.item())
```

---

## Extract GLCM Radiomics Features

```python
import cv2
from medxai.radiomics import extract_glcm

image = cv2.imread(
    "ultrasound_sample.png",
    cv2.IMREAD_GRAYSCALE
)

features = extract_glcm(image)

print(features)
```

---

## Generate Grad-CAM

```python
from medxai.explain import GradCAM

cam = GradCAM(
    model=model,
    target_layer=model.encoder.layer4
)

heatmap = cam.generate(
    input_tensor=image_tensor,
    class_idx=1
)

cam.remove_hooks()
```

---

## Run a 2D U-Net Forward Pass

```python
import torch
from medxai.models import UNet

# Initialize a U-Net for single-channel (e.g., X-ray) binary segmentation
model = UNet(n_channels=1, n_classes=1)
dummy_x = torch.randn(1, 1, 256, 256)
logits = model(dummy_x)

print(logits.shape)
```

---

## Clean a Noisy Prediction Mask

```python
import numpy as np
from medxai.postprocessing import clean_noisy_mask

# A binary mask with small false-positive noise
noisy_mask = np.zeros((100, 100), dtype=bool)
noisy_mask[40:60, 40:60] = True   # main region
noisy_mask[5:7, 5:7] = True       # small noise artifact

cleaned_mask = clean_noisy_mask(noisy_mask, min_size=64)

print(cleaned_mask.sum())  # noise removed, only main region remains
```
---

#  Module Overview

| Module | Description |
|---------|-------------|
| `medxai.metrics` | Segmentation evaluation metrics |
| `medxai.losses` | Medical segmentation loss functions |
| `medxai.preprocessing` | Image preprocessing utilities |
| `medxai.radiomics` | Radiomics feature extraction |
| `medxai.explain` | Explainable AI (Grad-CAM) |

---

#  Requirements

- Python ≥ 3.9
- PyTorch
- NumPy
- OpenCV
- scikit-image

---

#  Example Applications

MedXAI can be used in:

- Brain Tumor Segmentation
- Thyroid Nodule Analysis
- Breast Ultrasound
- Skin Lesion Classification
- Lung CT Analysis
- Retinal Disease Detection
- Histopathology
- General Medical Image Analysis

---

#  Contributing

Contributions are welcome!

If you would like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

Please ensure that new features include appropriate tests and documentation.

---

#  Citation

If you use **MedXAI** in your research, please cite:

```bibtex
@software{rahman2026medxai,
  author       = {Mohammad Amanour Rahman},
  title        = {MedXAI: A Lightweight Python Toolkit for Medical Imaging AI},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/aman0311x/medxai}
}
```

---

#  Support

If you find **MedXAI** useful in your research, please consider:

-  Starring the repository
-  Reporting bugs
-  Suggesting new features
-  Contributing to the project

Your support helps improve the toolkit for the medical AI research community.
