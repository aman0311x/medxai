# MedXAI

<p align="center">
  <img src="logo.svg" width="180" alt="MedXAI Logo">
</p>

<p align="center">

<strong>A Lightweight, Modular, and Production-Ready Python Toolkit for Medical Imaging AI</strong>

</p>

<p align="center">

<a href="https://pypi.org/project/medxai/">
  <img src="https://img.shields.io/pypi/v/medxai.svg" alt="PyPI Version">
</a>

<a href="https://pypi.org/project/medxai/">
  <img src="https://img.shields.io/pypi/pyversions/medxai.svg" alt="Python Versions">
</a>

<a href="https://github.com/aman0311x/medxai/blob/main/LICENSE">
  <img src="https://img.shields.io/github/license/aman0311x/medxai" alt="License">
</a>

<a href="https://github.com/aman0311x/medxai/actions">
  <img src="https://img.shields.io/github/actions/workflow/status/aman0311x/medxai/python-package.yml?branch=main" alt="Build Status">
</a>

<a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Code Style">
</a>

<a href="#">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-blue" alt="Platform">
</a>

<a href="https://github.com/aman0311x/medxai/stargazers">
  <img src="https://img.shields.io/github/stars/aman0311x/medxai" alt="GitHub Stars">
</a>

<a href="https://github.com/aman0311x/medxai/network/members">
  <img src="https://img.shields.io/github/forks/aman0311x/medxai" alt="GitHub Forks">
</a>

<a href="https://github.com/aman0311x/medxai/issues">
  <img src="https://img.shields.io/github/issues/aman0311x/medxai" alt="GitHub Issues">
</a>

<a href="https://pepy.tech/project/medxai">
  <img src="https://static.pepy.tech/badge/medxai" alt="Downloads">
</a>

<!-- Enable after Codecov is configured -->
<!--
<a href="https://codecov.io/gh/aman0311x/medxai">
  <img src="https://img.shields.io/codecov/c/github/aman0311x/medxai" alt="Coverage">
</a>
-->

</p>

---

# Overview

**MedXAI** is an open-source Python library that simplifies the development of modern Medical Imaging Artificial Intelligence (Medical AI) applications. The library provides a carefully designed collection of reusable components for medical image preprocessing, segmentation evaluation, loss functions, radiomics analysis, explainable AI, and post-processing utilities.

The primary goal of MedXAI is to reduce the amount of boilerplate code researchers repeatedly implement across projects while promoting reproducibility, modularity, and clean software engineering practices.

Whether you are building a segmentation model for ultrasound, CT, MRI, retinal imaging, histopathology, or dermoscopy, MedXAI provides standardized implementations of commonly used algorithms so researchers can focus on model development instead of repeatedly rewriting utility code.

Unlike many research repositories that contain task-specific implementations, MedXAI is designed as a general-purpose toolkit that integrates naturally into existing PyTorch pipelines while remaining lightweight, dependency-conscious, and easy to extend.

The library emphasizes

- Reproducible research
- Modular software design
- High-performance implementations
- CPU and CUDA compatibility
- Clean APIs
- Easy integration into existing projects
- Research-friendly documentation
- Production-ready code quality

---

# Features

## Segmentation Metrics

- Dice Coefficient
- IoU (Jaccard Index)
- Precision
- Recall
- Pixel Accuracy
- Automatic tensor validation
- Batch-wise evaluation
- GPU acceleration

---

## Loss Functions

- Dice Loss
- BCE + Dice Loss
- Focal Tversky Loss
- Hybrid losses
- Stable numerical implementations
- Memory-efficient computation

---

## Medical Image Preprocessing

- CLAHE
- ROI Cropping
- Image Normalization
- Contrast Enhancement
- Multi-channel support
- Batch preprocessing utilities

---

## Radiomics

- Gray-Level Co-occurrence Matrix (GLCM)
- Local Binary Pattern (LBP)
- Texture Statistics
- NumPy optimized implementation
- Lightweight feature extraction

---

## Explainable AI

- Grad-CAM
- Automatic Hook Management
- GPU-safe implementation
- Model-agnostic interface
- Visualization utilities

---

## Post-processing

- Binary Mask Cleaning
- Connected Component Filtering
- Small Object Removal
- Morphological Utilities

---

# Project Structure

```
medxai/

├── metrics/
├── losses/
├── preprocessing/
├── radiomics/
├── explain/
├── postprocessing/
├── models/
├── utils/
├── tests/
├── docs/
└── examples/
```

---

# Installation

## Install from PyPI

```bash
pip install medxai
```

---

## Install Latest Development Version

```bash
pip install git+https://github.com/aman0311x/medxai.git
```

---

## Clone Repository

```bash
git clone https://github.com/aman0311x/medxai.git

cd medxai
```

---

# Requirements

- Python 3.9+
- PyTorch
- NumPy
- OpenCV
- scikit-image
- matplotlib

---

# Development Setup

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install MedXAI in editable mode.

```bash
pip install -e .
```

Install development dependencies.

```bash
pip install -r requirements-dev.txt
```

---

# Running Tests

Execute all unit tests.

```bash
pytest
```

Run with coverage.

```bash
pytest --cov=medxai
```

---

# Documentation

Comprehensive documentation, API references, and tutorials are available in the **docs/** directory.

Future online documentation will be available via GitHub Pages.

---

# Roadmap

Upcoming releases aim to include

- MONAI interoperability
- 3D medical image support
- Transformer-based utilities
- Diffusion model utilities
- Vision-Language Model helpers
- Additional radiomics descriptors
- Medical foundation model interfaces
- DICOM utilities
- ONNX export
- Benchmark datasets
- More XAI algorithms
- Clinical evaluation metrics

---

# Contributing

We welcome contributions from researchers, students, clinicians, and open-source developers.

There are many ways you can contribute to MedXAI:

- Report bugs
- Improve documentation
- Suggest new features
- Improve existing implementations
- Add unit tests
- Optimize performance
- Add new Medical AI utilities
- Improve API consistency
- Fix typos
- Review pull requests

## Contribution Workflow

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/my-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature/my-feature
```

5. Open a Pull Request.

Please ensure that all new code

- follows PEP8
- includes documentation
- passes all tests
- maintains backward compatibility where possible

We appreciate every contribution, regardless of size.

---

# Code of Conduct

Please be respectful and constructive when participating in discussions, reporting issues, or contributing code.

By participating in this project, you agree to foster a welcoming and inclusive community.

---

# Security

If you discover a security vulnerability, please do **not** create a public issue.

Instead, contact the maintainer directly so the issue can be responsibly addressed before public disclosure.

---

# License

This project is distributed under the **MIT License**.

You are free to

- use
- modify
- distribute
- sublicense
- include in commercial projects

provided that the original copyright notice and license are retained.

See the **LICENSE** file for complete details.

---

# Citation

If MedXAI contributes to your research, software, or publication, please cite the project.

```bibtex
@software{rahman2026medxai,
  author       = {Mohammad Amanour Rahman},
  title        = {MedXAI: A Lightweight Python Toolkit for Medical Imaging AI},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/aman0311x/medxai},
  license      = {MIT}
}
```



# Support the Project

If MedXAI has been useful in your research or projects, consider supporting the project by

⭐ Starring the repository

🐛 Reporting bugs

💡 Suggesting new features

📝 Improving documentation

🤝 Contributing code

📢 Sharing the project with the Medical AI community

Every contribution helps make MedXAI better for researchers worldwide.

---
