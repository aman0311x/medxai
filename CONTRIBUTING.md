# Contributing to MedXAI 🩺

First of all, thank you for your interest in contributing to MedXAI!

MedXAI is an open-source toolkit developed to support the Medical Imaging AI community by providing reliable, efficient, and easy-to-use utilities for deep learning, radiomics, preprocessing, and explainable AI.

Whether you are fixing bugs, improving documentation, adding new algorithms, or proposing new ideas, every contribution is appreciated.

---

# Table of Contents

- Code of Conduct
- Ways to Contribute
- Reporting Bugs
- Suggesting New Features
- Development Setup
- Contribution Workflow
- Coding Standards
- Running Tests
- Pull Request Guidelines
- Documentation
- Community

---

# Code of Conduct

By participating in this project, you agree to foster a respectful, inclusive, and welcoming environment.

Please:

- Be respectful.
- Be constructive.
- Be professional.
- Welcome different viewpoints.
- Help others learn.

---

# Ways to Contribute

There are many ways to contribute to MedXAI.

### 🐞 Report Bugs

If you encounter a bug:

1. Search existing Issues first.
2. If the bug has not been reported, create a new Issue.
3. Include:

- Clear title
- Description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version
- PyTorch version
- Operating System
- Package version

---

### 💡 Suggest Features

New ideas are always welcome.

Useful examples include:

- New segmentation losses
- Medical image preprocessing
- Explainable AI methods
- Radiomics features
- Visualization tools
- Benchmark datasets
- Performance improvements

Please explain:

- Why the feature is useful
- Relevant research papers (if applicable)
- Expected API design

---

### 📚 Improve Documentation

Documentation improvements are valuable.

Examples:

- Fix typos
- Improve explanations
- Add examples
- Improve tutorials
- Update API documentation

---

# Development Setup

## 1. Fork the Repository

Click the **Fork** button on GitHub.

---

## 2. Clone Your Fork

```bash
git clone https://github.com/aman0311x/medxai.git

cd medxai
```

---

## 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -U pip

pip install -r requirements.txt

pip install pytest black isort flake8
```

---

# Contribution Workflow

Create a new feature branch.

```bash
git checkout -b feature/my-new-feature
```

Make your changes.

Run formatting.

```bash
black .

isort .
```

Run tests.

```bash
pytest tests/ -v
```

Commit your changes.

We recommend following Conventional Commits.

Examples:

```text
feat: add generalized dice loss

fix: resolve gradcam memory leak

docs: improve installation guide

refactor: simplify preprocessing module

test: add unit tests for metrics
```

Push your branch.

```bash
git push origin feature/my-new-feature
```

Finally, open a Pull Request.

---

# Coding Standards

Please follow these guidelines.

## Code Style

- Follow PEP 8
- Maximum line length: 88 characters
- Use Black formatting
- Use isort for imports

---

## Type Hints

All public functions should include Python type hints.

Example

```python
def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
```

---

## Documentation

Use Google-style docstrings.

Example

```python
def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image into the range [0, 1].

    Args:
        image:
            Input image.

    Returns:
        Normalized image.
    """
```

---

# Testing

Before submitting a Pull Request, ensure:

- All tests pass.
- No warnings are introduced.
- Existing functionality remains unchanged.

Run:

```bash
pytest tests/ -v
```

If you introduce a new feature, please include corresponding unit tests whenever possible.

---

# Pull Request Checklist

Before submitting your PR, verify:

- Code builds successfully
- Tests pass
- Documentation updated
- Examples updated (if needed)
- New functionality tested
- No unnecessary files included

---

# Documentation Contributions

Documentation improvements are encouraged.

Examples include:

- Better examples
- API documentation
- Tutorials
- Diagrams
- README improvements

---

# Community

If you have questions or ideas, feel free to open a GitHub Issue.

We appreciate every contribution, regardless of size.

Thank you for helping improve MedXAI and supporting the Medical AI research community.

Happy Coding!

🚀