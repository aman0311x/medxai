from typing import Dict, List

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy


def extract_glcm(
    image: np.ndarray, distances: List[int] = [1], angles: List[float] = [0.0]
) -> Dict[str, float]:
    """Safely extracts averaged GLCM features from an image."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        # Scale to uint8 safely
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (img_norm * 255).astype(np.uint8)

    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "ASM": float(graycoprops(glcm, "ASM").mean()),
    }


def extract_lbp_hist(
    image: np.ndarray, radius: int = 1, n_points: int = 8
) -> np.ndarray:
    """Extracts normalized Local Binary Pattern histogram."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True
    )
    return hist
