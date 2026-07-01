from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
import cv2

class GradCAM:
    """Memory-safe, device-agnostic GradCAM implementation for PyTorch models."""
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check your target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        
        # Avoid division by zero
        denom = cam.max() - cam.min()
        return (cam - cam.min()) / (denom + 1e-8) if denom > 0 else np.zeros_like(cam)

    def overlay(self, cam: np.ndarray, image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlays the heatmap onto original image."""
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)

    def remove_hooks(self) -> None:
        """Removes PyTorch hooks to avoid memory leak."""
        for handler in self.handlers:
            handler.remove()