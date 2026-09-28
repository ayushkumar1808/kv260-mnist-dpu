"""PyTorch equivalent of the ONNX Model Zoo MNIST-12 network."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn


class Mnist12(nn.Module):
    """Two-convolution MNIST-12 graph with the original ONNX weights."""

    def __init__(self, weights_path: Union[str, Path]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc = nn.Linear(16 * 4 * 4, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self._load_onnx_weights(Path(weights_path))

    def _load_onnx_weights(self, weights_path: Path) -> None:
        weights = np.load(weights_path)
        tensors = {
            "conv1.weight": weights["conv1_weight"],
            "conv1.bias": weights["conv1_bias"],
            "conv2.weight": weights["conv2_weight"],
            "conv2.bias": weights["conv2_bias"],
            "fc.weight": weights["fc_weight"],
            "fc.bias": weights["fc_bias"],
        }
        with torch.no_grad():
            for name, value in tensors.items():
                parameter = dict(self.named_parameters())[name]
                parameter.copy_(torch.from_numpy(value))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.pool1(self.relu1(self.conv1(image)))
        image = self.pool2(self.relu2(self.conv2(image)))
        image = torch.flatten(image, 1)
        return self.fc(image)
