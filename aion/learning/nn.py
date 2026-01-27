"""
AION Neural Network Building Blocks

Minimal NumPy-based MLP for use by the value function (critic),
RND curiosity module, and other components that need differentiable
function approximation without a PyTorch/TensorFlow dependency.

Placed in aion.learning.nn (no package __init__.py dependencies)
to avoid circular imports between policies and rewards subpackages.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class MLP:
    """Minimal two-layer MLP implemented in NumPy.

    Architecture: x -> Linear(d, h) -> ReLU -> Linear(h, out)

    Supports forward pass and manual gradient updates with
    gradient clipping for training stability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier/Glorot initialisation
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))

        self.W1 = np.random.uniform(-limit1, limit1, (hidden_dim, input_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = np.random.uniform(-limit2, limit2, (output_dim, hidden_dim)).astype(np.float64)
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (input_dim,) or (batch, input_dim)."""
        x = np.atleast_2d(x).astype(np.float64)
        self._last_x = x
        self._z1 = x @ self.W1.T + self.b1              # (batch, hidden)
        self._h1 = np.maximum(0, self._z1)               # ReLU
        out = self._h1 @ self.W2.T + self.b2              # (batch, output)
        return out

    def backward(self, d_out: np.ndarray, clip: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute gradients via backpropagation.

        Args:
            d_out: gradient of loss w.r.t. output, shape (batch, output_dim)
            clip: gradient clipping threshold

        Returns:
            Dictionary of parameter gradients.
        """
        d_out = np.atleast_2d(d_out).astype(np.float64)
        batch = d_out.shape[0]

        # Layer 2 gradients
        dW2 = d_out.T @ self._h1 / batch              # (output, hidden)
        db2 = d_out.mean(axis=0)                       # (output,)

        # Backprop through layer 2
        d_h1 = d_out @ self.W2                         # (batch, hidden)

        # ReLU backward
        d_z1 = d_h1 * (self._z1 > 0).astype(np.float64)  # (batch, hidden)

        # Layer 1 gradients
        dW1 = d_z1.T @ self._last_x / batch           # (hidden, input)
        db1 = d_z1.mean(axis=0)                        # (hidden,)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        # Global gradient clipping
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > clip:
            scale = clip / (total_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}

        return grads

    def apply_gradients(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        """Apply gradients with learning rate."""
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()

    def copy(self) -> "MLP":
        clone = MLP(self.input_dim, self.hidden_dim, self.output_dim)
        clone.set_params(self.get_params())
        return clone
