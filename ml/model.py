"""PyTorch model for binary readmission prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiabetesNet(nn.Module):
    def __init__(self, n_features: int = 20, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_model(n_features: int = 20) -> DiabetesNet:
    return DiabetesNet(n_features=n_features)


def get_parameters(model: nn.Module) -> list:
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    import numpy as np
    state = model.state_dict()
    for (k, _), p in zip(state.items(), parameters):
        state[k] = torch.tensor(p)
    model.load_state_dict(state, strict=True)