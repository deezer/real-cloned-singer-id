from typing import Any

import torch
from torchinfo import summary

from training.parameters import FEATURE_DIM, HIDDEN_DIM


class Classifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.nn.functional.normalize(x, dim=1)


class EvaluationWrapper(torch.nn.Module):
    """
    Wrapper module for model and projector.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int) -> None:
        """
        Check Classifier object args for details.
        """
        super().__init__()
        self.classifier = Classifier(input_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Any:
        return self.classifier(x)


if __name__ == "__main__":
    model = EvaluationWrapper(FEATURE_DIM, HIDDEN_DIM, 200).cuda()
    summary(model.eval(), input_size=(128, FEATURE_DIM))
