import torch.nn as nn

from models import utils
from timm.models.layers import DropPath


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, hidden_dims, num_class, normalization='BatchNorm1d', activation='ReLU', dropblock=True):
        super().__init__()

        normalization = getattr(nn, normalization)
        activation = getattr(nn, activation)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            DropPath(0.1) if dropblock else nn.Identity(),

            nn.Linear(in_features, hidden_dims),
            normalization(hidden_dims),
            activation(),

            nn.Linear(hidden_dims, num_class),
        )

        self.apply(utils.init_weights)

    def forward(self, feat):

        return self.classifier(feat)
